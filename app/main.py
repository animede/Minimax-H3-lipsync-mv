from __future__ import annotations

import asyncio
import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, settings
from .job_store import store
from .pipeline import run_pipeline
from .services.llm import LLMError, check_models
from .services.tts import TTSError, check_tts, read_text_file

app = FastAPI(title="Minimax-H3-lipsync-mv", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mv-job")

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
TEXT_SUFFIXES = {".txt"}


def _safe_filename(name: str, fallback: str) -> str:
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or fallback
    return f"{stem[:80]}{suffix}"


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> None:
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "ファイルサイズが上限を超えています")
            stream.write(chunk)


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    result = {"ok": True, "h3_gateway": settings.h3_gateway_url, "llms": {}, "tts": {}}
    try:
        result["llms"]["scenario"] = {
            "ok": True,
            "models": check_models(settings.scenario_llm_url),
        }
    except LLMError as exc:
        result["llms"]["scenario"] = {"ok": False, "error": str(exc)}
    try:
        info = check_tts()
        result["tts"] = {
            "ok": True,
            "version": info["version"],
            "speaker_id": settings.tts_speaker_id,
        }
    except TTSError as exc:
        result["tts"] = {"ok": False, "error": str(exc)}
    return result


@app.post("/api/jobs", status_code=202)
async def create_job(
    character: Annotated[UploadFile, File()],
    song: Annotated[UploadFile | None, File()] = None,
    text: Annotated[UploadFile | None, File()] = None,
    concept: Annotated[str, Form()] = "",
) -> dict:
    character_type = (character.content_type or mimetypes.guess_type(character.filename or "")[0] or "")
    if character_type not in IMAGE_TYPES:
        raise HTTPException(415, "キャラクター画像はPNG、JPEG、WebPに対応しています")
    if (song is None) == (text is None):
        raise HTTPException(422, "楽曲またはテキストのどちらか一方を指定してください")
    image_name = _safe_filename(character.filename or "character.png", "character")
    input_mode = "narration" if text is not None else "music"
    source_text = ""
    text_name = ""
    if song is not None:
        song_suffix = Path(song.filename or "").suffix.lower()
        if song_suffix not in AUDIO_SUFFIXES:
            raise HTTPException(415, "楽曲形式はWAV、MP3、FLAC、M4A、AAC、OGGに対応しています")
        song_name = _safe_filename(song.filename or "song.wav", "song")
    else:
        assert text is not None
        if Path(text.filename or "").suffix.lower() not in TEXT_SUFFIXES:
            raise HTTPException(415, "読み上げ原稿はTXT形式に対応しています")
        text_name = _safe_filename(text.filename or "script.txt", "script")
        song_name = "narration.wav"
    job = store.create(
        image_name, song_name, concept, input_mode=input_mode, text_file=text_name
    )
    input_dir = store.job_dir(job.id) / "input"
    try:
        await _save_upload(character, input_dir / image_name, settings.max_image_bytes)
        if song is not None:
            await _save_upload(song, input_dir / song_name, settings.max_audio_bytes)
        else:
            assert text is not None
            await _save_upload(text, input_dir / text_name, settings.max_text_bytes)
            source_text = read_text_file(input_dir / text_name)
            store.update(job.id, source_text=source_text)
    except Exception:
        store.update(job.id, status="failed", stage="upload", error="アップロードに失敗しました")
        raise
    finally:
        await character.close()
        if song is not None:
            await song.close()
        if text is not None:
            await text.close()
    executor.submit(run_pipeline, job.id)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": [job.to_dict() for job in store.list()[:50]]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "ジョブが見つかりません")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    if store.get(job_id) is None:
        raise HTTPException(404, "ジョブが見つかりません")

    async def stream():
        last = ""
        while True:
            job = store.get(job_id)
            if job is None:
                break
            data = json.dumps(job.to_dict(), ensure_ascii=False)
            if data != last:
                yield f"data: {data}\n\n"
                last = data
            if job.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "ジョブが見つかりません")
    if job.status not in {"queued", "running"}:
        return job.to_dict()
    store.update(job_id, cancel_requested=True, message="キャンセルを要求しました")
    return store.get(job_id).to_dict()


@app.post("/api/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "ジョブが見つかりません")
    if job.status not in {"failed", "cancelled"}:
        raise HTTPException(409, "失敗またはキャンセル済みのジョブだけ再開できます")
    store.update(
        job_id,
        status="queued",
        stage="queued",
        progress=0.0,
        message="再開待ち",
        error="",
        cancel_requested=False,
        timings=[],
        timing_stage="",
        timing_label="",
        timing_started_at=0.0,
        run_started_at=0.0,
        total_elapsed_s=0.0,
    )
    executor.submit(run_pipeline, job_id)
    return store.get(job_id).to_dict()


def _output(job_id: str) -> tuple[Path, str]:
    job = store.get(job_id)
    if job is None or job.status != "completed" or not job.output_file:
        raise HTTPException(404, "完成動画がありません")
    path = (store.job_dir(job_id) / job.output_file).resolve()
    if path.parent != store.job_dir(job_id).resolve() or not path.is_file():
        raise HTTPException(404, "完成動画がありません")
    return path, job.song_file


@app.get("/api/jobs/{job_id}/output")
def view_output(job_id: str) -> FileResponse:
    path, _ = _output(job_id)
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/download")
def download_output(job_id: str) -> FileResponse:
    path, song_name = _output(job_id)
    job = store.get(job_id)
    assert job is not None
    filename = f"{Path(song_name).stem}_H3_{'Talk' if job.input_mode == 'narration' else 'MV'}.mp4"
    return FileResponse(path, media_type="video/mp4", filename=filename)
