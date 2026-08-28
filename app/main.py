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

app = FastAPI(title="Minimax-H3-lipsync-mv", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mv-job")

IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


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
    result = {"ok": True, "h3_gateway": settings.h3_gateway_url, "llms": {}}
    try:
        result["llms"]["scenario"] = {
            "ok": True,
            "models": check_models(settings.scenario_llm_url),
        }
    except LLMError as exc:
        result["llms"]["scenario"] = {"ok": False, "error": str(exc)}
    return result


@app.post("/api/jobs", status_code=202)
async def create_job(
    character: Annotated[UploadFile, File()],
    song: Annotated[UploadFile, File()],
    concept: Annotated[str, Form()] = "",
) -> dict:
    character_type = (character.content_type or mimetypes.guess_type(character.filename or "")[0] or "")
    if character_type not in IMAGE_TYPES:
        raise HTTPException(415, "キャラクター画像はPNG、JPEG、WebPに対応しています")
    song_suffix = Path(song.filename or "").suffix.lower()
    if song_suffix not in AUDIO_SUFFIXES:
        raise HTTPException(415, "楽曲形式はWAV、MP3、FLAC、M4A、AAC、OGGに対応しています")
    image_name = _safe_filename(character.filename or "character.png", "character")
    song_name = _safe_filename(song.filename or "song.wav", "song")
    job = store.create(image_name, song_name, concept)
    input_dir = store.job_dir(job.id) / "input"
    try:
        await _save_upload(character, input_dir / image_name, settings.max_image_bytes)
        await _save_upload(song, input_dir / song_name, settings.max_audio_bytes)
    except Exception:
        store.update(job.id, status="failed", stage="upload", error="アップロードに失敗しました")
        raise
    finally:
        await character.close()
        await song.close()
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
    filename = f"{Path(song_name).stem}_H3_MV.mp4"
    return FileResponse(path, media_type="video/mp4", filename=filename)
