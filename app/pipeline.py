from __future__ import annotations

from typing import Any

import hashlib
import json
import time
import traceback
from pathlib import Path

from .config import settings
from .job_store import JobStore, store
from .services.audio import (
    AudioError,
    analyze_boundaries,
    ensure_media_tools,
    extract_scene_audio,
    plan_scenes,
    probe_duration,
    quantize_scene_plan,
    separate_vocals,
)
from .services.h3 import H3Client, H3Error, concatenate_and_mux
from .services.llm import LLMError, analyze_character, frame_kind, generate_scenario
from .services.tts import TTSError, synthesize_text


class Cancelled(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cancelled(job_id: str, jobs: JobStore) -> bool:
    job = jobs.get(job_id)
    return bool(job is None or job.cancel_requested)


def _check_cancel(job_id: str, jobs: JobStore) -> None:
    if _cancelled(job_id, jobs):
        raise Cancelled("ユーザーによりキャンセルされました")


def _stage(
    job_id: str, jobs: JobStore, stage: str, progress: float, message: str
) -> None:
    jobs.begin_timing(job_id, stage, message)
    jobs.update(job_id, stage=stage, progress=progress, message=message)
    jobs.log(job_id, message)


def run_pipeline(job_id: str, jobs: JobStore = store) -> None:
    job = jobs.get(job_id)
    if job is None:
        return
    job_dir = jobs.job_dir(job_id)
    source_dir = job_dir / "input"
    analysis_dir = job_dir / "analysis"
    scene_dir = job_dir / "scenes"
    source_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)
    image = source_dir / job.character_file
    song = source_dir / job.song_file

    try:
        jobs.start_run(job_id)
        jobs.update(job_id, status="running")
        ensure_media_tools()
        if job.input_mode == "narration":
            _stage(job_id, jobs, "synthesizing", 0.02, "AivisSpeechで文ごとに音声を合成しています")
            sentence_timings = synthesize_text(job.source_text, song)
            (analysis_dir / "sentence_timings.json").write_text(
                json.dumps(sentence_timings, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            jobs.log(job_id, f"{len(sentence_timings)}文の読み上げ音声を合成しました")
            _check_cancel(job_id, jobs)

        _stage(job_id, jobs, "probing", 0.04, "入力ファイルを検証しています")
        duration = probe_duration(song)
        if duration <= 0:
            raise AudioError("音声の長さが不正です")
        if duration > settings.max_song_seconds:
            raise AudioError(
                f"音声が上限を超えています: {duration:.1f}秒 > {settings.max_song_seconds:.0f}秒"
            )
        jobs.update(job_id, duration=round(duration, 3))
        _check_cancel(job_id, jobs)

        if job.input_mode == "narration":
            vocals = song
            jobs.log(job_id, "読み上げ音声をVocal Lockへ直接使用します")
        else:
            _stage(job_id, jobs, "separating", 0.06, "ボーカルを分離しています")
            vocals = separate_vocals(song, analysis_dir)
        _check_cancel(job_id, jobs)

        _stage(job_id, jobs, "analyzing_audio", 0.11, "息継ぎ・拍・セクションを解析しています")
        boundary_analysis = analyze_boundaries(song, vocals, analysis_dir)
        _check_cancel(job_id, jobs)

        scenes = quantize_scene_plan(plan_scenes(boundary_analysis), settings.fps)
        if job.input_mode == "narration":
            for scene in scenes:
                scene["narration"] = True
                scene["vocal_ratio"] = max(0.8, float(scene.get("vocal_ratio") or 0.0))
        (analysis_dir / "scene_plan.json").write_text(
            json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        jobs.update(job_id, scenes=scenes, scene_count=len(scenes))
        jobs.log(job_id, f"自然境界から{len(scenes)}シーンを計画しました")
        _check_cancel(job_id, jobs)

        _stage(job_id, jobs, "analyzing_character", 0.18, "参照画像から人物情報を固定しています")
        character = analyze_character(image)
        (analysis_dir / "character.json").write_text(
            json.dumps(character, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _stage(job_id, jobs, "planning", 0.22, "Gemma4-31Bで映像シナリオを作成しています")
        scenario = generate_scenario(
            scenes, job.concept, character,
            source_text=job.source_text if job.input_mode == "narration" else "",
            frame=frame_kind(job.width, job.height),
        )
        prompt_by_index = {
            int(item["index"]): item for item in scenario.get("scenes", [])
        }
        seed_base = int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16)
        for scene in scenes:
            prompt_data = prompt_by_index[int(scene["index"])]
            scene["prompt"] = str(prompt_data.get("prompt") or "").strip()
            scene["emotion"] = str(prompt_data.get("emotion") or scene.get("emotion") or "")
            scene["seed"] = (seed_base + int(scene["index"]) * 7919) % 2_147_483_647
        jobs.update(job_id, scenario=scenario, scenes=scenes)
        (analysis_dir / "scenario.json").write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _check_cancel(job_id, jobs)

        _stage(job_id, jobs, "loading_h3", 0.26, "MiniMax-H3を常駐構成で準備しています")
        h3 = H3Client()
        loaded = h3.ensure_loaded()
        if loaded.get("coresident"):
            # 同居時はプリセットが変わり1シーンあたり十数秒遅くなる。テキスト
            # エンコーダは 32B のままなので画質・追従は変わらないが、所要時間の
            # 見込みが変わるので黙らずに出しておく。
            _stage(
                job_id, jobs, "loading_h3", 0.27,
                f"{'・'.join(loaded.get('coresident_with') or [])}と同居のため"
                f"{loaded.get('preset_used')}で動作します(品質は同じ、生成は少し遅くなります)",
            )
        image_asset_id = h3.upload_asset(image)
        image_hash = _sha256(image)
        vocals_hash = _sha256(vocals)

        scene_files: list[Path] = []
        count = len(scenes)
        _stage(job_id, jobs, "generating", 0.28, "H3で各シーンを生成しています")

        # 先行投入: シーン N を投入(= 前シーンの denoise 完了待ち)してから、前シーン
        # N-1 の完了(decode・取得・整形)を待つ。decode を 2 枚目の GPU に回す構成では
        # 前シーンの decode と次シーンの denoise が並行し、連続生成の間隔が縮む。
        # 1 GPU 構成でも順序が変わるだけで結果は同じ。
        def finish(entry: dict[str, Any]) -> None:
            position = entry["position"]
            base = 0.28 + (position - 1) / count * 0.62
            span = 0.62 / count

            def scene_progress(
                value: float,
                state: str,
                base_progress: float = base,
                scene_span: float = span,
                scene_position: int = position,
            ) -> None:
                jobs.update(job_id, progress=min(0.9, base_progress + scene_span * value), message=(
                    f"H3シーン {scene_position} / {count}: {state} {value * 100:.0f}%"
                ))

            try:
                h3.finish_scene(
                    entry["job_id"],
                    destination=entry["output"],
                    target_seconds=float(entry["scene"]["duration"]),
                    width=job.width,
                    height=job.height,
                    fps=settings.fps,
                    on_progress=scene_progress,
                )
            except H3Error as exc:
                if _cancelled(job_id, jobs):
                    raise Cancelled("現在のH3シーン完了後にキャンセルしました") from exc
                raise
            finally:
                jobs.record_timing(
                    job_id,
                    "generating_scene",
                    f"H3シーン {position} / {count}",
                    time.monotonic() - entry["started"],
                    kind="scene",
                )
            entry["scene"]["video_file"] = entry["output"].name
            entry["metadata_path"].write_text(
                json.dumps(entry["metadata"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            jobs.update(job_id, scenes=scenes)

        pending: dict[str, Any] | None = None
        for position, scene in enumerate(scenes, start=1):
            _check_cancel(job_id, jobs)
            scene_started = time.monotonic()
            output = scene_dir / f"scene_{position:03d}.mp4"
            metadata_path = scene_dir / f"scene_{position:03d}.meta.json"
            scene_files.append(output)
            wav = scene_dir / f"scene_{position:03d}_vocals.wav"
            extract_scene_audio(
                vocals, float(scene["start"]), float(scene["duration"]), wav
            )
            generation_metadata = {
                "pipeline_revision": 4,
                "image_sha256": image_hash,
                "vocals_sha256": vocals_hash,
                "start": scene["start"],
                "duration": scene["duration"],
                "prompt": scene["prompt"],
                "seed": scene["seed"],
                "width": job.width,
                "height": job.height,
                "fps": settings.fps,
                "turbo": True,
                "vocal_lock": True,
                "reference_short_edge": settings.h3_ref_short_edge,
            }
            cached_metadata = None
            if metadata_path.is_file():
                try:
                    cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    cached_metadata = None
            if (
                output.is_file()
                and output.stat().st_size > 1024
                and cached_metadata == generation_metadata
            ):
                jobs.log(job_id, f"シーン{position}は同一設定で生成済みのため再利用します")
                jobs.record_timing(
                    job_id,
                    "generating_scene",
                    f"H3シーン {position} / {count}（キャッシュ再利用）",
                    time.monotonic() - scene_started,
                    kind="scene",
                )
                continue
            jobs.update(
                job_id,
                stage="generating",
                progress=0.28 + (position - 1) / count * 0.62,
                message=f"H3でシーン {position} / {count} を生成しています",
                current_scene=position,
            )
            jobs.log(job_id, f"シーン{position}: {scene['start']:.2f}–{scene['end']:.2f}秒")
            try:
                submitted = h3.submit_scene(
                    image_asset_id=image_asset_id,
                    audio_path=wav,
                    prompt=str(scene["prompt"]),
                    seed=int(scene["seed"]),
                    width=job.width,
                    height=job.height,
                )
            except H3Error as exc:
                if _cancelled(job_id, jobs):
                    raise Cancelled("現在のH3シーン完了後にキャンセルしました") from exc
                raise
            if pending is not None:
                finish(pending)
            pending = {
                "job_id": submitted, "position": position, "scene": scene,
                "output": output, "metadata_path": metadata_path,
                "metadata": generation_metadata, "started": scene_started,
            }
        if pending is not None:
            finish(pending)

        _check_cancel(job_id, jobs)
        _stage(job_id, jobs, "rendering", 0.93, "シーンを結合して音声を合成しています")
        output = job_dir / "output.mp4"
        concatenate_and_mux(scene_files, song, output, duration)
        jobs.finish_run(job_id)
        jobs.update(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            message="読み上げ動画が完成しました" if job.input_mode == "narration" else "MVが完成しました",
            output_file=output.name,
            current_scene=count,
        )
        jobs.log(job_id, "読み上げ動画が完成しました" if job.input_mode == "narration" else "MVが完成しました")
    except Cancelled as exc:
        jobs.finish_run(job_id)
        jobs.update(
            job_id, status="cancelled", stage="cancelled", message=str(exc), error=str(exc)
        )
        jobs.log(job_id, str(exc))
    except (AudioError, TTSError, LLMError, H3Error, OSError, ValueError) as exc:
        jobs.finish_run(job_id)
        jobs.update(
            job_id, status="failed", stage="failed", message="生成に失敗しました", error=str(exc)
        )
        jobs.log(job_id, f"エラー: {exc}")
    except Exception as exc:  # noqa: BLE001
        jobs.finish_run(job_id)
        jobs.update(
            job_id, status="failed", stage="failed", message="予期しないエラー", error=str(exc)
        )
        jobs.log(job_id, traceback.format_exc()[-2000:])
