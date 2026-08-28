from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from ..config import settings
from .audio import run_command


class H3Error(RuntimeError):
    pass


ProgressCallback = Callable[[float, str], None]


class H3Client:
    def __init__(self, base_url: str = settings.h3_gateway_url) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            response = requests.request(method, f"{self.base_url}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)
        except requests.RequestException as exc:
            raise H3Error(f"H3ゲートウェイに接続できません: {exc}") from exc
        return response

    def ensure_loaded(self) -> dict[str, Any]:
        payload = {
            "backend": "h3",
            "preset": settings.h3_preset,
            "gpus": settings.h3_gpus,
            "strategy": "resident",
            "overrides": {
                "H3_REF_PREFIX_CACHE_SINGLE": "1",
                "H3_VOCAL_LOCK": "1",
                "H3_TURBO_LORA_FILE": "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
            },
        }
        response = self._request("POST", "/api/v1/backend/load", json=payload, timeout=300)
        if not response.ok:
            raise H3Error(f"H3ロードに失敗しました: HTTP {response.status_code}: {response.text[:600]}")
        data = response.json()
        status = self.status()
        process = status.get("process") or {}
        if process.get("preset") != settings.h3_preset:
            raise H3Error(f"H3プリセットが一致しません: {process.get('preset')}")
        return data

    def status(self) -> dict[str, Any]:
        response = self._request("GET", "/api/v1/status", timeout=15)
        if not response.ok:
            raise H3Error(f"H3状態取得に失敗しました: HTTP {response.status_code}")
        return response.json()

    def upload_asset(self, path: Path) -> str:
        with path.open("rb") as stream:
            response = self._request(
                "POST", "/api/v1/assets",
                files={"file": (path.name, stream)}, timeout=300,
            )
        if not response.ok:
            raise H3Error(f"H3アセット登録に失敗しました: {response.text[:500]}")
        asset_id = str(response.json().get("id") or "")
        if not asset_id:
            raise H3Error("H3アセットIDが返されませんでした")
        return asset_id

    def generate_scene(
        self,
        *,
        image_asset_id: str,
        audio_path: Path,
        prompt: str,
        seed: int,
        destination: Path,
        target_seconds: float,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        audio_asset_id = self.upload_asset(audio_path)
        body = {
            "backend": "h3",
            "mode": "ref2v",
            "params": {
                "prompt": prompt,
                "width": settings.width,
                "height": settings.height,
                "seed": seed,
            },
            "asset_ids": [image_asset_id, audio_asset_id],
            "extra": {"turbo": True},
            "auto_load": False,
        }
        response = self._request("POST", "/api/v1/generate", json=body, timeout=60)
        if not response.ok:
            raise H3Error(f"H3生成受付に失敗しました: HTTP {response.status_code}: {response.text[:800]}")
        job_id = str(response.json().get("id") or "")
        if not job_id:
            raise H3Error("H3ジョブIDが返されませんでした")

        while True:
            job_response = self._request("GET", f"/api/v1/jobs/{job_id}", timeout=30)
            if not job_response.ok:
                raise H3Error(f"H3ジョブ取得に失敗しました: {job_response.text[:500]}")
            job = job_response.json()
            state = str(job.get("status") or "")
            progress = float(job.get("progress") or 0.0)
            if on_progress:
                on_progress(progress, state)
            if state == "completed":
                break
            if state in {"failed", "interrupted", "cancelled"}:
                raise H3Error(str(job.get("error") or f"H3ジョブが{state}になりました"))
            time.sleep(2)

        result = job.get("result") or {}
        video_url = str(result.get("video_url") or "")
        if not video_url:
            raise H3Error("H3出力URLがありません")
        download = requests.get(urljoin(f"{self.base_url}/", video_url.lstrip("/")), timeout=600)
        download.raise_for_status()
        raw = destination.with_name(f"{destination.stem}_raw.mp4")
        raw.write_bytes(download.content)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target_frames = max(1, round(target_seconds * settings.fps))
        run_command([
            "ffmpeg", "-y", "-v", "error", "-i", str(raw),
            "-an", "-frames:v", str(target_frames), "-r", str(settings.fps),
            "-vf", f"scale={settings.width}:{settings.height}:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(destination),
        ])
        raw.unlink(missing_ok=True)
        return {"gateway_job_id": job_id, "result": result, "file": str(destination)}


def concatenate_and_mux(scene_files: list[Path], song: Path, destination: Path, duration: float) -> Path:
    if not scene_files:
        raise H3Error("結合するシーン動画がありません")
    concat_file = destination.with_suffix(".concat.txt")
    video_only = destination.with_name(f"{destination.stem}_video.mp4")
    concat_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in scene_files),
        encoding="utf-8",
    )
    try:
        run_command([
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(video_only),
        ])
        run_command([
            "ffmpeg", "-y", "-v", "error", "-i", str(video_only), "-i", str(song),
            "-map", "0:v:0", "-map", "1:a:0", "-t", f"{duration:.3f}",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart",
            str(destination),
        ])
    finally:
        concat_file.unlink(missing_ok=True)
        video_only.unlink(missing_ok=True)
    return destination
