from __future__ import annotations

import subprocess
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

    # MV 本番の既定。単独運用ではこれだけを渡す(テキストエンコーダは 32B のまま)。
    BASE_OVERRIDES = {
        "H3_REF_PREFIX_CACHE_SINGLE": "1",
        "H3_VOCAL_LOCK": "1",
        "H3_TURBO_LORA_FILE": "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    }

    # 同居(coresident)のときだけ使うプリセット。
    # 単独用の `96gb-int8` は ref2va ピークが 73.8GB あり LTX(常駐 33GB)と同じ GPU に
    # 載らないため、同居時だけ 48GB 級のフェーズ循環構成へ落とす。
    #
    # 当初は `96gb-int8` + 投影TE(H3_TE_PROJ)で収める案だったが、2026-09-15 の実測で
    # `48gb-lowvram` が優ると判明したため差し替えた(ref2va 768×448・5秒・turbo、
    # LTX 常駐中、いずれも連続実行の定常値):
    #
    #   48gb-lowvram     : 40.5秒 / GPU0 全体ピーク 74.5GiB(余裕21.0) / **32B TE のまま**
    #   96gb-int8+投影TE : 27.5秒 / GPU0 全体ピーク 83.2GiB(余裕12.4) / 投影4B TE
    #
    # denoise(14.4s)も decode(3.4s)もほぼ同一で、差は全部フェーズ循環の固定費。
    # 1本あたり13秒の代償で**品質劣化ゼロ + 余裕 8.6GiB 増**になる。MV は6セクション記法で
    # 細部を指定する品質重視のバッチ用途なので、投影TE の近似(PSNR 22.64dB / 鮮鋭度 -23%、
    # 「大意は保持し細部は失う」)を受け入れる理由がない。
    CORESIDENT_PRESET = "48gb-lowvram"

    # 96GB 級未満の GPU 用の既定プリセット(ck-w4a8 pruned + pinned 系)。
    # 2026-10-10 実測(RTX PRO 5000 48GB、1024×768・10.1s・243f・turbo):
    # peak 35.6GB / 定常 131s(denoise 98s + decode 21s)。96gb-int8 は同条件
    # peak 77.4GB で 48GB には載らない。
    SMALL_GPU_PRESET = "ref2va-only-32gb"
    # 24GB 級(RTX 4090 等)用。VAE をフェーズごとに退避する分だけ 32GB 用より常駐が小さい。
    TINY_GPU_PRESET = "ref2va-only-24gb"
    SMALL_GPU_PRESETS = (SMALL_GPU_PRESET, TINY_GPU_PRESET)
    # 32GB 用構成を選ぶ下限(RTX 5090 の総容量は約 31.8GiB)。
    SMALL_GPU_MIN_GB = 28.0
    # 10 秒シーンで収まる画素数の上限(GPU 1 枚・decode も同じ GPU、2026-10-10 実測):
    #   32GB 窓(空き 30GiB): 1024×576 = 589,824px で peak 28.5GiB(余裕 1.5GiB)
    #   24GB 窓(空き 23GiB):  768×448 = 344,064px で peak 19.9GiB(余裕 3GiB)
    # 40GB 以上は全サイズ可(48GB で 1344×768 が peak 33.6GB、GPU1 decode 時)。
    MAX_PIXELS_32GB = 1024 * 576
    MAX_PIXELS_24GB = 768 * 448
    # これ未満の GPU では LTX との同居(coresident)を試みない(48GB では LTX ~29GB と
    # H3 の同居が成立しないため、gateway に入れ替えさせる)。
    CORESIDENT_MIN_GPU_GB = 90.0

    def _gpu_total_gb(self) -> float | None:
        """計算 GPU(H3_GPUS の先頭)の総 VRAM(GB)。取得できなければ None。"""
        index = (settings.h3_gpus or "0").split(",")[0].strip() or "0"
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--id={index}", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            return float(out.splitlines()[0]) / 1024
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _gpu_indices(self) -> list[str]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
            return [line.strip() for line in out.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            return []

    def _decode_gpu(self) -> str | None:
        """decode を回す 2 枚目の GPU の物理 index(使わないなら None)。"""
        value = settings.h3_decode_gpu
        if value == "off":
            return None
        compute = (settings.h3_gpus or "0").split(",")[0].strip() or "0"
        if value == "auto":
            others = [i for i in self._gpu_indices() if i != compute]
            return others[0] if others else None
        return value if value != compute else None

    def _resolve_preset(self) -> str:
        if settings.h3_preset and settings.h3_preset != "auto":
            return settings.h3_preset
        total = self._gpu_total_gb() or 0
        if total >= self.CORESIDENT_MIN_GPU_GB:
            return "96gb-int8"
        if total >= self.SMALL_GPU_MIN_GB:
            return self.SMALL_GPU_PRESET
        return self.TINY_GPU_PRESET

    def capacity(self) -> dict[str, Any]:
        """UI 向け: この GPU で 10 秒シーンが収まる画素数の目安。"""
        total = self._gpu_total_gb()
        preset = self._resolve_preset()
        if (total or 0) >= 40.0:
            max_pixels = None
        elif (total or 0) >= self.SMALL_GPU_MIN_GB:
            max_pixels = self.MAX_PIXELS_32GB
        else:
            max_pixels = self.MAX_PIXELS_24GB
        return {
            "gpu_gb": round(total, 1) if total else None,
            "preset": preset,
            "decode_gpu": self._decode_gpu() if preset in self.SMALL_GPU_PRESETS else None,
            "max_pixels": max_pixels,
        }

    def _other_loaded_backends(self) -> list[str]:
        """gateway に重みを載せている h3 以外のバックエンド。"""
        status = self.status()
        loaded = status.get("loaded_backends")
        if loaded is None:  # coresident 対応前の gateway 互換
            active = status.get("active_backend")
            loaded = [active] if active else []
        return [name for name in loaded if name != "h3"]

    def ensure_loaded(self) -> dict[str, Any]:
        """H3 を常駐させる。他バックエンドがロード済みなら同居モードで載せる。

        同居時は strategy=coresident(相手の VRAM を奪わない)+ 48GB 級プリセット。
        単独時は従来どおり strategy=resident + 設定のプリセット。
        **どちらも 32B TE のままなので、テキスト条件付けの品質は変わらない。**
        """
        others = self._other_loaded_backends()
        total = self._gpu_total_gb()
        # 同居は 96GB 級でだけ試す。それ未満では resident で gateway に入れ替えさせる。
        coresident = bool(others) and (total or 0) >= self.CORESIDENT_MIN_GPU_GB
        preset = self.CORESIDENT_PRESET if coresident else self._resolve_preset()
        gpus = settings.h3_gpus
        overrides = dict(self.BASE_OVERRIDES)
        decode_gpu = self._decode_gpu() if preset in self.SMALL_GPU_PRESETS else None
        if decode_gpu is not None:
            # 計算 GPU を先頭、decode GPU を 2 枚目にして可視化し、cuda:1 で decode させる。
            # バックエンドは denoise 完了でロックを手放すので、次シーンを先に投入して
            # おけば decode と次の denoise が並行する(pipeline.py 側で先行投入する)。
            gpus = f"{gpus.split(',')[0]},{decode_gpu}"
            overrides["H3_DECODE_DEVICE"] = "cuda:1"
        payload = {
            "backend": "h3",
            "preset": preset,
            "gpus": gpus,
            "strategy": "coresident" if coresident else "resident",
            "overrides": overrides,
        }
        response = self._request("POST", "/api/v1/backend/load", json=payload, timeout=300)
        if not response.ok:
            raise H3Error(f"H3ロードに失敗しました: HTTP {response.status_code}: {response.text[:600]}")
        data = response.json()
        status = self.status()
        # coresident では status.process は「ロード済みの先頭1つ」しか表さないので、
        # backends[h3] を一次ソースにする(無い旧 gateway では process にフォールバック)。
        info = (status.get("backends") or {}).get("h3") or status.get("process") or {}
        if info.get("preset") != preset:
            raise H3Error(f"H3プリセットが一致しません: {info.get('preset')}(期待: {preset})")
        data["coresident"] = coresident
        data["coresident_with"] = others
        data["preset_used"] = preset
        data["text_encoder"] = "qwen3-vl-32b"  # 同居時も 32B のまま
        data["decode_gpu"] = decode_gpu
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

    def submit_scene(
        self,
        *,
        image_asset_id: str,
        audio_path: Path,
        prompt: str,
        seed: int,
        width: int = settings.width,
        height: int = settings.height,
        submit_timeout: float = 900.0,
    ) -> str:
        """シーンを投入し、バックエンドで着手した時点でジョブ ID を返す。

        前のシーンの denoise 中は gateway が 409(busy)を返すので、着手できるまで
        再試行する(前シーンが decode に入った瞬間に通る = decode と次の denoise が
        並行する)。着手確認まで待つのは、受理直後の数 ms の窓で次の投入が 409 ->
        failed になるのを防ぐため(realtime-narration-video と同じ方式)。
        """
        audio_asset_id = self.upload_asset(audio_path)
        extra: dict[str, Any] = {"turbo": True}
        if settings.h3_ref_short_edge > 0:
            extra["reference_image_short_edge"] = settings.h3_ref_short_edge
        body = {
            "backend": "h3",
            "mode": "ref2v",
            "params": {"prompt": prompt, "width": width, "height": height, "seed": seed},
            "asset_ids": [image_asset_id, audio_asset_id],
            "extra": extra,
            "auto_load": False,
        }
        deadline = time.monotonic() + submit_timeout
        while True:
            response = self._request("POST", "/api/v1/generate", json=body, timeout=60)
            if response.status_code == 409 and time.monotonic() < deadline:
                time.sleep(0.2)
                continue
            if not response.ok:
                raise H3Error(f"H3生成受付に失敗しました: HTTP {response.status_code}: {response.text[:800]}")
            break
        job_id = str(response.json().get("id") or "")
        if not job_id:
            raise H3Error("H3ジョブIDが返されませんでした")
        self._wait_started(job_id, deadline)
        return job_id

    def _wait_started(self, job_id: str, deadline: float) -> None:
        while time.monotonic() < deadline:
            try:
                busy = self._request("GET", "/h3/api/status", timeout=10)
                if busy.ok and busy.json().get("busy"):
                    return
                job = self._request("GET", f"/api/v1/jobs/{job_id}", timeout=10)
                if job.ok and job.json().get("status") in {"completed", "failed", "interrupted", "cancelled"}:
                    return  # 着手前に終わった/失敗した: finish_scene が結果を報告する
            except (H3Error, ValueError):
                pass
            time.sleep(0.05)
        raise H3Error("H3ジョブがバックエンドで開始されませんでした(タイムアウト)")

    def finish_scene(
        self,
        job_id: str,
        *,
        destination: Path,
        target_seconds: float,
        width: int = settings.width,
        height: int = settings.height,
        fps: int = settings.fps,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """投入済みジョブの完了を待ち、出力を取得してシーン尺に整える。"""
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
        target_frames = max(1, round(target_seconds * fps))
        run_command([
            "ffmpeg", "-y", "-v", "error", "-i", str(raw),
            "-an", "-frames:v", str(target_frames), "-r", str(fps),
            "-vf", f"scale={width}:{height}:flags=lanczos,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(destination),
        ])
        raw.unlink(missing_ok=True)
        return {"gateway_job_id": job_id, "result": result, "file": str(destination)}

    def generate_scene(
        self,
        *,
        image_asset_id: str,
        audio_path: Path,
        prompt: str,
        seed: int,
        destination: Path,
        target_seconds: float,
        width: int = settings.width,
        height: int = settings.height,
        fps: int = settings.fps,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """互換: 1 シーンを投入して完了まで待つ(先行投入しない呼び出し向け)。"""
        job_id = self.submit_scene(
            image_asset_id=image_asset_id, audio_path=audio_path, prompt=prompt,
            seed=seed, width=width, height=height,
        )
        return self.finish_scene(
            job_id, destination=destination, target_seconds=target_seconds,
            width=width, height=height, fps=fps, on_progress=on_progress,
        )


def concatenate_and_mux(
    scene_files: list[Path], song: Path, destination: Path, duration: float,
    subtitles_ass: Path | None = None,
) -> Path:
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
        # テロップを焼き込むときだけ映像を再エンコードする(それ以外はコピーのまま)。
        video_codec = ["-c:v", "copy"]
        if subtitles_ass is not None:
            # ass フィルタの引数はパス中の : や \ をエスケープする必要がある。
            ass_arg = str(subtitles_ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            video_codec = ["-vf", f"ass='{ass_arg}'", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "18", "-pix_fmt", "yuv420p"]
        run_command([
            "ffmpeg", "-y", "-v", "error", "-i", str(video_only), "-i", str(song),
            "-map", "0:v:0", "-map", "1:a:0", "-t", f"{duration:.3f}",
            *video_codec, "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart",
            str(destination),
        ])
    finally:
        concat_file.unlink(missing_ok=True)
        video_only.unlink(missing_ok=True)
    return destination
