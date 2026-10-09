from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("APP_HOST", "127.0.0.1")
    port: int = int(os.getenv("APP_PORT", "8780"))
    max_song_seconds: float = float(os.getenv("APP_MAX_SONG_SECONDS", "300"))
    retention_days: int = int(os.getenv("APP_JOB_RETENTION_DAYS", "7"))
    max_image_bytes: int = 20 * 1024 * 1024
    max_audio_bytes: int = 500 * 1024 * 1024
    max_text_bytes: int = 2 * 1024 * 1024

    jobs_dir: Path = ROOT / "data" / "jobs"
    scenario_llm_url: str = os.getenv(
        "SCENARIO_LLM_URL", "http://127.0.0.1:64650/v1"
    ).rstrip("/")
    scenario_llm_model: str = os.getenv(
        "SCENARIO_LLM_MODEL", "your-model-id"
    )
    llm_api_key: str = os.getenv("LLM_API_KEY", "")

    stem_api_url: str = os.getenv("STEM_API_URL", "http://127.0.0.1:8889").rstrip("/")
    demucs_python: str = os.getenv("DEMUCS_PYTHON", "python3")
    h3_gateway_url: str = os.getenv("H3_GATEWAY_URL", "http://127.0.0.1:8630").rstrip("/")
    # "auto"(既定)= 計算 GPU の VRAM で選ぶ: 90GB 以上なら 96gb-int8(従来)、
    # それ未満なら ck-w4a8 系の ref2va-only-32gb(1024×768×10s で peak 35.6GB 実測)。
    # 明示値はそのまま使う。
    h3_preset: str = os.getenv("H3_PRESET", "auto")
    h3_gpus: str = os.getenv("H3_GPUS", "0")
    # 2枚目の GPU で decode を並行させる(連続生成の間隔短縮)。"auto" = 2枚目があれば
    # 使う / "off" = 使わない / 数字 = その GPU を使う。96GB 級プリセットでは使わない。
    h3_decode_gpu: str = os.getenv("H3_DECODE_GPU", "auto").strip()
    # H3 に渡す参照画像の短辺(px)。小さいほど denoise が速い。2026-10-10 実測
    # (1024×768・10s): 2048 -> 1024 で denoise 98.3s -> 83.5s(-15%)、同一 seed の
    # 目視で同一性・衣装・鮮鋭度は同等。0 = バックエンド既定(2048)。
    h3_ref_short_edge: int = int(os.getenv("H3_REF_SHORT_EDGE", "1024"))
    tts_url: str = os.getenv("TTS_URL", "http://127.0.0.1:10101").rstrip("/")
    tts_speaker_id: int = int(os.getenv("TTS_SPEAKER_ID", "888753760"))

    width: int = 1024
    height: int = 768
    fps: int = 24
    min_scene_seconds: float = 5.0
    target_scene_seconds: float = 9.0
    max_scene_seconds: float = 10.0


settings = Settings()
settings.jobs_dir.mkdir(parents=True, exist_ok=True)
