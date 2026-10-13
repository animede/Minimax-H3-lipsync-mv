from __future__ import annotations

import io
import re
import wave
from pathlib import Path
from typing import Any

import requests

from ..config import settings


class TTSError(RuntimeError):
    pass


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > settings.max_text_bytes:
        raise TTSError("テキストファイルが上限を超えています")
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise TTSError("テキストはUTF-8またはShift_JISで保存してください")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise TTSError("テキストファイルが空です")
    return text


def split_sentences(text: str) -> list[str]:
    """Split Japanese/Latin prose while retaining sentence-ending punctuation."""
    normalized = re.sub(r"[ \t]+", " ", text.strip())
    parts = re.split(r"(?<=[。！？!?])(?:[\u3000 \t]*|\n+)|\n+", normalized)
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences:
        raise TTSError("読み上げる文章がありません")
    return sentences


def check_tts(timeout: float = 5) -> dict[str, Any]:
    try:
        version = requests.get(f"{settings.tts_url}/version", timeout=timeout)
        version.raise_for_status()
        speakers = requests.get(f"{settings.tts_url}/speakers", timeout=timeout)
        speakers.raise_for_status()
        return {"ok": True, "version": version.json(), "speakers": speakers.json()}
    except (requests.RequestException, ValueError) as exc:
        raise TTSError(f"AivisSpeech Engineに接続できません: {exc}") from exc


def list_speakers(timeout: float = 5) -> list[dict[str, Any]]:
    """AivisSpeech の話者一覧を UI のプルダウン用に平坦化する(1スタイル=1項目)。"""
    try:
        response = requests.get(f"{settings.tts_url}/speakers", timeout=timeout)
        response.raise_for_status()
        speakers = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise TTSError(f"AivisSpeech Engineに接続できません: {exc}") from exc
    options: list[dict[str, Any]] = []
    for speaker in speakers:
        styles = speaker.get("styles") or []
        for style in styles:
            name = str(speaker.get("name") or "不明な話者")
            if len(styles) > 1:
                name = f"{name}({style.get('name')})"
            options.append({"id": style.get("id"), "name": name})
    return options


def _synthesize_sentence(sentence: str, speaker_id: int) -> bytes:
    params = {"text": sentence, "speaker": speaker_id}
    try:
        query = requests.post(
            f"{settings.tts_url}/audio_query", params=params, timeout=60
        )
        query.raise_for_status()
        synthesis = requests.post(
            f"{settings.tts_url}/synthesis",
            params={"speaker": speaker_id},
            json=query.json(),
            timeout=180,
        )
        synthesis.raise_for_status()
        return synthesis.content
    except (requests.RequestException, ValueError) as exc:
        raise TTSError(f"AivisSpeechの音声合成に失敗しました: {exc}") from exc


def synthesize_text(
    text: str, destination: Path, speaker_id: int = 0
) -> list[dict[str, Any]]:
    speaker_id = speaker_id or settings.tts_speaker_id
    sentences = split_sentences(text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm_parts: list[bytes] = []
    params: tuple[int, int, int] | None = None
    timings: list[dict[str, Any]] = []
    elapsed = 0.0
    for index, sentence in enumerate(sentences, start=1):
        try:
            with wave.open(io.BytesIO(_synthesize_sentence(sentence, speaker_id)), "rb") as source:
                current = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                if params is None:
                    params = current
                elif current != params:
                    raise TTSError("文ごとの合成音声形式が一致しません")
                frames = source.readframes(source.getnframes())
                duration = source.getnframes() / source.getframerate()
        except (wave.Error, EOFError) as exc:
            raise TTSError("AivisSpeechから不正なWAVデータが返されました") from exc
        pcm_parts.append(frames)
        timings.append({
            "index": index,
            "text": sentence,
            "start": round(elapsed, 3),
            "end": round(elapsed + duration, 3),
            "duration": round(duration, 3),
        })
        elapsed += duration
    assert params is not None
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(params[0])
        output.setsampwidth(params[1])
        output.setframerate(params[2])
        for frames in pcm_parts:
            output.writeframes(frames)
    return timings
