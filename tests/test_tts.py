from __future__ import annotations

import io
import wave

from app.services import tts


def _wav_bytes(frames: int = 1600) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * frames)
    return output.getvalue()


def test_split_sentences_keeps_punctuation() -> None:
    assert tts.split_sentences("こんにちは。元気ですか？\nはい！") == [
        "こんにちは。", "元気ですか？", "はい！"
    ]


def test_synthesize_text_calls_engine_once_per_sentence(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def synthesize(sentence: str) -> bytes:
        calls.append(sentence)
        return _wav_bytes()

    monkeypatch.setattr(tts, "_synthesize_sentence", synthesize)
    destination = tmp_path / "narration.wav"
    timings = tts.synthesize_text("最初です。次です。", destination)

    assert calls == ["最初です。", "次です。"]
    assert len(timings) == 2
    assert timings[1]["start"] == 0.1
    with wave.open(str(destination), "rb") as wav:
        assert wav.getnframes() == 3200
