from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_index_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Character Video Studio" in response.text


def test_rejects_wrong_upload_types() -> None:
    response = client.post(
        "/api/jobs",
        files={
            "character": ("character.txt", b"not image", "text/plain"),
            "song": ("song.mp3", b"not audio", "audio/mpeg"),
        },
    )
    assert response.status_code == 415


def test_requires_exactly_one_audio_or_text_source() -> None:
    response = client.post(
        "/api/jobs",
        files={"character": ("character.png", b"png", "image/png")},
    )
    assert response.status_code == 422
