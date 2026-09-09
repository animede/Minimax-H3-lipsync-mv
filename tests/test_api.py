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


def test_rejects_unsupported_video_format() -> None:
    response = client.post(
        "/api/jobs",
        data={"width": "999", "height": "999", "fps": "30"},
        files={
            "character": ("character.png", b"png", "image/png"),
            "song": ("song.mp3", b"audio", "audio/mpeg"),
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "対応していない動画サイズです"


def test_official_768p_landscape_and_portrait_sizes_are_supported() -> None:
    from app.main import VIDEO_SIZES

    assert (1344, 768) in VIDEO_SIZES
    assert (768, 1344) in VIDEO_SIZES
