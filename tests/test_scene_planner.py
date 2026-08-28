from __future__ import annotations

import pytest

from app.services.audio import plan_scenes, quantize_scene_plan
from app.services.llm import enforce_reference_and_lipsync, format_h3_official_ref2va_prompt


def analysis(duration: float, candidates: list[tuple[float, float, str]] | None = None) -> dict:
    return {
        "duration": duration,
        "window_seconds": 0.1,
        "rms_db": [-12.0] * max(1, round(duration * 10)),
        "candidates": [
            {"time": time, "score": score, "kind": kind, "details": {}}
            for time, score, kind in (candidates or [])
        ]
        + [
            {"time": round(value * 0.25, 2), "score": 0.0, "kind": "fallback", "details": {}}
            for value in range(1, round(duration * 4))
        ],
    }


@pytest.mark.parametrize("duration", [24.0, 27.0, 32.0, 57.0, 180.0])
def test_plan_covers_song_with_h3_bounds(duration: float) -> None:
    scenes = plan_scenes(analysis(duration))
    assert scenes[0]["start"] == 0
    assert scenes[-1]["end"] == duration
    assert all(5.0 <= scene["duration"] <= 10.0 for scene in scenes)
    assert sum(scene["duration"] for scene in scenes) == pytest.approx(duration)


def test_natural_silence_wins_over_equal_split() -> None:
    data = analysis(27.0, [(8.2, 100.0, "silence"), (17.1, 100.0, "silence")])
    scenes = plan_scenes(data)
    boundaries = [scene["end"] for scene in scenes[:-1]]
    assert boundaries == pytest.approx([8.2, 17.1], abs=0.1)


def test_short_song_is_padded_later_but_keeps_real_duration() -> None:
    scenes = plan_scenes(analysis(3.7))
    assert scenes == [{
        "index": 1,
        "start": 0.0,
        "end": 3.7,
        "duration": 3.7,
        "boundary_kind": "song_end",
        "boundary_score": 0.0,
    }]


def test_no_word_midpoint_when_phrase_boundary_exists() -> None:
    data = analysis(19.0, [(9.4, 76.0, "phrase_end")])
    scenes = plan_scenes(data)
    assert scenes[0]["boundary_kind"].find("phrase_end") >= 0


def test_frame_quantization_has_no_cumulative_error() -> None:
    scenes = plan_scenes(analysis(57.03, [(8.113, 76, "phrase_end"), (17.227, 76, "phrase_end")]))
    quantized = quantize_scene_plan(scenes, 24)
    assert sum(scene["frame_count"] for scene in quantized) == round(57.03 * 24)
    assert quantized[-1]["end_frame"] == round(57.03 * 24)


def test_singing_prompt_forces_identity_and_visible_mouth() -> None:
    prompt = enforce_reference_and_lipsync(
        {"vocal_ratio": 0.9},
        "Extreme close-up of the eye, silver hair, wide shot from behind",
        {"identity_anchor": "same black-haired woman in the reference image"},
    )
    assert "same black-haired woman" in prompt
    assert "mouth remain clearly visible" in prompt
    assert "precise synchronization" in prompt
    assert "wide shot" not in prompt.lower()
    assert "from behind" not in prompt.lower()


def test_instrumental_prompt_keeps_mouth_closed() -> None:
    prompt = enforce_reference_and_lipsync(
        {"vocal_ratio": 0.1}, "walking in moonlight", {"identity_anchor": "same person"}
    )
    assert "closed, relaxed mouth" in prompt


def test_official_ref2va_prompt_uses_exact_six_section_order() -> None:
    prompt = format_h3_official_ref2va_prompt(
        {"vocal_ratio": 0.9},
        "Medium close-up of the singer under soft window light.",
        {
            "subject": "young East Asian woman",
            "hair": "long dark-brown hair with wispy bangs",
            "outfit": "beige cardigan over a sailor-collar top",
            "identity_anchor": "preserve the exact same woman",
        },
    )
    fields = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    assert [prompt.index(field) for field in fields] == sorted(prompt.index(field) for field in fields)
    assert "<Subject 1> is the exact character shown in <Picture 1>" in prompt
    assert "<Picture 1> (" not in prompt
    assert "<Audio 1>: fully_copy" in prompt
    assert "[reference generation + audio reuse]" in prompt
    assert "[Shot 1]" in prompt
    assert "[Shot 2]" not in prompt
    assert "<d>" not in prompt
    assert "<Subject 1> (S1)" in prompt
    assert "precise synchronization with <Audio 1>" in prompt


def test_official_ref2va_non_vocal_scene_prevents_false_speech() -> None:
    prompt = format_h3_official_ref2va_prompt(
        {"vocal_ratio": 0.1}, "The character watches rain at a window.", {"subject": "same person"}
    )
    assert "does not sing" in prompt
    assert "closed, relaxed mouth" in prompt
    assert "non_diegetic_music:\nN/A" in prompt
