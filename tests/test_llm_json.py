from __future__ import annotations

import pytest

from app.services import llm


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('```json\n{"title":"MV"}\n```', {"title": "MV"}),
        ('Here is the result: {"title":"MV"}\nDone.', {"title": "MV"}),
        (
            'I considered [shot continuity] first. Final: [{"index":1,"prompt":"sing"}]',
            [{"index": 1, "prompt": "sing"}],
        ),
        (
            '<think>Use {immutable identity} throughout.</think>\n{"title":"MV"}',
            {"title": "MV"},
        ),
    ],
)
def test_extract_json_ignores_gemma_wrapping(response: str, expected: object) -> None:
    assert llm._extract_json(response) == expected


def test_scenario_request_retries_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            {"choices": [{"message": {"content": '{"title": "unfinished"'}}]},
            {"choices": [{"message": {"content": '{"title":"repaired"}'}}]},
        ]
    )
    calls: list[dict] = []

    def fake_chat(*args: object, **kwargs: object) -> dict:
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(llm, "chat", fake_chat)

    assert llm._scenario_request("Create a treatment") == {"title": "repaired"}
    assert len(calls) == 2
    assert calls[1]["temperature"] == 0.0


def test_generate_scenario_retries_batch_with_wrong_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenes = [
        {"index": index, "start": index - 1, "end": index, "vocal_ratio": 0.8}
        for index in range(1, 7)
    ]
    first_batch = [
        {"index": index, "emotion": "", "shot": "", "camera": "", "prompt": "sing"}
        for index in range(1, 6)
    ]
    corrected_second_batch = [
        {"index": 6, "emotion": "", "shot": "", "camera": "", "prompt": "sing"}
    ]
    responses = iter(
        [
            {"title": "MV"},
            first_batch,
            [],
            corrected_second_batch,
        ]
    )
    prompts: list[str] = []

    def fake_request(prompt: str, max_tokens: int = 3000) -> object:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(llm, "_scenario_request", fake_request)

    result = llm.generate_scenario(scenes, "", {"identity_anchor": "same person"})

    assert [scene["index"] for scene in result["scenes"]] == list(range(1, 7))
    assert "indexes [6]" in prompts[2]
    assert "CORRECTION" in prompts[3]
