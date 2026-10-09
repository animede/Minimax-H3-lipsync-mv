from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import requests

from ..config import settings


class LLMError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    return headers


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Gemma may put reasoning or a short introduction before the answer. That text can
        # itself contain '[' or '{', so trying only the first opening character can miss a
        # perfectly valid JSON value later in the response.
        starts = [match.start() for match in re.finditer(r"[\[{]", cleaned)]
        if not starts:
            raise LLMError("LLM応答にJSONがありません")
        decoder = json.JSONDecoder()
        for start in starts:
            try:
                value, _ = decoder.raw_decode(cleaned, start)
                return value
            except json.JSONDecodeError:
                continue
        raise LLMError("LLM応答のJSONを解析できません")


def chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    timeout: float = 180,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    try:
        response = requests.post(
            f"{base_url}/chat/completions", headers=_headers(), json=payload, timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise LLMError(f"LLM接続に失敗しました: {base_url}: {exc}") from exc


def check_models(base_url: str, timeout: float = 8) -> list[str]:
    try:
        response = requests.get(f"{base_url}/models", headers=_headers(), timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return [str(item.get("id") or item.get("name")) for item in data.get("data", [])]
    except (requests.RequestException, ValueError) as exc:
        raise LLMError(f"モデル一覧を取得できません: {base_url}: {exc}") from exc


def _scenario_request(prompt: str, max_tokens: int = 3000) -> Any:
    system = (
        "You are a music-video director and MiniMax-H3 prompt engineer. "
        "Preserve one character identity and return strictly valid JSON."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    data = chat(
        settings.scenario_llm_url,
        settings.scenario_llm_model,
        messages,
        max_tokens=max_tokens,
        temperature=0.35,
        timeout=240,
    )
    try:
        content = str(data["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("シナリオLLMの応答形式が不正です") from exc
    try:
        return _extract_json(content)
    except LLMError:
        # A single corrective turn handles truncated prose, missing delimiters and other
        # occasional malformed generations without restarting the complete MV job.
        repair_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Return the complete answer again "
                    "as JSON only, with no Markdown fence, commentary, or reasoning. Preserve the "
                    "exact schema and requested number of scenes."
                ),
            },
        ]
        repaired = chat(
            settings.scenario_llm_url,
            settings.scenario_llm_model,
            repair_messages,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=240,
        )
        try:
            repaired_content = str(
                repaired["choices"][0]["message"].get("content") or ""
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("シナリオLLMの再応答形式が不正です") from exc
        return _extract_json(repaired_content)


def analyze_character(image_path: Path) -> dict[str, Any]:
    """Describe only identity-preserving visual facts from the reference image."""
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data = chat(
        settings.scenario_llm_url,
        settings.scenario_llm_model,
        [
            {
                "role": "system",
                "content": (
                    "You analyze a character reference for identity-preserving video generation. "
                    "Describe only visible facts. Never invent a different hairstyle, hair color, outfit, age, "
                    "ethnicity, accessories, or art style. Return strictly valid JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Return {\"subject\":\"\",\"face\":\"\",\"hair\":\"\","
                            "\"outfit\":\"\",\"style\":\"\",\"identity_anchor\":\"\"}. "
                            "identity_anchor must be a concise English instruction preserving the exact same person."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            },
        ],
        max_tokens=700,
        temperature=0.0,
        timeout=180,
    )
    try:
        parsed = _extract_json(str(data["choices"][0]["message"].get("content") or ""))
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("参照画像解析の応答形式が不正です") from exc
    if not isinstance(parsed, dict):
        raise LLMError("参照画像解析がJSONオブジェクトではありません")
    return parsed


def _identity_instruction(character: dict[str, Any]) -> str:
    anchor = str(character.get("identity_anchor") or "").strip()
    visible = ", ".join(
        str(character.get(key) or "").strip()
        for key in ("subject", "face", "hair", "outfit", "style")
        if str(character.get(key) or "").strip()
    )
    return anchor or visible or "the exact same person and appearance shown in the reference image"


def _single_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().rstrip(".")


def _subject_definition(character: dict[str, Any]) -> str:
    facts = [
        _single_line(character.get(key))
        for key in ("subject", "face", "hair", "outfit", "style")
        if _single_line(character.get(key))
    ]
    description = ", ".join(facts)
    if not description:
        description = "the exact person, face, hair, clothing, and visual style visible in the image"
    return description


def frame_kind(width: int, height: int) -> str:
    """生成サイズから構図の種類を返す: "portrait" / "square" / "landscape"。"""
    ratio = width / max(height, 1)
    if ratio < 0.9:
        return "portrait"
    if ratio > 1.1:
        return "landscape"
    return "square"


# LLM へのシーン指示(縦長・正方形では被写体を中央に置かせる)。横長は従来どおり。
_FRAME_LLM_GUIDANCE = {
    "portrait": (
        "The output is a VERTICAL frame for mobile viewing (9:16 or 4:5). Keep the character "
        "horizontally centered. Place props, screens and scenery above, behind or around the "
        "character, never to one side in a way that pushes the character toward the frame edge. "
    ),
    "square": (
        "The output is a SQUARE 1:1 frame. Keep the character centered; keep props and scenery "
        "behind the character rather than to one side. "
    ),
    "landscape": "",
}

# [Shot 1] の冒頭に置く構図文。H3 は文頭の指示が最も効く(2026-10-08 実測、固定カメラ句の
# 配置 A/B)。横長は LLM の構図に任せる。
_FRAME_SHOT_PREFIX = {
    "portrait": (
        "Vertical frame composition: <Subject 1> is centered horizontally, with the head and "
        "shoulders in the upper-middle of the frame and the background arranged above and behind "
        "the subject. "
    ),
    "square": "Square frame composition: <Subject 1> is centered in the frame. ",
    "landscape": "",
}


def format_h3_official_ref2va_prompt(
    scene: dict[str, Any], visual_direction: str, character: dict[str, Any],
    frame: str = "landscape",
) -> str:
    """Wrap one generated clip in MiniMax H3's official Ref2VA six-section format.

    The reference image defines a reusable subject rather than a keyframe, so the official
    guide requires it to be cited inside ``<Subject 1>`` instead of receiving a standalone
    ``<Picture 1>`` retention entry.  Each pipeline scene is one H3 request, therefore its
    timeline always contains one untimestamped ``[Shot 1]``.

    Exact lyrics are intentionally omitted.  The local H3 validation notes show that Vocal
    Lock can transfer the spoken/sung content from ``<Audio 1>`` without guessing text in a
    ``<d>`` block; guessed lyrics would violate the official guide.
    """
    subject = _subject_definition(character)
    identity = _single_line(_identity_instruction(character))
    direction = _single_line(visual_direction)
    vocal_ratio = float(scene.get("vocal_ratio") or 0.0)
    narration = bool(scene.get("narration"))
    singing = vocal_ratio >= 0.35

    if narration:
        performance = (
            "<Subject 1> (S1) naturally speaks the supplied narration. The full face and "
            "unobstructed mouth remain visible, and the lips, jaw, and facial muscles articulate "
            "every audible phoneme in precise synchronization with <Audio 1>. No words are added, "
            "removed, inferred, or displayed on screen."
        )
        summary_action = "speaks the supplied narration with clearly visible synchronized lip movement"
    elif singing:
        performance = (
            "<Subject 1> (S1) physically performs the supplied vocal. The full face and "
            "unobstructed mouth remain visible, and the lips, jaw, and facial muscles articulate "
            "every audible phoneme in precise synchronization with <Audio 1>. No lyrics are "
            "written or inferred because the exact source words are not supplied."
        )
        summary_action = "performs the supplied vocal with clearly visible synchronized lip movement"
    else:
        performance = (
            "<Audio 1> remains the reused timing and vocal track for this interval. "
            "<Subject 1> does not sing during the near-silent vocal interval and keeps a naturally "
            "closed, relaxed mouth without unrelated speech-like movement."
        )
        summary_action = "remains on screen with a closed, relaxed mouth during the non-vocal interval"

    return (
        "subject_definitions:\n"
        f"<Subject 1> is the exact character shown in <Picture 1>: {subject}. "
        "The same identity, facial structure, hair, clothing, and visual style must remain unchanged.\n"
        "<Audio 1> is the supplied isolated vocal signal for this entire target clip and is reused "
        "as the clip's synchronized audio track for <Subject 1> (S1).\n\n"
        "summary:\n"
        "[reference generation + audio reuse] The target video is a single continuous music-video "
        f"shot in which <Subject 1> {summary_action}, while <Audio 1> is reused for synchronization.\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - the referenced character's identity, "
        f"face, hair, clothing, and visual style are retained exactly: {identity}.\n"
        "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target clip's complete synchronized "
        "audio track.\n\n"
        "detailed_description:\n"
        "The target video uses a coherent cinematic music-video style while fully preserving the "
        "reference character. It contains one continuous shot with no internal cut.\n"
        f"[Shot 1] {_FRAME_SHOT_PREFIX.get(frame, '')}{direction} {performance}\n\n"
        "overall_soundscape:\n"
        "No additional dialogue, ambience, crowd voices, or sound effects are introduced; the "
        "synchronized vocal content comes only from <Audio 1>.\n\n"
        "non_diegetic_music:\n"
        "N/A"
    )


def enforce_reference_and_lipsync(
    scene: dict[str, Any], raw_prompt: str, character: dict[str, Any]
) -> str:
    """Apply non-negotiable identity and mouth-visibility rules after the LLM."""
    identity = _identity_instruction(character)
    prompt = str(raw_prompt or "").strip()
    vocal_ratio = float(scene.get("vocal_ratio") or 0.0)
    narration = bool(scene.get("narration"))
    if vocal_ratio >= 0.35:
        # Remove camera directions that make mouth synchronization impossible to judge.
        replacements = {
            r"\bextreme close-up(?: shot)? of (?:the )?(?:eye|eyes)\b": "medium close-up portrait",
            r"\b(?:extreme )?wide shot\b": "medium close-up shot",
            r"\b(?:full-body|full body|long) shot\b": "medium close-up shot",
            r"\bover-the-shoulder shot\b": "front three-quarter medium close-up shot",
            r"\bfrom behind\b": "from a front three-quarter angle",
            r"\bprofile(?: shot)?\b": "three-quarter face view",
            r"\btilts? (?:her|his|their) head back\b": "keeps the face oriented toward the camera",
        }
        for pattern, replacement in replacements.items():
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
        # The official validator treats repeated shot-size words inside one [Shot] as a
        # possible framing change. The mandatory prefix below owns the single shot size;
        # collapse all shot-size remnants in the creative direction to a neutral phrase.
        prompt = re.sub(
            r"\b(?:medium close-up(?: shot| portrait)?|medium shot|medium-wide shot)\b",
            "performance framing",
            prompt,
            flags=re.IGNORECASE,
        )
        prefix = (
            "Medium close-up performance shot, front-facing or three-quarter face view. "
            "Use the exact same person from the reference image; preserve facial identity, hair, outfit, "
            f"and visual style unchanged: {identity}. "
            "The full face and mouth remain clearly visible and unobstructed throughout the shot. "
            f"The character {'speaks the supplied narration' if narration else 'delivers the supplied vocal'} naturally; the lips and jaw articulate every phoneme "
            "in precise synchronization with the reference vocal audio. Subtle natural breathing and facial "
            "muscle motion, direct performance presence. "
        )
    else:
        prefix = (
            "Use the exact same person from the reference image; preserve facial identity, hair, outfit, "
            f"and visual style unchanged: {identity}. "
            "The character is not singing in this interval and keeps a naturally closed, relaxed mouth. "
        )
    return prefix + prompt


def generate_scenario(
    scenes: list[dict[str, Any]],
    concept: str,
    character: dict[str, Any],
    *, source_text: str = "", frame: str = "landscape",
) -> dict[str, Any]:
    narration_instruction = (
        "This is a spoken narration video, not a music video. Build the visual story directly from "
        f"the complete source text below. Do not rewrite its meaning.\nSOURCE_TEXT: {source_text}\n"
        if source_text else ""
    )
    overview = _scenario_request(
        ("Create a coherent spoken-video treatment. " if source_text else "Create a coherent MV treatment. ")
        + "Do not change scene times.\n"
        + narration_instruction
        +
        f"USER_CONCEPT: {concept or '(derive from the music and character)'}\n"
        f"CHARACTER_REFERENCE_FACTS: {json.dumps(character, ensure_ascii=False)}\n"
        "The reference character's identity, hair, face, outfit and style are immutable. "
        "Do not invent a named alternate character or redesign their appearance.\n"
        f"SCENES: {json.dumps(scenes, ensure_ascii=False)}\n"
        "Return: {\"title\":\"\",\"visual_theme\":\"\",\"color_script\":\"\","
        "\"story_arc\":\"\",\"continuity_rules\":[\"\"]}."
    )
    if not isinstance(overview, dict):
        raise LLMError("全体シナリオがJSONオブジェクトではありません")

    outputs: list[dict[str, Any]] = []
    for start in range(0, len(scenes), 5):
        batch = scenes[start:start + 5]
        previous = outputs[-1] if outputs else None
        expected = [int(item["index"]) for item in batch]
        response_template = [
            {"index": index, "emotion": "", "shot": "", "camera": "", "prompt": ""}
            for index in expected
        ]
        batch_prompt = (
            "Write one production prompt per fixed scene. Do not add/remove scenes or change times. "
            "Prompts must be English and suitable for MiniMax-H3 ref2va. Keep exactly one character. "
            "Never describe hair color, hairstyle, face, clothing, age or art style differently from "
            "CHARACTER_REFERENCE_FACTS. For vocal_ratio >= 0.35 use a medium close-up, keep the entire "
            "face and mouth visible, and explicitly request phoneme-level lip articulation synchronized to "
            "the supplied vocal. Never use a wide shot, rear view, eye-only shot or obscured mouth for a "
            "singing scene. For lower vocal_ratio explicitly describe a closed relaxed mouth. "
            "Use restrained camera motion and avoid cuts inside a generated clip.\n"
            + _FRAME_LLM_GUIDANCE.get(frame, "")
            + narration_instruction
            +
            f"CHARACTER_REFERENCE_FACTS: {json.dumps(character, ensure_ascii=False)}\n"
            f"TREATMENT: {json.dumps(overview, ensure_ascii=False)}\n"
            f"PREVIOUS: {json.dumps(previous, ensure_ascii=False)}\n"
            f"SCENES: {json.dumps(batch, ensure_ascii=False)}\n"
            f"Return exactly {len(batch)} array items with indexes {expected}. "
            "Use this exact JSON structure and replace only the empty string values: "
            f"{json.dumps(response_template, ensure_ascii=False)}"
        )
        result = _scenario_request(batch_prompt)
        actual = (
            [int(item.get("index", -1)) for item in result if isinstance(item, dict)]
            if isinstance(result, list)
            else []
        )
        if not isinstance(result, list) or len(result) != len(batch) or actual != expected:
            result = _scenario_request(
                batch_prompt
                + "\nCORRECTION: The previous answer had the wrong item count or indexes. "
                + f"Return exactly {len(batch)} items in this order: {expected}. JSON only."
            )
            actual = (
                [int(item.get("index", -1)) for item in result if isinstance(item, dict)]
                if isinstance(result, list)
                else []
            )
        if not isinstance(result, list) or len(result) != len(batch):
            actual_count = len(result) if isinstance(result, list) else 0
            raise LLMError(
                f"シーン{start + 1}以降のプロンプト数が一致しません: "
                f"expected={len(batch)}, actual={actual_count}"
            )
        if actual != expected:
            raise LLMError(f"シーン番号が一致しません: expected={expected}, actual={actual}")
        for scene, item in zip(batch, result, strict=True):
            visual_direction = enforce_reference_and_lipsync(
                scene, str(item.get("prompt") or ""), character
            )
            item["prompt"] = format_h3_official_ref2va_prompt(
                scene, visual_direction, character, frame=frame
            )
        outputs.extend(result)
    return {
        "prompt_format": "MiniMax H3 official Ref2VA six-section format",
        "character": character,
        "overview": overview,
        "scenes": outputs,
    }
