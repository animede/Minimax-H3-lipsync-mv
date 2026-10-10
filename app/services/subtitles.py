"""読み上げ動画のテロップ(焼き込み用 ASS と、アップロード用 SRT)を作る。

入力は TTS が 1 文ずつ合成したときの文タイミング(analysis/sentence_timings.json)。
読み上げ音声そのものの時刻なので、音声認識を使わずに正確に同期する。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

FONT_NAME = "Noto Sans CJK JP"
MAX_LINES = 2

# ここで区切ると読みやすい位置(句読点・括弧・空白の直後)。
_BREAK_AFTER = "、。，．,.!?！？」』）)】:：;；…"


def _layout(width: int, height: int) -> dict[str, int]:
    """画面サイズからフォントサイズ・1 行の文字数・下端からの余白を決める。"""
    portrait = height > width * 1.1
    font = max(20, round(min(width, height) * (0.066 if portrait else 0.062)))
    chars = max(8, int(width * 0.86 / font))
    # 縦長は TikTok・ショート等の UI が画面下部に重なるので、テロップを上げる。
    margin = round(height * (0.20 if portrait else 0.06))
    return {"font": font, "chars": chars, "margin": margin}


def _ascii_word(ch: str) -> bool:
    return bool(ch) and ch.isascii() and ch.isalnum()


def _hiragana(ch: str) -> bool:
    return "ぁ" <= ch <= "ゟ"


def split_caption(text: str, max_chars: int) -> list[str]:
    """1 文を「1 行 max_chars 文字以内」の行に分ける。区切りは読みやすい位置を優先。"""
    text = re.sub(r"\s+", " ", text).strip()
    # 句読点がすぐ先にあるときは、語の途中で切るより少し長い行を許す(上限の 15% まで。
    # 1 行の文字数は画面幅の 86% で見積もっているので、はみ出さない)。
    slack = max(1, round(max_chars * 0.15))
    lines: list[str] = []
    while len(text) > max_chars + slack:
        window = range(min(len(text), max_chars + slack), max(0, max_chars // 2) - 1, -1)
        # 句読点を最優先にし、空白(英単語の区切り)で切るのは句読点が無いときだけ。
        cut = next((i for i in window if text[i - 1] in _BREAK_AFTER), -1)
        if cut <= 0:
            # 英単語どうしの間("Phase C" の空白など)では切らない。
            cut = next((i for i in window if text[i - 1] == " "
                        and not (_ascii_word(text[i - 2]) and _ascii_word(text[i: i + 1]))), -1)
        if cut <= 0:
            # ひらがなの後に漢字・カタカナ・英字が続く位置(「まとめて|磨き」のような
            # 助詞・語尾の後)。形態素解析なしで語の途中を避ける簡易規則。
            cut = next((i for i in window if _hiragana(text[i - 1])
                        and i < len(text) and not _hiragana(text[i])
                        and text[i] not in _BREAK_AFTER), -1)
        if cut <= 0:
            cut = max_chars
        lines.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        lines.append(text)
    return lines


def build_cues(timings: list[dict[str, Any]], max_chars: int) -> list[tuple[float, float, str]]:
    """文タイミングから表示単位(最大 2 行)を作る。長い文は文字数比で時間を配分する。"""
    cues: list[tuple[float, float, str]] = []
    for item in timings:
        text = str(item.get("text") or "").strip()
        start, end = float(item["start"]), float(item["end"])
        if not text or end <= start:
            continue
        lines = split_caption(text, max_chars)
        chunks = ["\n".join(lines[i:i + MAX_LINES]) for i in range(0, len(lines), MAX_LINES)]
        total = sum(len(c.replace("\n", "")) for c in chunks) or 1
        t = start
        for index, chunk in enumerate(chunks):
            span = (end - start) * len(chunk.replace("\n", "")) / total
            chunk_end = end if index == len(chunks) - 1 else t + span
            cues.append((t, chunk_end, chunk))
            t = chunk_end
    return cues


def _ass_time(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _ass_escape(text: str) -> str:
    # ASS の制御文字({} とバックスラッシュ)は全角に置き換えて表示させる。
    return (text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
            .replace("\n", "\\N"))


def write_subtitles(
    timings: list[dict[str, Any]], width: int, height: int, ass_path: Path, srt_path: Path,
) -> int:
    """ASS(焼き込み用)と SRT(アップロード用)を書き出し、テロップの数を返す。"""
    layout = _layout(width, height)
    cues = build_cues(timings, layout["chars"])
    outline = max(2, round(layout["font"] * 0.09))
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{FONT_NAME},{layout['font']},&H00FFFFFF,&H00FFFFFF,&H00000000,"
        f"&H80000000,1,0,0,0,100,100,0,0,1,{outline},1,2,"
        f"{round(width * 0.05)},{round(width * 0.05)},{layout['margin']},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = "".join(
        f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Caption,,0,0,0,,{_ass_escape(t)}\n"
        for s, e, t in cues
    )
    ass_path.write_text(header + events, encoding="utf-8")
    srt = "".join(
        f"{i}\n{_srt_time(s)} --> {_srt_time(e)}\n{t}\n\n" for i, (s, e, t) in enumerate(cues, 1)
    )
    srt_path.write_text(srt, encoding="utf-8")
    return len(cues)
