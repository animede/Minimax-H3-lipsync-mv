from itertools import pairwise

from app.services.subtitles import build_cues, split_caption, write_subtitles


def test_split_prefers_punctuation():
    lines = split_caption("しばらくこのまま制作で使い込んでみてください。気になる点が溜まってきたら", 20)
    assert all(len(line) <= 23 for line in lines)  # 句読点までなら 15% 超過を許す
    assert lines[0].endswith("。")


def test_long_sentence_is_split_and_time_is_conserved():
    timings = [{"text": "あ" * 50 + "。", "start": 1.0, "end": 6.0}]
    cues = build_cues(timings, 10)
    assert len(cues) >= 3  # 51 文字 / 1 行 10 文字前後 / 2 行ずつ
    assert cues[0][0] == 1.0 and cues[-1][1] == 6.0
    assert all(a[1] == b[0] for a, b in pairwise(cues))
    assert all(c[2].count("\n") <= 1 for c in cues)


def test_writes_ass_and_srt(tmp_path):
    timings = [{"text": "テスト{です}", "start": 0.0, "end": 2.5}]
    n = write_subtitles(timings, 576, 1024, tmp_path / "s.ass", tmp_path / "s.srt")
    assert n == 1
    ass = (tmp_path / "s.ass").read_text(encoding="utf-8")
    assert "PlayResY: 1024" in ass and "｛です｝" in ass  # ASS 制御文字は全角化
    srt = (tmp_path / "s.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,500" in srt


def test_does_not_separate_number_from_counter():
    lines = split_caption("改善案としては、テロップ付きとテロップなしの 2 本を毎回両方出力するのが手軽です。", 24)
    assert not any(line.endswith("2") for line in lines)
