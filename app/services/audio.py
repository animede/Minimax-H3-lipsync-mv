from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import requests

from ..config import settings


class AudioError(RuntimeError):
    pass


@dataclass
class BoundaryCandidate:
    time: float
    score: float
    kind: str
    details: dict[str, float]


def run_command(command: list[str], timeout: float | None = None) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise AudioError(f"command failed: {command[0]} | {result.stderr[-500:]}")
    return result.stdout + result.stderr


def probe_duration(path: Path) -> float:
    output = run_command([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ])
    try:
        return float(output.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise AudioError("楽曲の長さを取得できません") from exc


def normalize_audio(source: Path, destination: Path, *, sample_rate: int = 44100) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-v", "error", "-i", str(source), "-ac", "2",
        "-ar", str(sample_rate), str(destination),
    ])
    return destination


_DEMUCS_SCRIPT = r"""
import sys, torch, soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
src, dst = sys.argv[1], sys.argv[2]
model = get_model('htdemucs')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device).eval()
wav, sr = sf.read(src, dtype='float32', always_2d=True)
x = torch.from_numpy(wav.T).unsqueeze(0).to(device)
ref = x.mean(0)
x = (x - ref.mean()) / max(ref.std(), torch.tensor(1e-6, device=device))
with torch.no_grad():
    sources = apply_model(model, x, device=device, split=True, overlap=0.25, progress=False)[0]
sources = sources * ref.std() + ref.mean()
vocals = sources[model.sources.index('vocals')].cpu().numpy()
sf.write(dst, vocals.T, sr, subtype='PCM_16')
"""


def separate_vocals(source: Path, work_dir: Path, *, timeout: float = 1800) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as stream:
            response = requests.post(
                f"{settings.stem_api_url}/api/stems/upload",
                files={"file": (source.name, stream)}, timeout=30,
            )
        if response.ok:
            job_id = str(response.json().get("job_id") or "")
            deadline = time.monotonic() + timeout
            while job_id and time.monotonic() < deadline:
                job = requests.get(
                    f"{settings.stem_api_url}/api/stems/{job_id}", timeout=30
                ).json()
                state = str(job.get("status") or "")
                if state in {"done", "completed", "finished"}:
                    stem = requests.get(
                        f"{settings.stem_api_url}/api/stems/{job_id}/vocals", timeout=300
                    )
                    stem.raise_for_status()
                    destination = work_dir / "vocals.wav"
                    destination.write_bytes(stem.content)
                    return destination
                if state in {"error", "failed"}:
                    raise AudioError(f"ボーカル分離に失敗しました: {job.get('error')}")
                time.sleep(2)
    except requests.RequestException:
        pass

    configured_python = settings.demucs_python
    demucs_python = (
        str(Path(configured_python).expanduser())
        if Path(configured_python).expanduser().is_file()
        else shutil.which(configured_python)
    )
    if not demucs_python:
        raise AudioError("Demucsサービスに接続できず、フォールバック環境もありません")
    normalized = normalize_audio(source, work_dir / "source_44k.wav")
    destination = work_dir / "vocals.wav"
    result = subprocess.run(
        [demucs_python, "-c", _DEMUCS_SCRIPT, str(normalized), str(destination)],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0 or not destination.is_file():
        raise AudioError(f"Demucsに失敗しました: {result.stderr[-500:]}")
    return destination


def _local_minima(values: np.ndarray, radius: int = 2) -> Iterable[int]:
    for index in range(radius, len(values) - radius):
        window = values[index - radius:index + radius + 1]
        if values[index] <= np.min(window):
            yield index


def _merge_candidates(candidates: list[BoundaryCandidate], tolerance: float = 0.22) -> list[BoundaryCandidate]:
    if not candidates:
        return []
    merged: list[BoundaryCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.time):
        if merged and candidate.time - merged[-1].time <= tolerance:
            previous = merged[-1]
            weight = max(previous.score, 1.0) + max(candidate.score, 1.0)
            previous.time = round(
                (previous.time * max(previous.score, 1.0) + candidate.time * max(candidate.score, 1.0))
                / weight, 3
            )
            previous.score += candidate.score
            kinds = set(previous.kind.split("+")) | set(candidate.kind.split("+"))
            previous.kind = "+".join(sorted(kinds))
            previous.details.update(candidate.details)
        else:
            merged.append(candidate)
    return merged


def analyze_boundaries(song: Path, vocals: Path, work_dir: Path) -> dict[str, Any]:
    """Extract explainable boundary candidates from vocals and the full mix."""
    sample_rate = 16000
    hop_length = 1600  # 100 ms
    vocal_wave, _ = librosa.load(vocals, sr=sample_rate, mono=True)
    song_wave, _ = librosa.load(song, sr=sample_rate, mono=True)
    duration = len(song_wave) / sample_rate

    rms = librosa.feature.rms(y=vocal_wave, frame_length=2048, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)
    flatness = librosa.feature.spectral_flatness(
        y=vocal_wave, n_fft=2048, hop_length=hop_length
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        vocal_wave, frame_length=2048, hop_length=hop_length
    )[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)

    candidates: list[BoundaryCandidate] = []
    for index in _local_minima(rms_db, radius=2):
        second = float(times[index])
        if not 0.5 < second < duration - 0.5:
            continue
        db = float(rms_db[index])
        if db <= -38:
            score, kind = 100.0, "silence"
        elif db <= -28:
            score, kind = 68.0, "vocal_valley"
        elif db <= -20:
            score, kind = 36.0, "soft_valley"
        else:
            continue
        candidates.append(BoundaryCandidate(second, score, kind, {"vocal_db": round(db, 2)}))

    # Breath is broad-band and unvoiced. It is a supporting cue, never a sole hard decision.
    flat_threshold = float(np.quantile(flatness, 0.72)) if len(flatness) else 1.0
    for index in range(1, min(len(rms_db), len(flatness), len(zcr)) - 1):
        db = float(rms_db[index])
        if -38 < db < -15 and flatness[index] >= flat_threshold and zcr[index] >= 0.08:
            second = float(times[index])
            candidates.append(BoundaryCandidate(
                second, 42.0, "breath",
                {"vocal_db": round(db, 2), "flatness": round(float(flatness[index]), 3)},
            ))

    tempo, beat_frames = librosa.beat.beat_track(
        y=song_wave, sr=sample_rate, hop_length=hop_length, units="frames"
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    for beat_index, second in enumerate(beat_times):
        if 0.5 < second < duration - 0.5:
            is_bar = beat_index % 4 == 0
            candidates.append(BoundaryCandidate(
                float(second), 26.0 if is_bar else 10.0,
                "bar" if is_bar else "beat", {},
            ))

    # Structural novelty from chroma changes. Peaks are candidates, not authoritative labels.
    chroma = librosa.feature.chroma_cqt(y=song_wave, sr=sample_rate, hop_length=hop_length)
    if chroma.shape[1] > 2:
        novelty = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
        threshold = float(np.quantile(novelty, 0.9))
        for index in _local_minima(-novelty, radius=3):
            if novelty[index] >= threshold:
                second = float(times[min(index + 1, len(times) - 1)])
                candidates.append(BoundaryCandidate(
                    second, 46.0, "section_change", {"novelty": round(float(novelty[index]), 3)}
                ))

    # Dense fallback grid guarantees a feasible 5–10 second path, but receives no bonus.
    for second in np.arange(0.25, duration, 0.25):
        candidates.append(BoundaryCandidate(float(second), 0.0, "fallback", {}))

    merged = _merge_candidates(candidates)
    analysis = {
        "duration": round(duration, 3),
        "tempo": round(float(np.asarray(tempo).reshape(-1)[0]), 2),
        "window_seconds": hop_length / sample_rate,
        "rms_db": [round(float(value), 2) for value in rms_db],
        "candidates": [asdict(item) for item in merged],
    }
    (work_dir / "boundary_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return analysis


def _boundary_penalty(duration: float, target: float) -> float:
    # 7–10 seconds remains comfortable; 5 seconds is legal but deliberately expensive.
    penalty = abs(duration - target) * 5.0
    if duration < 7.0:
        penalty += (7.0 - duration) * 9.0
    return penalty


def plan_scenes(
    analysis: dict[str, Any], *, min_seconds: float = settings.min_scene_seconds,
    target_seconds: float = settings.target_scene_seconds,
    max_seconds: float = settings.max_scene_seconds,
) -> list[dict[str, Any]]:
    total = float(analysis["duration"])
    if total <= min_seconds:
        return [{
            "index": 1, "start": 0.0, "end": round(total, 3),
            "duration": round(total, 3), "boundary_kind": "song_end",
            "boundary_score": 0.0,
        }]

    raw = [BoundaryCandidate(**item) for item in analysis.get("candidates", [])]
    points = [BoundaryCandidate(0.0, 0.0, "song_start", {})]
    points.extend(item for item in raw if 0 < item.time < total)
    points.append(BoundaryCandidate(total, 0.0, "song_end", {}))
    points = _merge_candidates(points, tolerance=0.08)

    best = [-math.inf] * len(points)
    previous = [-1] * len(points)
    best[0] = 0.0
    for end_index in range(1, len(points)):
        for start_index in range(end_index - 1, -1, -1):
            duration = points[end_index].time - points[start_index].time
            if duration > max_seconds + 1e-6:
                continue
            if duration < min_seconds - 1e-6:
                continue
            value = best[start_index]
            if not math.isfinite(value):
                continue
            boundary_reward = 0.0 if end_index == len(points) - 1 else points[end_index].score
            value += boundary_reward - _boundary_penalty(duration, target_seconds) - 18.0
            if value > best[end_index]:
                best[end_index] = value
                previous[end_index] = start_index

    if previous[-1] < 0:
        raise AudioError("5～10秒の範囲で楽曲全体を分割できませんでした")

    path: list[int] = []
    cursor = len(points) - 1
    while cursor >= 0:
        path.append(cursor)
        if cursor == 0:
            break
        cursor = previous[cursor]
    path.reverse()

    scenes: list[dict[str, Any]] = []
    rms_db = analysis.get("rms_db", [])
    window = float(analysis.get("window_seconds", 0.1))
    for index, (left, right) in enumerate(pairwise(path), start=1):
        start = points[left].time
        end = points[right].time
        lo = max(0, int(start / window))
        hi = min(len(rms_db), max(lo + 1, math.ceil(end / window)))
        vocal_ratio = 0.0
        if hi > lo:
            vocal_ratio = sum(float(value) > -35 for value in rms_db[lo:hi]) / (hi - lo)
        scenes.append({
            "index": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "vocal_ratio": round(vocal_ratio, 3),
            "boundary_kind": points[right].kind,
            "boundary_score": round(points[right].score, 2),
        })
    return scenes


def extract_scene_audio(
    vocals: Path, start: float, duration: float, destination: Path,
    *, minimum_h3_seconds: float = 5.0,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    padded = max(minimum_h3_seconds, duration)
    run_command([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(vocals),
        "-af", (
            f"atrim=duration={duration:.6f},asetpts=PTS-STARTPTS,"
            f"apad=whole_dur={padded:.6f}"
        ),
        "-t", f"{padded:.3f}", "-ac", "1", "-ar", "16000", str(destination),
    ])
    return destination


def quantize_scene_plan(scenes: list[dict[str, Any]], fps: int) -> list[dict[str, Any]]:
    """Snap all boundaries to a shared frame grid without accumulating rounding error."""
    if not scenes:
        return []
    boundaries = [float(scenes[0]["start"])] + [float(scene["end"]) for scene in scenes]
    frames = [round(second * fps) for second in boundaries]
    frames[0] = 0
    result: list[dict[str, Any]] = []
    for scene, (start_frame, end_frame) in zip(scenes, pairwise(frames), strict=True):
        item = dict(scene)
        item["start_frame"] = start_frame
        item["end_frame"] = end_frame
        item["frame_count"] = end_frame - start_frame
        item["start"] = round(start_frame / fps, 6)
        item["end"] = round(end_frame / fps, 6)
        item["duration"] = round((end_frame - start_frame) / fps, 6)
        result.append(item)
    return result


def ensure_media_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise AudioError(f"必要なコマンドがありません: {', '.join(missing)}")
