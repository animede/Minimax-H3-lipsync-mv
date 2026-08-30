from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


@dataclass
class Scene:
    index: int
    start: float
    end: float
    duration: float
    vocal_ratio: float = 0.0
    boundary_kind: str = "fallback"
    boundary_score: float = 0.0
    emotion: str = ""
    prompt: str = ""
    seed: int = 0
    video_file: str = ""


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "待機中"
    created_at: float = 0.0
    updated_at: float = 0.0
    character_file: str = ""
    song_file: str = ""
    input_mode: str = "music"
    text_file: str = ""
    source_text: str = ""
    concept: str = ""
    duration: float = 0.0
    current_scene: int = 0
    scene_count: int = 0
    scenes: list[dict[str, Any]] = field(default_factory=list)
    scenario: dict[str, Any] = field(default_factory=dict)
    output_file: str = ""
    error: str = ""
    cancel_requested: bool = False
    logs: list[str] = field(default_factory=list)
    timings: list[dict[str, Any]] = field(default_factory=list)
    timing_stage: str = ""
    timing_label: str = ""
    timing_started_at: float = 0.0
    run_started_at: float = 0.0
    total_elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})
