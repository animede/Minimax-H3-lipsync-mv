from __future__ import annotations

import json
import re
import threading
import time
import uuid
from itertools import pairwise
from pathlib import Path
from typing import Any

from .config import settings
from .models import Job


class JobStore:
    def __init__(self, root: Path = settings.jobs_dir) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for manifest in self.root.glob("*/job.json"):
            try:
                job = Job.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
                self._migrate_legacy_timings(job)
                if job.status == "running":
                    job.status = "failed"
                    job.stage = "interrupted"
                    job.error = "サーバー再起動によりジョブが中断されました"
                self._jobs[job.id] = job
                self.save(job)
            except (OSError, ValueError, TypeError):
                continue

    @staticmethod
    def _migrate_legacy_timings(job: Job) -> None:
        """Convert the former [HH:MM:SS] log display into approximate durations."""
        if job.timings:
            return
        entries: list[tuple[int, str]] = []
        for line in job.logs:
            match = re.match(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+)$", line)
            if not match:
                continue
            second = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
            entries.append((second, match.group(4)))
        if len(entries) < 2:
            return
        for (started, label), (ended, _) in pairwise(entries):
            duration = ended - started
            if duration < 0:
                duration += 24 * 3600
            is_scene = bool(re.match(r"^シーン\d+:", label))
            job.timings.append({
                "stage": "generating_scene" if is_scene else "legacy_stage",
                "label": label,
                "duration_s": float(duration),
                "kind": "scene" if is_scene else "stage",
            })
        total = entries[-1][0] - entries[0][0]
        if total < 0:
            total += 24 * 3600
        job.total_elapsed_s = float(total)

    def create(
        self, character_file: str, song_file: str, concept: str, *,
        input_mode: str = "music", text_file: str = "", source_text: str = "",
        width: int = 1024, height: int = 768,
    ) -> Job:
        now = time.time()
        job = Job(
            id=uuid.uuid4().hex[:16],
            created_at=now,
            updated_at=now,
            character_file=character_file,
            song_file=song_file,
            concept=concept.strip(),
            input_mode=input_mode,
            text_file=text_file,
            source_text=source_text,
            width=width,
            height=height,
        )
        with self._lock:
            self._jobs[job.id] = job
            self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
            self.save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def save(self, job: Job) -> None:
        with self._lock:
            job.updated_at = time.time()
            path = self.job_dir(job.id) / "job.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    def update(self, job_id: str, **fields: Any) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in fields.items():
                if not hasattr(job, key):
                    raise AttributeError(key)
                setattr(job, key, value)
            self.save(job)
            return job

    def start_run(self, job_id: str, *, reset: bool = True) -> None:
        """Start duration accounting for one pipeline run."""
        with self._lock:
            job = self._jobs[job_id]
            if reset:
                job.timings = []
                job.logs = []
            job.timing_stage = ""
            job.timing_label = ""
            job.timing_started_at = 0.0
            job.run_started_at = time.time()
            job.total_elapsed_s = 0.0
            self.save(job)

    def _finish_timing_locked(self, job: Job, now: float) -> None:
        if not job.timing_stage or job.timing_started_at <= 0:
            return
        job.timings.append({
            "stage": job.timing_stage,
            "label": job.timing_label or job.timing_stage,
            "duration_s": round(max(0.0, now - job.timing_started_at), 3),
            "kind": "stage",
        })
        job.timing_stage = ""
        job.timing_label = ""
        job.timing_started_at = 0.0

    def begin_timing(self, job_id: str, stage: str, label: str) -> None:
        """Finish the previous stage and begin timing the next one."""
        with self._lock:
            job = self._jobs[job_id]
            now = time.time()
            self._finish_timing_locked(job, now)
            job.timing_stage = stage
            job.timing_label = label
            job.timing_started_at = now
            self.save(job)

    def record_timing(
        self, job_id: str, stage: str, label: str, duration_s: float, *, kind: str = "detail"
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.timings.append({
                "stage": stage,
                "label": label,
                "duration_s": round(max(0.0, duration_s), 3),
                "kind": kind,
            })
            self.save(job)

    def finish_run(self, job_id: str) -> None:
        """Close the active stage and store the total run duration."""
        with self._lock:
            job = self._jobs[job_id]
            now = time.time()
            self._finish_timing_locked(job, now)
            if job.run_started_at > 0:
                job.total_elapsed_s = round(max(0.0, now - job.run_started_at), 3)
            job.timing_stage = ""
            job.timing_label = ""
            job.timing_started_at = 0.0
            self.save(job)

    def log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(message)
            job.logs = job.logs[-300:]
            self.save(job)


store = JobStore()
