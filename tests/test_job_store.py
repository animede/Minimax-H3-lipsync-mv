from __future__ import annotations

from app.job_store import JobStore
from app.models import Job


def test_generation_information_records_durations_without_clock_timestamps(tmp_path) -> None:
    jobs = JobStore(tmp_path)
    job = jobs.create("character.png", "song.wav", "")

    jobs.start_run(job.id)
    jobs.begin_timing(job.id, "probing", "入力ファイルを検証しています")
    jobs.record_timing(job.id, "generating_scene", "H3シーン 1 / 1", 12.345, kind="scene")
    jobs.log(job.id, "検証が完了しました")
    jobs.finish_run(job.id)

    saved = jobs.get(job.id)
    assert saved is not None
    assert saved.logs == ["検証が完了しました"]
    assert saved.timings[0] == {
        "stage": "generating_scene",
        "label": "H3シーン 1 / 1",
        "duration_s": 12.345,
        "kind": "scene",
    }
    assert saved.timings[1]["stage"] == "probing"
    assert saved.timings[1]["duration_s"] >= 0
    assert saved.total_elapsed_s >= 0


def test_legacy_clock_logs_are_migrated_to_durations() -> None:
    job = Job(
        id="legacy",
        logs=[
            "[18:12:32] 入力ファイルを検証しています",
            "[18:12:36] ボーカルを分離しています",
            "[18:13:40] MVが完成しました",
        ],
    )
    JobStore._migrate_legacy_timings(job)
    assert [item["duration_s"] for item in job.timings] == [4.0, 64.0]
    assert job.total_elapsed_s == 68.0
