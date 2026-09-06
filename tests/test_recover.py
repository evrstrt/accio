"""The one failure the pipeline cannot record is the one that kills the process."""

import json
import time

import pytest

from accio.core import extract
from accio.jobs.pipeline import (ERROR_FILE, JOB_FILE, read_failure, read_job,
                                 save_job)
from accio.jobs.runner import JOBS_KEPT, Runner


def drained(r: Runner, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while r._q.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert r._q.unfinished_tasks == 0


def walk(root, name, *, source=True, manifest=False, error=False):
    d = root / name
    d.mkdir(parents=True)
    if source:
        (d / "source.json").write_text(json.dumps({"fps": 30.0, "frames": 100}))
    if manifest:
        (d / "manifest.csv").write_text("pano_idx,t_sec,yaw,path,kept\n")
    if error:
        (d / ERROR_FILE).write_text(json.dumps(
            {"stage": "embed", "message": "cuda oom", "detail": "", "at": "x"}))
    return d


def test_killed_in_stitch_retries_from_stitch(tmp_path):
    """Panoramas, no manifest, no failure: a state no route accepts unaided."""
    d = walk(tmp_path, "interrupted")
    (d / "pano").mkdir()
    Runner(tmp_path).recover()

    failure = read_failure(d)
    assert failure is not None
    assert failure["stage"] == "stitch"
    assert "stopped mid-run" in failure["message"]


def test_killed_after_stitch_skips_stitch(tmp_path):
    """panos.json marks a finished stitch; re-entering at the gate skips Docker."""
    d = walk(tmp_path, "died-in-embed")
    (d / "pano").mkdir()
    (d / "pano" / extract.PANO_INDEX).write_text("[]")
    Runner(tmp_path).recover()
    assert read_failure(d)["stage"] == "gate"


def test_finished_walk_untouched(tmp_path):
    d = walk(tmp_path, "finished", manifest=True)
    Runner(tmp_path).recover()
    assert read_failure(d) is None


def test_recorded_failure_kept(tmp_path):
    """Overwriting the stage with 'stitch' would turn a re-embed into a full re-run."""
    d = walk(tmp_path, "broken", error=True)
    Runner(tmp_path).recover()
    assert read_failure(d)["stage"] == "embed"
    assert read_failure(d)["message"] == "cuda oom"


def test_non_walk_dir_ignored(tmp_path):
    (tmp_path / "stray").mkdir()
    (tmp_path / "loose.txt").write_text("x")
    Runner(tmp_path).recover()
    assert not (tmp_path / "stray" / ERROR_FILE).exists()


def test_recover_missing_root(tmp_path):
    Runner(tmp_path / "nothing" / "here").recover()
    Runner(tmp_path).recover()


def test_killed_queued_job_fails_visibly(tmp_path):
    """A job killed before it starts leaves a video and a DB row with no directory."""
    save_job(tmp_path / "never-started", None)
    Runner(tmp_path).recover()
    assert read_failure(tmp_path / "never-started")["stage"] == "stitch"
    assert read_job(tmp_path / "never-started") is None


def test_killed_rerun_not_finished(tmp_path):
    """A re-run rebuilds faces/ before the manifest, so a stale manifest survives the kill."""
    d = walk(tmp_path, "rerun-died", manifest=True)
    save_job(d, "faces")
    Runner(tmp_path).recover()
    assert read_failure(d)["stage"] == "faces"
    assert read_job(d) is None


def test_submit_writes_sentinel_first(tmp_path, monkeypatch):
    monkeypatch.setattr(Runner, "_work", lambda self: None)
    r = Runner(tmp_path)
    r.submit(tmp_path / "w1.insv")
    assert read_job(tmp_path / "w1") is not None


def test_resolved_job_clears_sentinel(tmp_path, monkeypatch):
    """A sentinel left behind would re-mark the walk broken on the next restart."""
    monkeypatch.setattr(Runner, "_work", lambda self: None)
    r = Runner(tmp_path)
    job = r.submit(tmp_path / "no-such.insv")
    r._run_one(job)
    assert job.status == "error"
    assert read_failure(tmp_path / "no-such") is not None
    assert read_job(tmp_path / "no-such") is None


def test_failure_records_innermost_frames(tmp_path,
                                                                monkeypatch):
    """format_exc(limit=2) keeps the two outermost frames, never the raise."""
    monkeypatch.setattr(Runner, "_work", lambda self: None)
    r = Runner(tmp_path)
    job = r.submit(tmp_path / "no-such.insv")
    r._run_one(job)
    assert "in probe" in job.error
    assert read_failure(tmp_path / "no-such")["detail"] == job.error


def test_corrupt_job_file_runner_starts(tmp_path):
    d = walk(tmp_path, "odd")
    (d / JOB_FILE).write_text("{not json")
    Runner(tmp_path)


def test_worker_survives_handler_error(tmp_path, monkeypatch):
    """A failing handler (disk full) must not kill the worker thread."""
    ran = []

    def handler(self, job):
        ran.append(job.walkId)
        if job.walkId == "boom":
            raise OSError("No space left on device")

    monkeypatch.setattr(Runner, "_run_one", handler)
    r = Runner(tmp_path)
    r.submit(tmp_path / "boom.insv")
    drained(r)
    assert r.alive()

    r.submit(tmp_path / "after.insv")
    drained(r)
    assert ran == ["boom", "after"]
    assert r.alive()


def test_finished_jobs_pruned_to_fifty(tmp_path, monkeypatch):
    monkeypatch.setattr(Runner, "_work", lambda self: None)
    r = Runner(tmp_path)
    r.submit(tmp_path / "w0.insv").status = "running"
    for i in range(1, JOBS_KEPT + 10):
        r.submit(tmp_path / f"w{i}.insv").status = "done"
    r.submit(tmp_path / "last.insv")
    ids = [j["id"] for j in r.list()]
    assert len(ids) == JOBS_KEPT + 2
    assert ids[0] == JOBS_KEPT + 11
    assert 1 in ids and 2 not in ids
    assert r.busy("w0")


def test_late_rerun_needs_no_video(tmp_path, monkeypatch):
    monkeypatch.setattr(Runner, "_work", lambda self: None)
    r = Runner(tmp_path)
    job = r.submit(None, first="select", walk_id="w1")
    assert (job.walkId, job.video) == ("w1", None)
    with pytest.raises(ValueError):
        r.submit(None)


def test_incoming_is_reaped_on_start(tmp_path, monkeypatch):
    from accio.server import app as server

    staging = tmp_path / ".incoming" / "tmpabc"
    staging.mkdir(parents=True)
    (staging / "half.insv").write_bytes(b"0" * 16)
    monkeypatch.setattr(server, "INCOMING", tmp_path / ".incoming")
    monkeypatch.setattr(server, "EXPORT_TMP", tmp_path / ".exports")

    server.reap_incoming()
    assert not (tmp_path / ".incoming").exists()
