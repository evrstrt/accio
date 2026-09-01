"""What survives a process that does not.

The pipeline writes its own failures, but only from an except block, so the
one failure it cannot record is the one that kills the process. These are the
paths that turn that into something a user can act on.
"""

import json

import pytest

from accio.core import extract
from accio.jobs.pipeline import (ERROR_FILE, read_failure, read_job, save_job)
from accio.jobs.runner import Runner


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


def test_a_walk_killed_inside_the_stitch_retries_from_the_stitch(tmp_path):
    """Panoramas, no manifest, no failure: every route refused this, and the
    only working action also deleted the original video."""
    d = walk(tmp_path, "interrupted")
    (d / "pano").mkdir()
    Runner(tmp_path).recover()

    failure = read_failure(d)
    assert failure is not None
    assert failure["stage"] == "stitch"       # what retry re-enters at
    assert "stopped mid-run" in failure["message"]


def test_a_walk_killed_after_the_stitch_does_not_pay_for_it_twice(tmp_path):
    """panos.json is written when the stitch finishes, and a finished stitch
    is not reused by name: its output has been renamed out of the digit form
    the skip looks for. Re-entering at the gate is the difference between a
    click and another five minutes of Docker."""
    d = walk(tmp_path, "died-in-embed")
    (d / "pano").mkdir()
    (d / "pano" / extract.PANO_INDEX).write_text("[]")
    Runner(tmp_path).recover()
    assert read_failure(d)["stage"] == "gate"


def test_a_finished_walk_is_left_alone(tmp_path):
    d = walk(tmp_path, "finished", manifest=True)
    Runner(tmp_path).recover()
    assert read_failure(d) is None


def test_a_walk_that_already_recorded_why_it_broke_keeps_its_reason(tmp_path):
    """The recorded stage is what retry uses, so overwriting it with 'stitch'
    would turn a cheap re-embed into a full re-run."""
    d = walk(tmp_path, "broken", error=True)
    Runner(tmp_path).recover()
    assert read_failure(d)["stage"] == "embed"
    assert read_failure(d)["message"] == "cuda oom"


def test_a_directory_that_is_not_a_walk_is_ignored(tmp_path):
    (tmp_path / "stray").mkdir()
    (tmp_path / "loose.txt").write_text("x")
    Runner(tmp_path).recover()
    assert not (tmp_path / "stray" / ERROR_FILE).exists()


def test_recovering_an_empty_or_missing_root_is_not_an_error(tmp_path):
    Runner(tmp_path / "nothing" / "here").recover()
    Runner(tmp_path).recover()


def test_a_job_killed_while_still_queued_becomes_a_visible_failure(tmp_path):
    """Jobs live in the runner's memory. One killed before it started used to
    leave a video and a DB row with no directory: not listed, not retryable,
    not deletable. The sentinel submit() writes is the trace recover() needs."""
    save_job(tmp_path / "never-started", None)
    Runner(tmp_path).recover()
    assert read_failure(tmp_path / "never-started")["stage"] == "stitch"
    assert read_job(tmp_path / "never-started") is None


def test_a_rerun_killed_mid_stage_is_not_mistaken_for_a_finished_walk(tmp_path):
    """A re-run rebuilds faces/ before it rewrites the manifest, so last run's
    manifest survives the crash and used to read as 'this walk is fine'."""
    d = walk(tmp_path, "rerun-died", manifest=True)
    save_job(d, "faces")
    Runner(tmp_path).recover()
    assert read_failure(d)["stage"] == "faces"    # where retry re-enters
    assert read_job(d) is None


def test_submit_puts_the_sentinel_on_disk_before_the_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(Runner, "_work", lambda self: None)
    r = Runner(tmp_path)
    r.submit(tmp_path / "w1.insv")
    assert read_job(tmp_path / "w1") is not None


def test_a_job_that_resolves_clears_its_sentinel(tmp_path, monkeypatch):
    """Done or failed, the walk's own files now say what happened; a sentinel
    left behind would re-mark the walk broken on the next restart."""
    monkeypatch.setattr(Runner, "_work", lambda self: None)
    r = Runner(tmp_path)
    job = r.submit(tmp_path / "no-such.insv")
    r._run_one(job)
    assert job.status == "error"
    assert read_failure(tmp_path / "no-such") is not None
    assert read_job(tmp_path / "no-such") is None


def test_the_worker_survives_a_handler_that_fails(tmp_path):
    """The likeliest reason a run failed is a full disk, which is also the
    likeliest reason writing the failure fails. That used to kill the thread
    and every later walk sat queued forever with no error."""
    r = Runner(tmp_path)
    boom = type("Job", (), {"walkId": "x", "status": "queued", "stages": {},
                            "stats": {}, "error": "", "params": None,
                            "first": None, "video": tmp_path / "no.insv"})()

    def explode(_job):
        raise OSError("No space left on device")

    r._run_one = explode
    with pytest.raises(OSError):
        r._run_one(boom)           # the handler really does raise
    # and the loop swallows it rather than exiting
    r._q.put(boom)
    assert r._q.get(timeout=2) is boom


def test_incoming_is_reaped_on_start(tmp_path, monkeypatch):
    from accio.server import app as server

    staging = tmp_path / "videos" / ".incoming" / "tmpabc"
    staging.mkdir(parents=True)
    (staging / "half.insv").write_bytes(b"0" * 16)
    monkeypatch.setattr(server, "VIDEO_DIR", tmp_path / "videos")

    server.reap_incoming()
    assert not (tmp_path / "videos" / ".incoming").exists()
