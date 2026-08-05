"""What survives a process that does not.

The pipeline writes its own failures, but only from an except block, so the
one failure it cannot record is the one that kills the process. These are the
paths that turn that into something a user can act on.
"""

import json

import pytest

from accio.jobs.pipeline import ERROR_FILE, read_failure
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


def test_a_walk_killed_mid_run_comes_back_retryable(tmp_path):
    """Panoramas, no manifest, no failure: every route refused this, and the
    only working action also deleted the original video."""
    d = walk(tmp_path, "interrupted")
    (d / "pano").mkdir()
    Runner(tmp_path).recover()

    failure = read_failure(d)
    assert failure is not None
    assert failure["stage"] == "stitch"       # what retry re-enters at
    assert "stopped mid-run" in failure["message"]


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
