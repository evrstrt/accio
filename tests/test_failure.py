"""A failed run is recorded next to the walk; jobs live only in the runner's memory."""

from accio.jobs.pipeline import (ERROR_FILE, clear_failure, read_failure,
                                 save_failure)


def test_failure_records_stage_and_reason(tmp_path):
    save_failure(tmp_path, "gate", "no such file", "Traceback...")
    f = read_failure(tmp_path)
    assert f["stage"] == "gate"
    assert f["message"] == "no such file"
    assert f["detail"].startswith("Traceback")
    assert f["at"]


def test_failure_written_without_walk_dir(tmp_path):
    out = tmp_path / "never-got-going"
    save_failure(out, "stitch", "docker not on PATH")
    assert read_failure(out)["stage"] == "stitch"


def test_no_failure_reads_none(tmp_path):
    assert read_failure(tmp_path) is None


def test_later_failure_replaces_earlier(tmp_path):
    save_failure(tmp_path, "stitch", "first")
    save_failure(tmp_path, "embed", "second")
    assert read_failure(tmp_path)["stage"] == "embed"


def test_clear_failure(tmp_path):
    save_failure(tmp_path, "gate", "boom")
    clear_failure(tmp_path)
    assert read_failure(tmp_path) is None
    assert not (tmp_path / ERROR_FILE).exists()


def test_clear_failure_idempotent(tmp_path):
    clear_failure(tmp_path)


def test_corrupt_record_reads_none(tmp_path):
    (tmp_path / ERROR_FILE).write_text("{torn")
    assert read_failure(tmp_path) is None
    (tmp_path / ERROR_FILE).write_text("[1, 2]")
    assert read_failure(tmp_path) is None
