"""A run that broke, recorded next to the walk rather than only in memory.

Jobs live in the runner's memory. Without this, restarting the server loses
every trace of a failed ingest, leaving a directory and a multi-gigabyte video
that nothing in the app can explain or remove.
"""

from accio.jobs.pipeline import (ERROR_FILE, clear_failure, read_failure,
                                 save_failure)


def test_a_failure_says_where_and_why(tmp_path):
    save_failure(tmp_path, "gate", "no such file", "Traceback...")
    f = read_failure(tmp_path)
    assert f["stage"] == "gate"
    assert f["message"] == "no such file"
    assert f["detail"].startswith("Traceback")
    assert f["at"]


def test_it_is_written_even_if_the_walk_made_no_directory(tmp_path):
    """The stitch can break before anything else exists."""
    out = tmp_path / "never-got-going"
    save_failure(out, "stitch", "docker not on PATH")
    assert read_failure(out)["stage"] == "stitch"


def test_a_walk_that_never_broke_has_nothing_to_report(tmp_path):
    assert read_failure(tmp_path) is None


def test_a_later_failure_replaces_an_earlier_one(tmp_path):
    save_failure(tmp_path, "stitch", "first")
    save_failure(tmp_path, "embed", "second")
    assert read_failure(tmp_path)["stage"] == "embed"


def test_clearing_is_what_a_successful_run_does(tmp_path):
    save_failure(tmp_path, "gate", "boom")
    clear_failure(tmp_path)
    assert read_failure(tmp_path) is None
    assert not (tmp_path / ERROR_FILE).exists()


def test_clearing_a_walk_that_never_failed_is_not_an_error(tmp_path):
    clear_failure(tmp_path)
