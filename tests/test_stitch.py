"""The stitch when the container misbehaves: the SDK hangs on teardown, Docker
may be down, and a killed container leaves torn files.
"""

import time

import pytest

from accio.core import extract
from accio.core.extract import JPEG_EOI, run_sdk, whole_jpeg


@pytest.fixture(autouse=True)
def fast_timeouts(monkeypatch):
    """The real waits are minutes."""
    monkeypatch.setattr(extract, "SDK_POLL", 0.05)
    monkeypatch.setattr(extract, "SDK_KILL_GRACE", 0.2)


def jpeg(path, whole=True):
    path.write_bytes(b"\xff\xd8" + b"\x00" * 400 + (JPEG_EOI if whole else b""))
    return path


def test_truncated_frame_not_counted(tmp_path):
    """A counted torn frame is cv2.imread returning None three stages later."""
    assert whole_jpeg(jpeg(tmp_path / "ok.jpg"))
    assert not whole_jpeg(jpeg(tmp_path / "cut.jpg", whole=False))


def test_empty_or_missing_frame_not_counted(tmp_path):
    (tmp_path / "empty.jpg").write_bytes(b"")
    assert not whole_jpeg(tmp_path / "empty.jpg")
    assert not whole_jpeg(tmp_path / "gone.jpg")


def test_unreadable_image_named(tmp_path):
    from accio.jobs.pipeline import imread
    bad = tmp_path / "pano_00007.jpg"
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(RuntimeError, match="pano_00007.jpg is not a readable"):
        imread(bad)


def test_hang_after_complete_export_succeeds():
    """The SDK finishes the export and never exits."""
    calls = []

    def landed():
        calls.append(1)
        return True if len(calls) > 1 else 0

    why = run_sdk(["sleep", "30"], "cname", landed, stall=60, cap=60)
    assert why == ""
    assert len(calls) < 5


def test_stalled_container_abandoned():
    why = run_sdk(["sleep", "30"], "cname", lambda: 3, stall=0.1, cap=60)
    assert "stopped making progress" in why


def test_nonzero_exit_reports_output(tmp_path):
    """Docker Desktop not running exits 125."""
    why = run_sdk(["sh", "-c", "echo 'Cannot connect to the Docker daemon' >&2; "
                               "exit 125"], "cname", lambda: 0, stall=30, cap=30)
    assert "exited 125" in why
    assert "Cannot connect to the Docker daemon" in why


def test_clean_exit_succeeds():
    assert run_sdk(["true"], "cname", lambda: True, stall=30, cap=30) == ""


def test_cap_bounds_dribbling_container():
    n = iter(range(1, 10_000))

    why = run_sdk(["sleep", "30"], "cname", lambda: next(n), stall=30, cap=0.1)
    assert "stopped making progress" in why


def test_stall_window_covers_real_stitch():
    """Measured 0.41 to 0.78 s per panorama."""
    assert extract.SDK_STALL >= 120
    assert extract.SDK_CAP_PER_FRAME >= 3.0


def test_abandon_does_not_block_on_wait():
    """docker kill is not guaranteed to land."""
    start = time.monotonic()
    why = run_sdk(["sleep", "60"], "cname", lambda: 0, stall=0.1, cap=60)
    assert "stopped making progress" in why
    assert time.monotonic() - start < 10
