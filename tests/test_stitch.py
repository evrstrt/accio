"""What the stitch does when the container misbehaves.

The SDK hangs on teardown as a matter of course, Docker is not always running,
and a killed container leaves whatever it was mid-write. All three used to end
in the same message, a frame count, which pointed at the video when the
problem was the machine.
"""

import time

import pytest

from accio.core import extract
from accio.core.extract import JPEG_EOI, run_sdk, whole_jpeg


@pytest.fixture(autouse=True)
def brisk(monkeypatch):
    """These exercise the give-up paths, and the real waits are minutes."""
    monkeypatch.setattr(extract, "SDK_POLL", 0.05)
    monkeypatch.setattr(extract, "SDK_KILL_GRACE", 0.2)


def jpeg(path, whole=True):
    path.write_bytes(b"\xff\xd8" + b"\x00" * 400 + (JPEG_EOI if whole else b""))
    return path


# --- a frame is only a frame once it is finished ---------------------------

def test_a_truncated_frame_is_not_counted(tmp_path):
    """The killed-mid-write case. Counting it made the stitch report success
    and left cv2.imread to return None three stages later."""
    assert whole_jpeg(jpeg(tmp_path / "ok.jpg"))
    assert not whole_jpeg(jpeg(tmp_path / "cut.jpg", whole=False))


def test_an_empty_or_missing_frame_is_not_counted(tmp_path):
    (tmp_path / "empty.jpg").write_bytes(b"")
    assert not whole_jpeg(tmp_path / "empty.jpg")
    assert not whole_jpeg(tmp_path / "gone.jpg")


def test_an_unreadable_image_names_itself(tmp_path):
    """Rather than an AttributeError blamed on whichever stage touched it."""
    from accio.jobs.pipeline import imread
    bad = tmp_path / "pano_00007.jpg"
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(RuntimeError, match="pano_00007.jpg is not a readable"):
        imread(bad)


# --- waiting on the frames, not on the process -----------------------------

def test_a_container_that_hangs_after_writing_everything_is_a_success():
    """The documented SDK behaviour: it finishes the export and never exits.
    Waiting for exit turned that into the full timeout, twice."""
    calls = []

    def landed():
        calls.append(1)
        return True if len(calls) > 1 else 0

    why = run_sdk(["sleep", "30"], "cname", landed, stall=60, cap=60)
    assert why == ""
    assert len(calls) < 5          # it stopped as soon as the frames were in


def test_a_container_that_stops_making_progress_is_given_up_on():
    why = run_sdk(["sleep", "30"], "cname", lambda: 3, stall=0.1, cap=60)
    assert "stopped making progress" in why


def test_a_container_that_exits_nonzero_says_so_with_its_output(tmp_path):
    """Docker Desktop not running exits 125, and used to surface as a frame
    count that pointed at the video."""
    why = run_sdk(["sh", "-c", "echo 'Cannot connect to the Docker daemon' >&2; "
                               "exit 125"], "cname", lambda: 0, stall=30, cap=30)
    assert "exited 125" in why
    assert "Cannot connect to the Docker daemon" in why


def test_a_run_that_exits_cleanly_with_every_frame_is_a_success():
    assert run_sdk(["true"], "cname", lambda: True, stall=30, cap=30) == ""


def test_the_cap_bounds_a_container_that_dribbles_frames_forever():
    """A stall detector alone cannot catch one that writes just often enough
    to look alive."""
    n = iter(range(1, 10_000))

    why = run_sdk(["sleep", "30"], "cname", lambda: next(n), stall=30, cap=0.1)
    assert "stopped making progress" in why


def test_the_stall_window_is_generous_enough_for_a_real_stitch():
    """Measured 0.41 to 0.78 s per panorama, so a five-minute gap is stuck."""
    assert extract.SDK_STALL >= 120
    assert extract.SDK_CAP_PER_FRAME >= 3.0


def test_giving_up_does_not_then_wait_on_the_process_forever():
    """docker kill is not guaranteed to land, and waiting on the client after
    it would undo the deadline that was just enforced."""
    start = time.monotonic()
    why = run_sdk(["sleep", "60"], "cname", lambda: 0, stall=0.1, cap=60)
    assert "stopped making progress" in why
    assert time.monotonic() - start < 10        # not the full 60
