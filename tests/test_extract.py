from pathlib import Path

from accio.core.extract import (frame_numbers, front_lens, lens_files,
                                pano_frames, sdk_cmd)
from accio.core.params import ExtractParams


def test_frame_numbers_decimates_to_target_fps():
    nos = frame_numbers(native_fps=29.97, nframes=300, target_fps=2.0)
    assert nos[:3] == [0, 15, 30]          # step = round(29.97 / 2) = 15
    assert all(b - a == 15 for a, b in zip(nos, nos[1:]))


def test_frame_numbers_never_upsamples():
    assert frame_numbers(native_fps=1.0, nframes=5, target_fps=2.0) == [0, 1, 2, 3]


def test_lens_files_pairs_dual_file_recordings(tmp_path):
    front = tmp_path / "VID_20260722_00_001.insv"
    back = tmp_path / "VID_20260722_10_001.insv"
    front.touch(), back.touch()
    assert lens_files(front) == [front, back]


def test_lens_files_single_file_recording(tmp_path):
    solo = tmp_path / "VID_20260722_00_001.insv"
    solo.touch()
    assert lens_files(solo) == [solo]


def test_front_lens_names_the_walk():
    front = Path("VID_20260722_00_001.insv")
    back = Path("VID_20260722_10_001.insv")
    assert front_lens([back, front]) == front       # order-independent
    assert front_lens([Path("X3_single.insv")]) == Path("X3_single.insv")


def test_sdk_cmd_flowstate_on_and_inputs_not_last(tmp_path):
    cmd = sdk_cmd(tmp_path / "walk.insv", tmp_path, [0, 15], "sdk-walk",
                  ExtractParams())
    assert "-enable_flowstate" in cmd
    assert cmd[cmd.index("-stitch_type") + 1] == "optflow"
    # the example binary's -inputs parser eats args until the next dash,
    # so the input paths must be followed by another flag
    i = cmd.index("-inputs") + 1
    while cmd[i].startswith("/in/"):
        i += 1
    assert cmd[i].startswith("-")


def test_pano_frames_ordering_and_timestamps(tmp_path):
    for i in (2, 0, 1, 10):
        (tmp_path / f"pano_{i:05d}.jpg").touch()
    frames = pano_frames(tmp_path, fps=2.0)
    assert [f.index for f in frames] == [0, 1, 2, 10]
    assert [f.t_sec for f in frames] == [0.0, 0.5, 1.0, 5.0]


def test_pano_frames_empty_dir(tmp_path):
    assert pano_frames(tmp_path, fps=2.0) == []
