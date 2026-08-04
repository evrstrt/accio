from pathlib import Path

import pytest

from accio.core.extract import (PanoFrame, frame_numbers, front_lens,
                                lens_files, load_panos, save_panos, sdk_cmd)
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


def written(tmp_path, t_secs):
    frames = []
    for i, t in enumerate(t_secs):
        path = tmp_path / f"pano_{i:05d}.jpg"
        path.touch()
        frames.append(PanoFrame(index=i, t_sec=t, path=path))
    save_panos(tmp_path, frames)
    return frames


def test_panos_round_trip_with_their_true_timestamps(tmp_path):
    # 15-frame decimation of 29.97 fps footage: not the 0.5s a nominal 2 fps
    # would give, which is exactly why the stitch records it
    frames = written(tmp_path, [0.0, 0.5005, 1.001])
    assert load_panos(tmp_path) == frames


def test_load_panos_says_which_panoramas_are_missing(tmp_path):
    written(tmp_path, [0.0, 0.5005])
    (tmp_path / "pano_00001.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="pano_00001"):
        load_panos(tmp_path)


def test_panos_empty_walk(tmp_path):
    save_panos(tmp_path, [])
    assert load_panos(tmp_path) == []
