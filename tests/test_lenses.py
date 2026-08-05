"""Telling a whole recording from half of one.

Both Insta360 packings are dual-fisheye, they differ in storage: two circles
side by side in a 2:1 frame, or one circle per square file with a _00_/_10_
pair. Stitching a lone square file produces a panorama whose back half is the
front smeared across it, which reads as a bad stitch rather than the missing
file it is.
"""

import pytest

from accio.core.extract import front_lens, lens_files, lenses_in_frame
from pathlib import Path


@pytest.mark.parametrize("size, circles", [
    ((3840, 1920), 2),      # X3: both lenses in one file
    ((5760, 2880), 2),      # the same packing at 5.7K
    ((2880, 2880), 1),      # X-series: one lens per file
    ((1920, 1920), 1),
])
def test_the_frame_shape_says_how_many_lenses_are_in_it(size, circles):
    assert lenses_in_frame(*size) == circles


def test_a_pair_on_disk_is_a_whole_recording(tmp_path):
    front = tmp_path / "VID_00_001.insv"
    back = tmp_path / "VID_10_001.insv"
    front.touch(), back.touch()
    # one circle per file, but both files are here: two circles in total
    assert lenses_in_frame(2880, 2880) * len(lens_files(front)) == 2


def test_a_lone_front_file_is_half_a_sphere(tmp_path):
    front = tmp_path / "VID_00_001.insv"
    front.touch()
    assert lenses_in_frame(2880, 2880) * len(lens_files(front)) == 1


def test_a_single_file_recording_needs_no_sibling(tmp_path):
    solo = tmp_path / "VID_00_001.insv"
    solo.touch()
    # 2:1 frame: both circles are already in it
    assert lenses_in_frame(3840, 1920) * len(lens_files(solo)) == 2


def test_the_front_file_names_the_walk_either_way():
    assert front_lens([Path("a_10_1.insv"), Path("a_00_1.insv")]) == Path("a_00_1.insv")
    assert front_lens([Path("x3.insv")]) == Path("x3.insv")
