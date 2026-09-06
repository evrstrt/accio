"""Insta360 stores both lenses in one 2:1 file or one lens per square file in
a _00_/_10_ pair. A lone square file stitches to a smeared back half.
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
def test_lenses_in_frame(size, circles):
    assert lenses_in_frame(*size) == circles


def test_pair_is_whole_recording(tmp_path):
    front = tmp_path / "VID_00_001.insv"
    back = tmp_path / "VID_10_001.insv"
    front.touch(), back.touch()
    assert lenses_in_frame(2880, 2880) * len(lens_files(front)) == 2


def test_lone_front_file_half_sphere(tmp_path):
    front = tmp_path / "VID_00_001.insv"
    front.touch()
    assert lenses_in_frame(2880, 2880) * len(lens_files(front)) == 1


def test_single_file_needs_no_sibling(tmp_path):
    solo = tmp_path / "VID_00_001.insv"
    solo.touch()
    assert lenses_in_frame(3840, 1920) * len(lens_files(solo)) == 2


def test_front_file_names_walk():
    assert front_lens([Path("a_10_1.insv"), Path("a_00_1.insv")]) == Path("a_00_1.insv")
    assert front_lens([Path("x3.insv")]) == Path("x3.insv")
