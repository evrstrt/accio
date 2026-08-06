"""The blur stage vetoes; it does not thin.

The property that matters is not which frames it drops but that the number it
keeps is a fact about the footage rather than about the parameters. The old
windowed argmax kept frames-in over four whatever the walk looked like, which
is how 421 of 562 panoramas were discarded on a walk where 93 of them were
sharper than the median frame that survived.
"""

import cv2
import numpy as np
import pytest

from accio.core.blur import (central_band, legible, local_max, score_pano,
                             vol_score)
from accio.core.params import GateParams

RNG = np.random.default_rng(0)


def test_blur_lowers_the_score():
    sharp = RNG.integers(0, 256, (200, 400), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 3)
    assert vol_score(blurred) < 0.5 * vol_score(sharp)


def test_central_band_excludes_poles():
    img = np.arange(100, dtype=np.uint8).repeat(10).reshape(100, 10)
    band = central_band(img, (0.25, 0.75))
    assert band.shape[0] == 50
    assert band[0, 0] == 25


def test_score_pano_runs_on_bgr():
    bgr = RNG.integers(0, 256, (100, 200, 3), dtype=np.uint8)
    assert score_pano(bgr, GateParams()) > 0


# --- the reference ---------------------------------------------------------

def test_local_max_is_centred_not_trailing():
    """A smear is bracketed by sharp frames; a trailing window only sees in."""
    assert list(local_max([1.0, 9.0, 1.0], window=3)) == [9.0, 9.0, 9.0]


def test_local_max_clips_at_the_ends_rather_than_padding():
    assert list(local_max([5.0, 1.0, 1.0, 1.0, 1.0], window=3)) == \
        [5.0, 5.0, 1.0, 1.0, 1.0]


def test_local_max_of_one_is_the_frame_itself():
    assert list(local_max([3.0, 7.0], window=1)) == [3.0, 7.0]


# --- the veto --------------------------------------------------------------

def test_a_sharp_walk_loses_nothing():
    """The point of the rewrite. Uniformly sharp footage keeps every frame,
    where the argmax kept a quarter of it."""
    scores = [100.0, 110.0, 95.0, 105.0, 98.0, 102.0, 99.0, 101.0]
    assert legible(scores, window=9, floor=0.40) == [True] * 8


def test_a_smear_between_sharp_frames_is_vetoed():
    scores = [100.0, 100.0, 12.0, 100.0, 100.0]
    assert legible(scores, window=5, floor=0.40) == \
        [True, True, False, True, True]


def test_flat_but_legible_survives():
    """Bare concrete scores low everywhere. Judged against its neighbours it
    is fine, which is the whole reason the reference is local."""
    scores = [40.0, 38.0, 41.0, 39.0, 40.0]
    assert all(legible(scores, window=5, floor=0.40))


def test_a_run_of_smears_does_not_certify_itself():
    """The local max inside a long smear is a smear, so the neighbourhood
    rule alone would pass all of it. The walk median is the second opinion."""
    scores = [200.0] * 6 + [3.0, 2.5, 3.2, 2.8] + [200.0] * 6
    got = legible(scores, window=3, floor=0.40)
    assert got[6:10] == [False] * 4
    assert all(got[:6]) and all(got[10:])


def test_the_count_follows_the_footage_not_the_window():
    """The property the old gate did not have: same frames, any window, and
    the number that survive does not move."""
    scores = [100.0, 95.0, 11.0, 98.0, 103.0, 97.0, 9.0, 99.0]
    counts = {w: sum(legible(scores, window=w, floor=0.40)) for w in (3, 5, 9, 21)}
    assert set(counts.values()) == {6}


def test_floor_of_zero_vetoes_nothing():
    assert all(legible([100.0, 0.0, 50.0], window=3, floor=0.0))


def test_a_floor_at_or_above_one_is_refused():
    """At 1 only the local maximum survives, which is the argmax again."""
    with pytest.raises(ValueError):
        legible([1.0, 2.0], window=3, floor=1.0)


def test_empty_and_single_frame_walks():
    assert legible([], window=9, floor=0.40) == []
    assert legible([42.0], window=9, floor=0.40) == [True]
