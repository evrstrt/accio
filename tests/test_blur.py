"""Sharpness, and the two comparisons it is allowed to make.

Variance of Laplacian measures blur and texture together, so the raw number
cannot tell "this frame is ruined" from "this wall is plain". Every test here
is really about that: whether a comparison holds the wall constant.

The gate keeps one blunt rule for footage that arrived broken. The comparison
that does the work lives in Select, where a group is one place by construction
and the texture cancels for free.
"""

import cv2
import numpy as np
import pytest

from accio.core.blur import (central_band, heading_ratio, legible, score_pano,
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


# --- the gate: broken footage only -----------------------------------------

def test_a_sharp_walk_loses_nothing():
    """The point of the rewrite. Uniformly sharp footage keeps every frame,
    where the old windowed argmax kept a quarter of it."""
    assert legible([100.0, 110.0, 95.0, 105.0, 98.0, 102.0], dead=0.15) == \
        [True] * 6


def test_a_stretch_smeared_right_through_is_dropped():
    """The one case nothing downstream catches: these frames group with each
    other, because blur is what they have in common, and Select would then
    export a smear as the sharpest of them."""
    scores = [200.0] * 6 + [3.0, 2.5, 3.2, 2.8] + [200.0] * 6
    got = legible(scores, dead=0.15)
    assert got[6:10] == [False] * 4
    assert all(got[:6]) and all(got[10:])


def test_a_lone_smear_is_left_for_select():
    """Not this stage's job. A smear between sharp frames either joins a group
    and loses the exemplar contest, or fails to and is judged by heading."""
    assert all(legible([100.0, 100.0, 45.0, 100.0, 100.0], dead=0.15))


def test_the_count_follows_the_footage_not_the_parameters():
    """The property the windowed argmax could not have."""
    scores = [100.0, 95.0, 2.0, 98.0, 103.0, 97.0, 1.0, 99.0]
    counts = {d: sum(legible(scores, dead=d)) for d in (0.05, 0.1, 0.15, 0.2)}
    assert set(counts.values()) == {6}


def test_dead_of_zero_vetoes_nothing():
    assert all(legible([100.0, 0.0, 50.0], dead=0.0))


def test_a_dead_at_or_above_one_is_refused():
    with pytest.raises(ValueError):
        legible([1.0, 2.0], dead=1.0)


def test_empty_and_single_frame_walks():
    assert legible([], dead=0.15) == []
    assert legible([42.0], dead=0.15) == [True]


def test_the_shipped_gate_is_a_valve():
    """Plausible footage, shipped setting, nothing goes."""
    g = GateParams()
    walk = list(100.0 + RNG.normal(0, 25, 200).clip(-70, None))
    assert all(legible(walk, g.dead))


# --- the heading ratio: what cancels the texture ----------------------------

def test_a_plain_wall_is_not_punished_for_being_plain():
    """The whole reason this exists. One heading faces a busy wall and scores
    high, another faces bare concrete and scores low, and both are pin-sharp.
    An absolute threshold picks the bare wall off; a per-heading one does not.
    """
    yaw = np.array([45, 135] * 5)
    t = np.repeat(np.arange(5.0), 2)
    sharp = np.array([200.0, 40.0] * 5)          # busy heading, plain heading
    r = heading_ratio(sharp, yaw, t, span=45.0)
    assert np.allclose(r, 1.0)                   # neither is called blurred


def test_a_smear_shows_up_against_its_own_heading():
    yaw = np.array([45] * 5)
    t = np.arange(5.0)
    sharp = np.array([100.0, 100.0, 25.0, 100.0, 100.0])
    r = heading_ratio(sharp, yaw, t, span=45.0)
    assert r[2] == pytest.approx(0.25)
    assert np.allclose(r[[0, 1, 3, 4]], 1.0)


def test_headings_are_judged_separately():
    """A busy heading must not set the bar for a plain one."""
    yaw = np.array([45, 135, 45, 135])
    t = np.array([0.0, 0.0, 1.0, 1.0])
    sharp = np.array([200.0, 50.0, 100.0, 25.0])
    r = heading_ratio(sharp, yaw, t, span=45.0)
    # each is half of its own heading's median, not of the walk's
    assert r[2] == pytest.approx(2 / 3)
    assert r[3] == pytest.approx(2 / 3)


def test_the_span_keeps_a_distant_room_out_of_it():
    """Sharpness drifts across a building; the norm should be local in time."""
    yaw = np.array([45] * 6)
    t = np.array([0.0, 1.0, 2.0, 500.0, 501.0, 502.0])
    sharp = np.array([100.0, 100.0, 100.0, 20.0, 20.0, 20.0])
    r = heading_ratio(sharp, yaw, t, span=45.0)
    assert np.allclose(r, 1.0)      # each trio is normal where it sits
    wide = heading_ratio(sharp, yaw, t, span=1000.0)
    assert not np.allclose(wide, 1.0)


def test_a_zero_reference_does_not_divide_by_zero():
    r = heading_ratio(np.zeros(3), np.array([45] * 3), np.arange(3.0), span=45.0)
    assert np.allclose(r, 1.0)
