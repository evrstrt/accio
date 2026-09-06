"""Sharpness. Variance of Laplacian measures blur and texture together, so
every comparison here has to hold the wall constant.
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


def test_sharp_walk_loses_nothing():
    """A windowed argmax keeps a quarter of uniformly sharp footage."""
    assert legible([100.0, 110.0, 95.0, 105.0, 98.0, 102.0], dead=0.15) == \
        [True] * 6


def test_smeared_stretch_dropped():
    """Smeared frames group with each other, so nothing downstream catches a stretch."""
    scores = [200.0] * 6 + [3.0, 2.5, 3.2, 2.8] + [200.0] * 6
    got = legible(scores, dead=0.15)
    assert got[6:10] == [False] * 4
    assert all(got[:6]) and all(got[10:])


def test_lone_smear_left_for_select():
    assert all(legible([100.0, 100.0, 45.0, 100.0, 100.0], dead=0.15))


def test_count_follows_footage():
    scores = [100.0, 95.0, 2.0, 98.0, 103.0, 97.0, 1.0, 99.0]
    counts = {d: sum(legible(scores, dead=d)) for d in (0.05, 0.1, 0.15, 0.2)}
    assert set(counts.values()) == {6}


def test_dead_of_zero_vetoes_nothing():
    assert all(legible([100.0, 0.0, 50.0], dead=0.0))


def test_dead_above_one_refused():
    with pytest.raises(ValueError):
        legible([1.0, 2.0], dead=1.0)


def test_empty_and_single_frame_walks():
    assert legible([], dead=0.15) == []
    assert legible([42.0], dead=0.15) == [True]


def test_shipped_gate_valve():
    g = GateParams()
    walk = list(100.0 + RNG.normal(0, 25, 200).clip(-70, None))
    assert all(legible(walk, g.dead))


def test_plain_wall_ratio_near_one():
    """An absolute threshold picks off a sharp frame of bare concrete."""
    yaw = np.array([45, 135] * 5)
    t = np.repeat(np.arange(5.0), 2)
    sharp = np.array([200.0, 40.0] * 5)          # busy heading, plain heading
    r = heading_ratio(sharp, yaw, t, span=45.0)
    assert np.allclose(r, 1.0)


def test_smear_relative_to_heading():
    yaw = np.array([45] * 5)
    t = np.arange(5.0)
    sharp = np.array([100.0, 100.0, 25.0, 100.0, 100.0])
    r = heading_ratio(sharp, yaw, t, span=45.0)
    assert r[2] == pytest.approx(0.25)
    assert np.allclose(r[[0, 1, 3, 4]], 1.0)


def test_headings_judged_separately():
    yaw = np.array([45, 135, 45, 135])
    t = np.array([0.0, 0.0, 1.0, 1.0])
    sharp = np.array([200.0, 50.0, 100.0, 25.0])
    r = heading_ratio(sharp, yaw, t, span=45.0)
    assert r[2] == pytest.approx(2 / 3)
    assert r[3] == pytest.approx(2 / 3)


def test_span_excludes_distant_room():
    yaw = np.array([45] * 6)
    t = np.array([0.0, 1.0, 2.0, 500.0, 501.0, 502.0])
    sharp = np.array([100.0, 100.0, 100.0, 20.0, 20.0, 20.0])
    r = heading_ratio(sharp, yaw, t, span=45.0)
    assert np.allclose(r, 1.0)
    wide = heading_ratio(sharp, yaw, t, span=1000.0)
    assert not np.allclose(wide, 1.0)


def test_zero_reference_no_divide_by_zero():
    r = heading_ratio(np.zeros(3), np.array([45] * 3), np.arange(3.0), span=45.0)
    assert np.allclose(r, 1.0)
