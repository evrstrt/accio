import cv2
import numpy as np

from accio.core.blur import central_band, score_pano, vol_score, windowed_keep
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


def test_windowed_keep_one_per_window():
    scores = [1.0, 5.0, 2.0, 3.0, 9.0, 1.0, 4.0]
    flags = windowed_keep(scores, window=4)
    assert flags == [False, True, False, False, True, False, False]
    assert sum(flags) == 2


def test_windowed_keep_handles_short_tail_and_empty():
    assert windowed_keep([3.0], window=4) == [True]
    assert windowed_keep([], window=4) == []
