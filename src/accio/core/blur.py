"""Sharpness scoring.

Variance of Laplacian measures blur and texture at once: a blank wall scores
low when sharp, a busy one scores high when smeared. So the raw number is
never thresholded on its own. It only decides anything in a comparison where
the texture cancels: within a dedup group (same wall, sharpest member wins),
or against the recent norm of the same heading (heading_ratio).

The gate is weak by design: a per-window "keep the sharpest" gate dropped 421
of 562 panoramas on the 7th Floor walk, 93 of them sharper than the median
frame it kept.
"""

import cv2
import numpy as np

from .params import GateParams


def vol_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def central_band(img: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    h = img.shape[0]
    return img[int(h * lo): int(h * hi)]


def score_pano(bgr: np.ndarray, params: GateParams) -> float:
    gray = cv2.cvtColor(central_band(bgr, params.band), cv2.COLOR_BGR2GRAY)
    return vol_score(gray)


def legible(scores: list[float], dead: float) -> list[bool]:
    """Drop panoramas under `dead` x the walk median.

    Only catches a stretch smeared right through, where the frames group with
    each other and Select would anchor on a smear.
    """
    if not 0.0 <= dead < 1.0:
        raise ValueError("dead is a fraction of the walk median, in [0, 1)")
    if not scores:
        return []
    floor = float(np.median(scores)) * dead
    return [bool(s >= floor) for s in scores]


def heading_ratio(sharpness: np.ndarray, yaw: np.ndarray, t_sec: np.ndarray,
                  span: float) -> np.ndarray:
    """Each face's sharpness over the median of its heading within `span` seconds.

    Near 1 is as sharp as this heading usually is; well under 1 is a smear.
    """
    out = np.ones(len(sharpness), dtype=np.float64)
    for heading in np.unique(yaw):
        rows = np.flatnonzero(yaw == heading)
        for i in rows:
            near = rows[np.abs(t_sec[rows] - t_sec[i]) <= span]
            ref = float(np.median(sharpness[near]))
            out[i] = sharpness[i] / ref if ref > 0 else 1.0
    return out
