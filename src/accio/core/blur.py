"""Stage 3: relative motion-blur gate.

Flat bare-RCC concrete has low Laplacian variance even when perfectly sharp,
so an absolute threshold throws away good frames. Instead every panorama gets
a variance-of-Laplacian score and the sharpest one per window of consecutive
frames is kept (windowed argmax, the standard trick from the video-deblurring
literature). Scoring is restricted to the central latitude band: the poles
are projection stretch and helmet.
"""

import cv2
import numpy as np

from .params import GateParams


def vol_score(gray: np.ndarray) -> float:
    """Variance of Laplacian: near zero on flat or smeared areas, high on edges."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def central_band(gray: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    h = gray.shape[0]
    return gray[int(h * lo): int(h * hi)]


def score_pano(bgr: np.ndarray, params: GateParams) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return vol_score(central_band(gray, params.band))


def windowed_keep(scores: list[float], window: int) -> list[bool]:
    """Keep the sharpest frame per window of `window` consecutive scores."""
    if window < 1:
        raise ValueError("window must be >= 1")
    flags = [False] * len(scores)
    for start in range(0, len(scores), window):
        chunk = scores[start:start + window]
        flags[start + int(np.argmax(chunk))] = True
    return flags
