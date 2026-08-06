"""Stage 3: motion-blur veto.

Two independent questions run through this pipeline. Is a frame legible at all
(a property of that frame, which is this stage), and does it show anything the
set does not already have (a property of the set, which is Select). This stage
answers only the first, and in particular it does not decide how many frames
survive. Select does that, against a calibrated false-merge budget.

It used to. The gate was a windowed argmax keeping the sharpest panorama per
four, which meant the count out was frames-in over four: a function of the
window and the stitch rate, and of nothing in the picture. Measured on the 7th
Floor walk, that discarded 421 of 562 panoramas, 93 of them sharper than the
median frame it kept, and left 18 panoramas with no exported frame within tau
of them. It also failed hardest exactly where the operator moved fastest,
which is where content per second is highest, so the loss fell on the varied
footage a generalisation set is for.

What is left is a veto. Flat bare-RCC concrete has low Laplacian variance even
when perfectly sharp, so an absolute threshold would throw away good frames;
the reference is the local maximum instead, over a window centred on the frame
itself. A frame below `floor` of that is smeared, not merely flat, and no
labeller can work with it. Everything else goes through. Scoring is restricted
to the central latitude band: the poles are projection stretch and helmet.
"""

import cv2
import numpy as np

from .params import GateParams


def vol_score(gray: np.ndarray) -> float:
    """Variance of Laplacian: near zero on flat or smeared areas, high on edges."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def central_band(img: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    h = img.shape[0]
    return img[int(h * lo): int(h * hi)]


def score_pano(bgr: np.ndarray, params: GateParams) -> float:
    gray = cv2.cvtColor(central_band(bgr, params.band), cv2.COLOR_BGR2GRAY)
    return vol_score(gray)


def local_max(scores: list[float], window: int) -> np.ndarray:
    """The best score within `window` frames centred on each position.

    Centred rather than trailing, because a smear is bracketed by the sharp
    frames on both sides of it and a trailing window only sees the way in.
    Clipped at the ends instead of padded: the first frames of a walk get a
    shorter reference, which is honest, where padding would invent one.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    s = np.asarray(scores, dtype=np.float64)
    half = window // 2
    return np.array([s[max(0, i - half): i + half + 1].max()
                     for i in range(len(s))])


def legible(scores: list[float], window: int, floor: float) -> list[bool]:
    """Which frames are sharp enough to be worth a labeller's time.

    Relative to the neighbourhood, so a dim corridor is judged against the rest
    of that corridor rather than against a bright bay fifty frames away.
    """
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor is a fraction of the local max, in [0, 1)")
    if not scores:
        return []
    ref = local_max(scores, window)
    # a stretch of frames that are all equally smeared has a local max that is
    # itself a smear, and every one of them would pass. The walk median is the
    # second opinion: nothing that dark is legible whatever its neighbours did.
    absolute = float(np.median(scores)) * floor * floor
    return [bool(s >= r * floor and s >= absolute) for s, r in zip(scores, ref)]
