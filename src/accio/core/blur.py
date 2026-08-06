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


def legible(scores: list[float], window: int, floor: float,
            dead: float) -> list[bool]:
    """Which frames are salvage, not which are good. Select judges good.

    Two rules, because they answer different questions and used to share one
    number. `floor` is against the neighbourhood, and catches a smear between
    sharp frames. `dead` is against the walk, and catches the case the first
    one cannot see: a stretch where every frame is smeared, so the local
    maximum is a smear too and the whole run certifies itself.

    Both are deliberately low. Select picks the sharpest member of each group,
    so anything with a sharper twin is already handled and never reaches a
    labeller; what is left for these rules is frames nothing can improve on.
    """
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor is a fraction of the local max, in [0, 1)")
    if not 0.0 <= dead < 1.0:
        raise ValueError("dead is a fraction of the walk median, in [0, 1)")
    if not scores:
        return []
    ref = local_max(scores, window)
    absolute = float(np.median(scores)) * dead
    return [bool(s >= r * floor and s >= absolute) for s, r in zip(scores, ref)]
