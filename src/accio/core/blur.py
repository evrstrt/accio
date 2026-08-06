"""Sharpness, and the two places it is allowed to decide anything.

Variance of Laplacian measures blur and texture at the same time. A blank wall
scores low pin-sharp and a busy one scores high smeared, so the raw number
cannot separate "this frame is ruined" from "this wall is plain". Everything
here follows from that, and the whole design is about arranging comparisons
where the texture appears on both sides and cancels.

There are exactly two such comparisons, and neither is a threshold on a walk.

Inside a dedup group, because a group is one place by construction, so its
members are looking at the same wall. `dedup.sharpest` picks the sharpest of
them and the texture cancels for free, with nothing to configure. Measured on
the 7th Floor walk, group members spread 1.97x in sharpness at the median and
3.58x at p90, so this is where nearly all the blur is caught.

Against the recent norm of a face's own heading, which is `heading_ratio`
below. This is for the frames the first comparison cannot reach: a view seen
once, absorbed by nothing, with no sharper twin to be replaced by. Select uses
it on those and only those.

What is deliberately not here is a gate. This stage used to keep the sharpest
panorama per window of four, which made the count out frames-in over four: a
function of the window and the stitch rate and nothing in the picture. On the
7th Floor walk it discarded 421 of 562 panoramas, 93 of them sharper than the
median frame it kept, and left 5.8% of the walk with no exported frame within
tau of it. `legible` is what remains, and its job is small.
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


def legible(scores: list[float], dead: float) -> list[bool]:
    """Panoramas worth stitching faces out of. One rule, and a blunt one.

    Only the case nothing downstream can catch: a stretch of walk smeared
    right through. Those frames group with each other, because blur is what
    they have in common, and Select then exports a smear as the group's
    sharpest member. Everything else is left alone here on purpose, because
    this is the wrong place to judge a frame.

    Why the wrong place: this score is variance of Laplacian over a whole
    panorama, which mixes all four headings into one number, and variance of
    Laplacian answers "how much texture" as much as "how sharp". A blank wall
    scores low pin-sharp; a busy one scores high smeared. Judging a frame on it
    means deleting the plain walls of a building made of plain walls.
    """
    if not 0.0 <= dead < 1.0:
        raise ValueError("dead is a fraction of the walk median, in [0, 1)")
    if not scores:
        return []
    floor = float(np.median(scores)) * dead
    return [bool(s >= floor) for s in scores]


def heading_ratio(sharpness: np.ndarray, yaw: np.ndarray, t_sec: np.ndarray,
                  span: float) -> np.ndarray:
    """Each face's sharpness over the recent norm for its own heading.

    The measure the raw score should have been. Texture is a property of what
    a heading is pointed at, so comparing a face only against other faces of
    the same heading nearby in time puts the same wall on both sides of the
    division, where it cancels. What is left is blur.

    A ratio near 1 means as sharp as this heading usually is. Well under 1
    means this particular frame is smeared, whatever the wall looks like.
    """
    out = np.ones(len(sharpness), dtype=np.float64)
    for heading in np.unique(yaw):
        rows = np.flatnonzero(yaw == heading)
        for i in rows:
            near = rows[np.abs(t_sec[rows] - t_sec[i]) <= span]
            ref = float(np.median(sharpness[near]))
            out[i] = sharpness[i] / ref if ref > 0 else 1.0
    return out
