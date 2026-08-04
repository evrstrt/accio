"""Per-walk threshold: set just above what "different" scores on this walk."""

import numpy as np

from accio.core.dedup import auto_tau, known_negative_cosines
from accio.core.params import DedupParams

YAWS = [45, 135, 225, 315]


def walk(n_panos: int, spread: float, seed: int = 0):
    """A walk where opposite faces sit `spread` radians apart on a circle, so
    the known-negative cosine is cos(spread) by construction."""
    rng = np.random.default_rng(seed)
    vecs, panos, yaws = [], [], []
    for p in range(n_panos):
        base = rng.uniform(0, 2 * np.pi)
        for k, y in enumerate(YAWS):
            a = base + (spread if y in (225, 315) else 0.0) + 0.01 * k
            vecs.append([np.cos(a), np.sin(a)])
            panos.append(p)
            yaws.append(y)
    return np.array(vecs, dtype=np.float32), panos, yaws


def test_only_opposite_faces_count_as_negatives():
    e, panos, yaws = walk(3, spread=0.5)
    neg = known_negative_cosines(e, panos, yaws)
    assert len(neg) == 6            # two opposite pairs per panorama, 3 panos


def test_threshold_follows_the_walks_own_scale():
    """Two walks, same pipeline, different visual self-similarity: the
    threshold moves with them where a fixed one could not."""
    similar, p1, y1 = walk(12, spread=0.6)    # cos 0.6 = 0.825
    distinct, p2, y2 = walk(12, spread=1.4)   # cos 1.4 = 0.170
    assert auto_tau(similar, p1, y1) > auto_tau(distinct, p2, y2)
    assert 0.60 <= auto_tau(distinct, p2, y2) <= 0.99


def test_falls_back_when_there_is_nothing_to_measure():
    e, panos, yaws = walk(1, spread=0.5)      # one pano: 2 pairs, too few
    assert auto_tau(e, panos, yaws) == DedupParams().tau
