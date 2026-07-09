import numpy as np
import pytest

from accio.core.dedup import greedy_dedup
from accio.core.embed import l2_normalise
from accio.core.params import DedupParams

P94 = DedupParams(tau=0.94)


def unit(*rows):
    return l2_normalise(np.array(rows, dtype=np.float32))


def test_distinct_vectors_all_kept():
    emb = unit([1, 0, 0], [0, 1, 0], [0, 0, 1])
    result = greedy_dedup(emb, P94)
    assert result.kept == [0, 1, 2]
    assert result.anchor_of == {}


def test_near_duplicate_absorbed_by_first_seen():
    emb = unit([1, 0, 0], [1, 0.01, 0], [0, 1, 0])
    result = greedy_dedup(emb, P94)
    assert result.kept == [0, 2]
    anchor, sim = result.anchor_of[1]
    assert anchor == 0
    assert sim > 0.99
    assert result.group_of[0] == [(1, sim)]
    assert result.group_of[2] == []


def test_tau_is_a_strict_keep_boundary():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.94, np.sqrt(1 - 0.94**2)], dtype=np.float32)
    sim = float(a @ b)
    at_tau = greedy_dedup(np.stack([a, b]), DedupParams(tau=sim))
    assert at_tau.kept == [0]  # exactly at tau: absorbed
    below_tau = greedy_dedup(np.stack([a, b]), DedupParams(tau=float(np.nextafter(sim, 1.0))))
    assert below_tau.kept == [0, 1]  # a hair under tau: kept


def test_revisit_is_caught_not_just_neighbours():
    wall_a, wall_b = [1, 0, 0], [0, 1, 0]
    emb = unit(wall_a, wall_b, wall_a)  # walk returns to wall A later
    result = greedy_dedup(emb, P94)
    assert result.kept == [0, 1]
    assert result.anchor_of[2][0] == 0


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        greedy_dedup(np.zeros(5, dtype=np.float32), P94)


def test_empty_input():
    result = greedy_dedup(np.zeros((0, 8), dtype=np.float32), P94)
    assert result.kept == []
    assert result.dropped == []
