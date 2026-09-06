import numpy as np
import pytest

from accio.core.dedup import drop_solo, greedy_dedup
from accio.core.embed import l2_normalise
from accio.core.params import DedupParams

P94 = DedupParams(tau=0.94)


def unit(*rows):
    return l2_normalise(np.array(rows, dtype=np.float32))


def flat(emb):
    """Equal sharpness, so the visit order falls back to capture order."""
    return np.ones(len(emb))


def dedup(emb, params=P94, sharpness=None, pano_of=None):
    return greedy_dedup(emb, params,
                        flat(emb) if sharpness is None else sharpness, pano_of)


def test_distinct_vectors_all_kept():
    emb = unit([1, 0, 0], [0, 1, 0], [0, 0, 1])
    result = dedup(emb)
    assert result.kept == [0, 1, 2]
    assert result.anchor_of == {}


def test_near_duplicate_absorbed_by_first():
    emb = unit([1, 0, 0], [1, 0.01, 0], [0, 1, 0])
    result = dedup(emb)
    assert result.kept == [0, 2]
    anchor, sim = result.anchor_of[1]
    assert anchor == 0
    assert sim > 0.99
    assert result.group_of[0] == [(1, sim)]
    assert result.group_of[2] == []


def test_tau_strict_boundary():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.94, np.sqrt(1 - 0.94**2)], dtype=np.float32)
    sim = float(a @ b)
    emb = np.stack([a, b])
    assert dedup(emb, DedupParams(tau=sim)).kept == [0]        # at tau: absorbed
    hair = DedupParams(tau=float(np.nextafter(sim, 1.0)))
    assert dedup(emb, hair).kept == [0, 1]                     # under tau: kept


def test_revisit_absorbed():
    wall_a, wall_b = [1, 0, 0], [0, 1, 0]
    emb = unit(wall_a, wall_b, wall_a)  # walk returns to wall A later
    result = dedup(emb)
    assert result.kept == [0, 1]
    assert result.anchor_of[2][0] == 0


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        greedy_dedup(np.zeros(5, dtype=np.float32), P94, np.ones(5))


def test_mismatched_sharpness_refused():
    with pytest.raises(ValueError, match="sharpness"):
        greedy_dedup(unit([1, 0], [0, 1]), P94, np.ones(1))


def test_empty_input():
    result = greedy_dedup(np.zeros((0, 8), dtype=np.float32), P94, np.ones(0))
    assert result.kept == []
    assert result.dropped == []


def test_same_panorama_headings_never_absorb():
    """Seam overlap pushes two yaws of one panorama over any threshold."""
    emb = unit([1, 0, 0], [1, 0.01, 0])
    assert dedup(emb).kept == [0]                # without the guard
    guarded = dedup(emb, pano_of=np.array([7, 7]))
    assert guarded.kept == [0, 1]
    assert guarded.anchor_of == {}


def test_guard_allows_later_station_absorbing():
    emb = unit([1, 0, 0], [1, 0.01, 0], [1, 0.02, 0])
    r = dedup(emb, pano_of=np.array([1, 1, 2]))
    assert r.kept == [0, 1]
    assert r.anchor_of[2][0] in (0, 1)


def test_revisit_earlier_station_merges():
    emb = unit([1, 0, 0], [0, 1, 0], [1, 0.01, 0])
    r = dedup(emb, pano_of=np.array([1, 1, 9]))
    assert r.kept == [0, 1]
    assert r.anchor_of[2][0] == 0


def test_mismatched_panorama_ids_refused():
    with pytest.raises(ValueError, match="panorama ids"):
        dedup(unit([1, 0], [0, 1]), pano_of=np.array([1]))


def wall(sharpness, *rows):
    """A walk where every frame is the same wall, so all of it is one group."""
    return greedy_dedup(unit(*rows), P94, np.array(sharpness))


def test_first_smear_not_anchor():
    """Anchoring in capture order lets two groups ship the same view."""
    r = wall([12.0, 88.0, 51.0], [1, 0, 0], [1, 0.02, 0], [1, 0.01, 0])
    assert r.kept == [1]
    assert set(r.anchor_of) == {0, 2}


def test_sharpest_first_frame_anchors():
    assert wall([90.0, 40.0], [1, 0, 0], [1, 0.02, 0]).kept == [0]


def test_tie_keeps_earlier_frame():
    assert wall([50.0, 50.0], [1, 0, 0], [1, 0.02, 0]).kept == [0]


def test_singleton_group_self_anchor():
    r = greedy_dedup(unit([1, 0, 0], [0, 1, 0]), P94, np.array([10.0, 20.0]))
    assert r.kept == [0, 1]
    assert r.anchor_of == {}


def test_shipped_frames_beyond_tau():
    """Shipping each group's sharpest member leaves 175 of 224 frames above tau on
    the 7th Floor walk."""
    rng = np.random.default_rng(0)
    emb = l2_normalise(rng.normal(size=(400, 24)).astype(np.float32))
    emb[200:] = l2_normalise(emb[:200] + rng.normal(0, 0.05, (200, 24)).astype(np.float32))
    sharp = rng.uniform(1, 200, 400)
    pano = np.arange(400) // 4
    r = greedy_dedup(emb, P94, sharp, pano)

    ship = np.array(sorted(r.kept))
    cos = emb[ship] @ emb[ship].T
    np.fill_diagonal(cos, -1.0)
    cos[pano[ship][:, None] == pano[ship][None, :]] = -1.0
    assert cos.max() < P94.tau

    for anchor, members in r.group_of.items():
        assert all(sharp[i] <= sharp[anchor] for i, _c in members)


def test_every_face_represented_as_sharp():
    rng = np.random.default_rng(1)
    emb = l2_normalise(rng.normal(size=(200, 16)).astype(np.float32))
    sharp = rng.uniform(1, 200, 200)
    r = greedy_dedup(emb, P94, sharp, np.arange(200) // 4)
    assert len(r.kept) + len(r.anchor_of) == 200
    for i, (anchor, _cos) in r.anchor_of.items():
        assert sharp[anchor] >= sharp[i]


def solo_case():
    """0 and 1 are one place seen twice (0 sharper), 2 and 3 are seen once."""
    emb = unit([1, 0, 0], [1, 0.05, 0], [0, 1, 0], [0, 0, 1])
    result = greedy_dedup(emb, P94, np.array([90.0, 10.0, 80.0, 5.0]))
    assert result.kept == [0, 2, 3]
    return result


def test_smear_with_sharper_twin_not_solo():
    kept = drop_solo(solo_case(), np.array([1.0, 0.1, 1.0, 1.0]), floor=0.5)
    assert 0 in kept.kept


def test_lone_smear_dropped():
    kept = drop_solo(solo_case(), np.array([1.0, 1.0, 1.0, 0.1]), floor=0.5)
    assert kept.kept == [0, 2]


def test_lone_plain_wall_kept():
    kept = drop_solo(solo_case(), np.array([1.0, 1.0, 1.0, 0.95]), floor=0.5)
    assert kept.kept == [0, 2, 3]


def test_zero_floor_drops_nothing():
    result = solo_case()
    assert drop_solo(result, np.zeros(4), floor=0.0) is result


def test_solo_drop_leaves_other_groups():
    result = solo_case()
    kept = drop_solo(result, np.array([1.0, 1.0, 1.0, 0.1]), floor=0.5)
    assert kept.anchor_of == result.anchor_of
    assert kept.group_of[0] == [(1, pytest.approx(result.anchor_of[1][1]))]
