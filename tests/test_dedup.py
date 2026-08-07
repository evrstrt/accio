import numpy as np
import pytest

from accio.core.dedup import drop_solo, greedy_dedup
from accio.core.embed import l2_normalise
from accio.core.params import DedupParams

P94 = DedupParams(tau=0.94)


def unit(*rows):
    return l2_normalise(np.array(rows, dtype=np.float32))


def flat(emb):
    """Equal sharpness, so the visit order falls back to capture order and a
    test can be about grouping alone."""
    return np.ones(len(emb))


def dedup(emb, params=P94, sharpness=None, pano_of=None):
    return greedy_dedup(emb, params,
                        flat(emb) if sharpness is None else sharpness, pano_of)


def test_distinct_vectors_all_kept():
    emb = unit([1, 0, 0], [0, 1, 0], [0, 0, 1])
    result = dedup(emb)
    assert result.kept == [0, 1, 2]
    assert result.anchor_of == {}


def test_near_duplicate_absorbed_by_the_first_of_equals():
    emb = unit([1, 0, 0], [1, 0.01, 0], [0, 1, 0])
    result = dedup(emb)
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
    emb = np.stack([a, b])
    assert dedup(emb, DedupParams(tau=sim)).kept == [0]        # at tau: absorbed
    hair = DedupParams(tau=float(np.nextafter(sim, 1.0)))
    assert dedup(emb, hair).kept == [0, 1]                     # under tau: kept


def test_revisit_is_caught_not_just_neighbours():
    wall_a, wall_b = [1, 0, 0], [0, 1, 0]
    emb = unit(wall_a, wall_b, wall_a)  # walk returns to wall A later
    result = dedup(emb)
    assert result.kept == [0, 1]
    assert result.anchor_of[2][0] == 0


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        greedy_dedup(np.zeros(5, dtype=np.float32), P94, np.ones(5))


def test_mismatched_sharpness_is_refused():
    with pytest.raises(ValueError, match="sharpness"):
        greedy_dedup(unit([1, 0], [0, 1]), P94, np.ones(1))


def test_empty_input():
    result = greedy_dedup(np.zeros((0, 8), dtype=np.float32), P94, np.ones(0))
    assert result.kept == []
    assert result.dropped == []


# --- a station's own headings are coverage, not duplicates -----------------

def test_two_headings_of_one_station_never_absorb_each_other():
    """A symmetric room, or the 20 degrees every seam overlaps by design,
    pushes two yaws of one panorama over any threshold. Merging them deletes a
    heading that nothing will shoot again."""
    emb = unit([1, 0, 0], [1, 0.01, 0])          # cosine well above tau
    assert dedup(emb).kept == [0]                # without the guard: merged
    guarded = dedup(emb, pano_of=np.array([7, 7]))
    assert guarded.kept == [0, 1]
    assert guarded.anchor_of == {}


def test_the_guard_does_not_stop_a_later_station_absorbing():
    """Only the same panorama is exempt; the next station along is exactly
    what dedup is for."""
    emb = unit([1, 0, 0], [1, 0.01, 0], [1, 0.02, 0])
    r = dedup(emb, pano_of=np.array([1, 1, 2]))
    assert r.kept == [0, 1]
    assert r.anchor_of[2][0] in (0, 1)


def test_a_revisit_to_an_earlier_station_still_merges():
    emb = unit([1, 0, 0], [0, 1, 0], [1, 0.01, 0])
    r = dedup(emb, pano_of=np.array([1, 1, 9]))
    assert r.kept == [0, 1]
    assert r.anchor_of[2][0] == 0


def test_mismatched_panorama_ids_are_refused():
    with pytest.raises(ValueError, match="panorama ids"):
        dedup(unit([1, 0], [0, 1]), pano_of=np.array([1]))


# --- the ordering is the design: the anchor is the frame that ships --------

def wall(sharpness, *rows):
    """A walk where every frame is the same wall, so all of it is one group."""
    return greedy_dedup(unit(*rows), P94, np.array(sharpness))


def test_the_smear_that_arrives_first_does_not_become_the_anchor():
    """The ordinary case: the operator enters a bay mid-turn and the smeared
    frame arrives first, then they settle and take several clean ones. In
    capture order that smear anchored the group and had to be swapped out
    afterwards, which is what let two groups ship the same view."""
    r = wall([12.0, 88.0, 51.0], [1, 0, 0], [1, 0.02, 0], [1, 0.01, 0])
    assert r.kept == [1]                         # the sharpest, not the first
    assert set(r.anchor_of) == {0, 2}


def test_the_first_frame_anchors_when_it_is_the_sharpest():
    assert wall([90.0, 40.0], [1, 0, 0], [1, 0.02, 0]).kept == [0]


def test_a_tie_keeps_the_earlier_frame():
    """Nothing to gain from shuffling, and the anchor is what review shows."""
    assert wall([50.0, 50.0], [1, 0, 0], [1, 0.02, 0]).kept == [0]


def test_a_group_of_one_is_its_own_anchor():
    r = greedy_dedup(unit([1, 0, 0], [0, 1, 0]), P94, np.array([10.0, 20.0]))
    assert r.kept == [0, 1]
    assert r.anchor_of == {}


def test_no_two_shipped_frames_are_within_tau():
    """The guarantee the old two-step broke. It compared anchors and shipped
    their sharpest members, and nothing compared those to each other: on the
    7th Floor walk 175 of 224 shipped frames sat above tau against another.
    """
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


def test_every_face_is_represented_by_something_at_least_as_sharp():
    """Dropping a face has to leave a frame that stands for it, and there is no
    point standing for it with a blurrier one."""
    rng = np.random.default_rng(1)
    emb = l2_normalise(rng.normal(size=(200, 16)).astype(np.float32))
    sharp = rng.uniform(1, 200, 200)
    r = greedy_dedup(emb, P94, sharp, np.arange(200) // 4)
    assert len(r.kept) + len(r.anchor_of) == 200
    for i, (anchor, _cos) in r.anchor_of.items():
        assert sharp[anchor] >= sharp[i]


# --- groups of one: the only route a smear has to a labeller ---------------

def solo_case():
    """Four faces. 0 and 1 are one place seen twice, 2 and 3 are seen once
    each. 0 is the sharper of the pair, so it anchors."""
    emb = unit([1, 0, 0], [1, 0.05, 0], [0, 1, 0], [0, 0, 1])
    result = greedy_dedup(emb, P94, np.array([90.0, 10.0, 80.0, 5.0]))
    assert result.kept == [0, 2, 3]              # 1 absorbed into 0
    return result


def test_a_smear_with_a_sharper_twin_is_never_the_solo_rule_s_problem():
    """It lost the anchor contest instead, which needs no threshold."""
    kept = drop_solo(solo_case(), np.array([1.0, 0.1, 1.0, 1.0]), floor=0.5)
    assert 0 in kept.kept          # the group survives, represented by its best


def test_a_smear_that_is_alone_goes():
    kept = drop_solo(solo_case(), np.array([1.0, 1.0, 1.0, 0.1]), floor=0.5)
    assert kept.kept == [0, 2]                   # 3 was alone and smeared


def test_a_plain_wall_seen_once_stays():
    """Its ratio is against its own heading, so being plain is not being blurred."""
    kept = drop_solo(solo_case(), np.array([1.0, 1.0, 1.0, 0.95]), floor=0.5)
    assert kept.kept == [0, 2, 3]


def test_a_floor_of_zero_drops_nothing():
    result = solo_case()
    assert drop_solo(result, np.zeros(4), floor=0.0) is result


def test_dropping_a_solo_leaves_the_other_groups_intact():
    """Its members must still point at their anchors afterwards."""
    result = solo_case()
    kept = drop_solo(result, np.array([1.0, 1.0, 1.0, 0.1]), floor=0.5)
    assert kept.anchor_of == result.anchor_of
    assert kept.group_of[0] == [(1, pytest.approx(result.anchor_of[1][1]))]
