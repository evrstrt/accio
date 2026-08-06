import numpy as np
import pytest

from accio.core.dedup import drop_solo, greedy_dedup, sharpest
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


# --- a station's own headings are coverage, not duplicates -----------------

def test_two_headings_of_one_station_never_absorb_each_other():
    """A symmetric room, or the 20 degrees every seam overlaps by design,
    pushes two yaws of one panorama over any threshold. Merging them deletes a
    heading that nothing will shoot again."""
    emb = unit([1, 0, 0], [1, 0.01, 0])          # cosine well above tau
    assert greedy_dedup(emb, P94).kept == [0]    # without the guard: merged
    guarded = greedy_dedup(emb, P94, np.array([7, 7]))
    assert guarded.kept == [0, 1]
    assert guarded.anchor_of == {}


def test_the_guard_does_not_stop_a_later_station_absorbing():
    """Only the same panorama is exempt; the next station along is exactly
    what dedup is for."""
    emb = unit([1, 0, 0], [1, 0.01, 0], [1, 0.02, 0])
    r = greedy_dedup(emb, P94, np.array([1, 1, 2]))
    assert r.kept == [0, 1]
    assert r.anchor_of[2][0] in (0, 1)


def test_a_revisit_to_an_earlier_station_still_merges():
    emb = unit([1, 0, 0], [0, 1, 0], [1, 0.01, 0])
    r = greedy_dedup(emb, P94, np.array([1, 1, 9]))
    assert r.kept == [0, 1]
    assert r.anchor_of[2][0] == 0


def test_mismatched_panorama_ids_are_refused():
    with pytest.raises(ValueError, match="panorama ids"):
        greedy_dedup(unit([1, 0], [0, 1]), P94, np.array([1]))


# --- which member of a group gets exported ---------------------------------

def wall(*rows):
    """A walk where every frame is the same wall, so all of it is one group."""
    return greedy_dedup(unit(*rows), P94)


def test_the_group_exports_its_sharpest_not_its_first():
    """The ordinary case: the operator enters a bay mid-turn and the smeared
    frame arrives first, then they settle and take several clean ones."""
    r = wall([1, 0, 0], [1, 0.02, 0], [1, 0.01, 0])
    assert r.kept == [0]                       # one group, anchored on the smear
    assert sharpest(r, np.array([12.0, 88.0, 51.0])) == {0: 1}


def test_the_anchor_wins_when_it_is_the_sharpest():
    r = wall([1, 0, 0], [1, 0.02, 0])
    assert sharpest(r, np.array([90.0, 40.0])) == {0: 0}


def test_a_tie_stays_on_the_anchor():
    """Nothing to gain from shuffling, and the anchor is what review shows."""
    r = wall([1, 0, 0], [1, 0.02, 0])
    assert sharpest(r, np.array([50.0, 50.0])) == {0: 0}


def test_a_group_of_one_is_its_own_pick():
    r = greedy_dedup(unit([1, 0, 0], [0, 1, 0]), P94)
    assert sharpest(r, np.array([10.0, 20.0])) == {0: 0, 1: 1}


def test_every_group_gets_exactly_one_pick_from_its_own_members():
    r = wall([1, 0, 0], [1, 0.02, 0], [1, 0.01, 0])
    r2 = greedy_dedup(unit([1, 0, 0], [1, 0.02, 0], [0, 1, 0], [0, 1, 0.02]), P94)
    for result in (r, r2):
        picks = sharpest(result, np.arange(10.0, 10.0 + len(result.anchor_of)
                                           + len(result.kept)))
        assert set(picks) == set(result.kept)
        for anchor, pick in picks.items():
            members = {anchor} | {i for i, _c in result.group_of[anchor]}
            assert pick in members


# --- groups of one: the only route a smear has to a labeller ---------------

def solo_case():
    """Four faces. 0 and 1 are one place seen twice, 2 and 3 are seen once
    each; 2 is sharp and 3 is a smear."""
    emb = unit([1, 0, 0], [1, 0.05, 0], [0, 1, 0], [0, 0, 1])
    result = greedy_dedup(emb, P94)
    assert result.kept == [0, 2, 3]              # 1 absorbed into 0
    return result


def test_a_smear_with_a_sharper_twin_is_never_the_solo_rule_s_problem():
    """It loses the exemplar contest instead, which needs no threshold."""
    result = solo_case()
    sharp = np.array([10.0, 90.0, 80.0, 5.0])
    picks = sharpest(result, sharp)
    assert picks[0] == 1                         # the group exports its best
    ratio = np.array([1.0, 1.0, 1.0, 0.1])
    kept = drop_solo(result, ratio, picks, floor=0.5)
    assert 0 in kept.kept                        # the group survives regardless


def test_a_smear_that_is_alone_goes():
    result = solo_case()
    picks = sharpest(result, np.array([10.0, 90.0, 80.0, 5.0]))
    kept = drop_solo(result, ratio=np.array([1.0, 1.0, 1.0, 0.1]),
                     picks=picks, floor=0.5)
    assert kept.kept == [0, 2]                   # 3 was alone and smeared


def test_a_plain_wall_seen_once_stays():
    """Its ratio is against its own heading, so being plain is not being blurred."""
    result = solo_case()
    picks = sharpest(result, np.array([10.0, 90.0, 80.0, 5.0]))
    kept = drop_solo(result, ratio=np.array([1.0, 1.0, 1.0, 0.95]),
                     picks=picks, floor=0.5)
    assert kept.kept == [0, 2, 3]


def test_a_floor_of_zero_drops_nothing():
    result = solo_case()
    picks = sharpest(result, np.array([10.0, 90.0, 80.0, 5.0]))
    assert drop_solo(result, np.zeros(4), picks, floor=0.0) is result


def test_dropping_a_solo_leaves_the_other_groups_intact():
    """Its members must still point at their anchors afterwards."""
    result = solo_case()
    picks = sharpest(result, np.array([10.0, 90.0, 80.0, 5.0]))
    kept = drop_solo(result, np.array([1.0, 1.0, 1.0, 0.1]), picks, floor=0.5)
    assert kept.anchor_of == result.anchor_of
    assert kept.group_of[0] == [(1, pytest.approx(result.anchor_of[1][1]))]
