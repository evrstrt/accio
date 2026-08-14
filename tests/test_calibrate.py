"""Calibration as its own stage: what it measures and what re-reads it.

Two measurements with different jobs. The far distribution places the
threshold, and the budget over it is a statistic, so moving it must not cost a
measurement. The reference is a health check on the stitch, and when it cannot
be taken the run says so rather than inventing a threshold.
"""

from dataclasses import replace

import numpy as np
import pytest

from accio.core.calibrate import (MIN_FAR, far_cosines, record, resolve_tau,
                                  sample_panos, thin)
from accio.core.params import CalibParams, DedupParams, PipelineParams
from accio.jobs.pipeline import apply_rule
from accio.server.app import Rerun, merge


def pairs(cosines):
    return [{"pano": i, "yaw": 45, "tSec": float(i), "face": f"y045_{i:05d}.jpg",
             "neighbour": f"n{i:05d}_y045.jpg", "cosine": c}
            for i, c in enumerate(cosines)]


def far(n=MIN_FAR, lo=0.5, hi=0.95):
    return np.linspace(lo, hi, n)


def test_samples_spread_across_the_whole_walk():
    # end to end, not the first n: the reference must cover the whole route
    assert sample_panos([0, 1, 2, 3, 40, 80, 90], 3) == [0, 3, 90]


def test_sampling_more_than_there_is_takes_everything():
    assert sample_panos([4, 0, 2], 10) == [0, 2, 4]


# --- what elsewhere scores -------------------------------------------------

def unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def test_far_pairs_are_only_taken_within_one_heading():
    """Two yaws of one station look unalike for reasons that are not distance,
    so mixing them in would drag the threshold down."""
    faces = [(0, 0.0, 45, "a"), (0, 0.0, 135, "b"),
             (9, 90.0, 45, "c"), (9, 90.0, 135, "d")]
    emb = np.array([unit([1, 0]), unit([0, 1]), unit([1, 0.1]), unit([0.1, 1])])
    cos = far_cosines(faces, emb, far_seconds=20.0)
    assert len(cos) == 2                      # one per heading, not six
    assert cos.min() > 0.9                    # both are same-heading pairs


def test_pairs_closer_than_the_window_are_not_elsewhere():
    faces = [(0, 0.0, 45, "a"), (1, 5.0, 45, "b")]
    emb = np.array([unit([1, 0]), unit([0, 1])])
    assert len(far_cosines(faces, emb, far_seconds=20.0)) == 0


def test_the_threshold_meets_the_budget_it_was_given():
    """1% of far pairs above tau is the whole contract."""
    cos = np.linspace(0.5, 0.99, 1000)
    tau = resolve_tau(cos, CalibParams(false_merge_pct=1.0))
    assert abs((cos >= tau).mean() - 0.01) < 0.005


def test_a_bigger_budget_cuts_harder():
    cos = np.linspace(0.5, 0.99, 1000)
    strict = resolve_tau(cos, CalibParams(false_merge_pct=1))
    loose = resolve_tau(cos, CalibParams(false_merge_pct=25))
    assert loose < strict          # a lower bar merges more


def test_a_site_whose_bays_are_distinguishable_gets_a_low_threshold():
    """Not clipped to a floor. ASHV measures p99 = 0.81, and overriding that
    with a guard rail would silently ignore the budget."""
    cos = np.concatenate([np.linspace(0.3, 0.78, 990), np.linspace(0.78, 0.82, 10)])
    tau = resolve_tau(cos, CalibParams(false_merge_pct=1.0))
    assert 0.75 < tau < 0.83
    assert abs((cos >= tau).mean() - 0.01) < 0.005     # the budget still holds


def test_too_few_far_pairs_measures_no_threshold_at_all():
    """A short walk has no far distribution. Saying so beats writing the
    default in and calling it calibrated."""
    assert resolve_tau(far(MIN_FAR - 1), CalibParams()) is None


def test_an_unmeasured_threshold_refuses_the_calibrated_rule():
    """Which is the default, so a walk too short to measure fails loudly rather
    than running on a constant that only looks like a measurement."""
    with pytest.raises(ValueError, match="too few to set a threshold"):
        apply_rule(PipelineParams(), record(pairs([0.98] * 12),
                                            far(MIN_FAR - 1), CalibParams(), 0.04))


def test_the_fixed_rule_is_unaffected_by_a_short_walk():
    """The fallback the error above names."""
    p = replace(PipelineParams(), dedup=DedupParams(rule="fixed"))
    got = apply_rule(p, record(pairs([0.98] * 12), far(MIN_FAR - 1),
                               CalibParams(), 0.04))
    assert got.dedup.tau == p.dedup.tau


def test_the_calibrated_rule_is_the_default_and_writes_the_measurement_back():
    """A cosine threshold is a property of the site and the backbone. Measured
    across the walks on hand the calibrated value spans 0.8371 to 0.9438, which
    is why no constant serves them all."""
    assert PipelineParams().dedup.rule == "calibrated"
    r = record(pairs([0.98] * 12), far(), CalibParams(), 0.04)
    got = apply_rule(PipelineParams(), r)
    assert got.dedup.tau == r["tau"] != DedupParams().tau


# --- the record ------------------------------------------------------------

def test_the_record_reports_both_measurements_and_sorts_the_worst_first():
    r = record(pairs([0.99, 0.90, 0.97, 0.95] * 3), far(), CalibParams(), 0.042)
    assert r["reference"]["n"] == 12
    assert r["reference"]["min"] == 0.90
    assert r["reference"]["median"] == 0.96
    assert [p["cosine"] for p in r["pairs"]][:3] == [0.90, 0.90, 0.90]
    assert r["far"]["n"] == MIN_FAR
    assert r["gapSeconds"] == 0.042
    assert r["falseMergePct"] == CalibParams().false_merge_pct


def test_a_low_reference_is_flagged_unhealthy():
    assert record(pairs([0.99] * 12), far(), CalibParams(), 0.04)["healthy"]
    assert not record(pairs([0.80] * 12), far(), CalibParams(), 0.04)["healthy"]


def test_a_health_check_that_could_not_run_is_not_a_bad_stitch():
    """Distinguishable, because they mean opposite things: one is a walk to
    look at, the other is a machine to look at."""
    r = record([], far(), CalibParams(), 0.04, "MediaSDK exited 125")
    assert r["healthy"] is False
    assert r["referenceError"] == "MediaSDK exited 125"
    assert r["tau"] is not None          # the threshold never depended on it


def test_re_resolving_the_budget_needs_no_new_measurement():
    """The same far pairs at a different budget give a different threshold."""
    measured = record(pairs([0.98] * 12), far(200), CalibParams(), 0.04)
    again = record(measured["pairs"], np.array(measured["far"]["cosines"]),
                   CalibParams(false_merge_pct=25), 0.04)
    assert again["tau"] < measured["tau"]
    assert again["reference"] == measured["reference"]   # nothing re-measured


def test_the_far_cosines_are_stored_as_a_percentile_sketch():
    """A long walk has hundreds of thousands of far pairs and the record only
    has to answer percentile questions about them."""
    cos = np.linspace(0.4, 0.99, 400_000)
    kept = thin(cos)
    assert len(kept) < 6000
    assert kept == sorted(kept)
    for p in (50, 95, 99):
        assert abs(np.percentile(kept, p) - np.percentile(cos, p)) < 0.001


# --- what re-runs -----------------------------------------------------------

@pytest.mark.parametrize("patch, stage", [
    ({"calib": {"false_merge_pct": 10}}, "calibrate"),
    ({"calib": {"far_seconds": 40}}, "calibrate"),
    ({"calib": {"samples": 20}}, "calibrate"),
    ({"calib": {"samples": 20}, "dedup": {"tau": 0.9}}, "calibrate"),
    ({"faces": {"fov_deg": 120}, "calib": {"false_merge_pct": 10}}, "faces"),
])
def test_calibration_settings_re_run_from_calibrate(patch, stage):
    _, first, _c = merge(PipelineParams(), Rerun(**patch))
    assert first == stage


def test_only_the_budget_moving_is_visible_to_the_caller():
    # what lets the server re-read the far pairs instead of measuring again
    _, _f, changed = merge(PipelineParams(), Rerun(calib={"false_merge_pct": 10}))
    assert changed == {"calib.false_merge_pct"}


def test_widening_the_window_is_a_real_measurement():
    """far_seconds redraws which pairs count as elsewhere, so it cannot be
    served from the stored sketch."""
    _, _f, changed = merge(PipelineParams(), Rerun(calib={"far_seconds": 40}))
    assert changed == {"calib.far_seconds"}
