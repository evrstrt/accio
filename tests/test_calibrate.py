"""Calibration as its own stage: what it measures and what re-reads it.

The percentile is a statistic over pairs already measured, so moving it must
not cost a measurement. The sample count is the measurement, so it must.
"""

import numpy as np
import pytest

from accio.core.calibrate import (CEILING, FLOOR, MIN_PAIRS, record,
                                  resolve_tau, sample_panos)
from accio.core.params import CalibParams, PipelineParams
from accio.server.app import Rerun, merge


def pairs(cosines):
    return [{"pano": i, "yaw": 45, "tSec": float(i), "face": f"y045_{i:05d}.jpg",
             "neighbour": f"n{i:05d}_y045.jpg", "cosine": c}
            for i, c in enumerate(cosines)]


def test_samples_spread_across_the_whole_walk():
    # end to end, not the first n: the reference must cover the whole route
    assert sample_panos([0, 1, 2, 3, 40, 80, 90], 3) == [0, 3, 90]


def test_sampling_more_than_there_is_takes_everything():
    assert sample_panos([4, 0, 2], 10) == [0, 2, 4]


def test_a_lower_percentile_gives_a_lower_threshold():
    cos = np.linspace(0.90, 0.99, 100)
    strict = resolve_tau(cos, CalibParams(quantile=1))
    loose = resolve_tau(cos, CalibParams(quantile=25))
    assert strict < loose


def test_the_threshold_stays_inside_its_guard_rails():
    assert resolve_tau(np.full(40, 0.2), CalibParams()) == FLOOR
    assert resolve_tau(np.full(40, 1.0), CalibParams()) == CEILING


def test_too_few_pairs_falls_back_to_the_fixed_default():
    few = np.full(MIN_PAIRS - 1, 0.90)
    assert resolve_tau(few, CalibParams()) == PipelineParams().dedup.tau


def test_the_record_reports_the_reference_and_sorts_the_worst_first():
    r = record(pairs([0.99, 0.90, 0.97, 0.95] * 3), CalibParams(quantile=10), 0.042)
    assert r["reference"]["n"] == 12
    assert r["reference"]["min"] == 0.90
    assert r["reference"]["median"] == 0.96
    assert [p["cosine"] for p in r["pairs"]][:3] == [0.90, 0.90, 0.90]
    assert r["gapSeconds"] == 0.042
    assert r["quantile"] == 10


def test_a_low_reference_is_flagged_unhealthy():
    assert record(pairs([0.99] * 12), CalibParams(), 0.04)["healthy"]
    assert not record(pairs([0.80] * 12), CalibParams(), 0.04)["healthy"]


def test_re_resolving_the_percentile_needs_no_new_measurement():
    """The same pairs at a different percentile give a different threshold."""
    measured = record(pairs(np.linspace(0.90, 0.99, 40)), CalibParams(), 0.04)
    again = record(measured["pairs"], CalibParams(quantile=25), 0.04)
    assert again["tau"] > measured["tau"]
    assert again["reference"] == measured["reference"]   # nothing re-measured


@pytest.mark.parametrize("patch, stage", [
    ({"calib": {"quantile": 10}}, "calibrate"),
    ({"calib": {"samples": 20}}, "calibrate"),
    ({"calib": {"samples": 20}, "dedup": {"tau": 0.9}}, "calibrate"),
    ({"faces": {"fov_deg": 120}, "calib": {"quantile": 10}}, "faces"),
])
def test_calibration_settings_re_run_from_calibrate(patch, stage):
    _, first, _c = merge(PipelineParams(), Rerun(**patch))
    assert first == stage


def test_only_the_percentile_moving_is_visible_to_the_caller():
    # what lets the server re-read the pairs instead of measuring again
    _, _f, changed = merge(PipelineParams(), Rerun(calib={"quantile": 10}))
    assert changed == {"calib.quantile"}
