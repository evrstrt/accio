"""Calibration: the far distribution sets the threshold, the reference is a
health check on the stitch.
"""

import subprocess
from dataclasses import replace

import cv2
import numpy as np
import pytest

from accio.core import calibrate as calibrate_mod
from accio.core.calibrate import (CALIB_DIR, MIN_FAR, calibrate, far_cosines,
                                  record, resolve_tau, sample_panos, thin)
from accio.core.embed import l2_normalise
from accio.core.extract import JPEG_EOI
from accio.core.params import (CalibParams, DedupParams, FaceParams,
                               PipelineParams)
from accio.jobs.pipeline import apply_rule


def pairs(cosines):
    return [{"pano": i, "yaw": 45, "tSec": float(i), "face": f"y045_{i:05d}.jpg",
             "neighbour": f"n{i:05d}_y045.jpg", "cosine": c}
            for i, c in enumerate(cosines)]


def far(n=MIN_FAR, lo=0.5, hi=0.95):
    return np.linspace(lo, hi, n)


def test_samples_span_walk():
    assert sample_panos([0, 1, 2, 3, 40, 80, 90], 3) == [0, 3, 90]


def test_oversample_takes_all():
    assert sample_panos([4, 0, 2], 10) == [0, 2, 4]


def test_sample_zero_empty():
    assert sample_panos([4, 0, 2], 0) == []
    assert sample_panos([4, 0, 2], 1) == [0]


def unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def test_far_pairs_within_heading():
    """Two yaws of one station differ for reasons other than distance."""
    faces = [(0, 0.0, 45, "a"), (0, 0.0, 135, "b"),
             (9, 90.0, 45, "c"), (9, 90.0, 135, "d")]
    emb = np.array([unit([1, 0]), unit([0, 1]), unit([1, 0.1]), unit([0.1, 1])])
    cos = far_cosines(faces, emb, far_seconds=20.0)
    assert len(cos) == 2
    assert cos.min() > 0.9


def test_near_pairs_excluded():
    faces = [(0, 0.0, 45, "a"), (1, 5.0, 45, "b")]
    emb = np.array([unit([1, 0]), unit([0, 1])])
    assert len(far_cosines(faces, emb, far_seconds=20.0)) == 0


def test_threshold_meets_budget():
    cos = np.linspace(0.5, 0.99, 1000)
    tau = resolve_tau(cos, CalibParams(false_merge_pct=1.0))
    assert abs((cos >= tau).mean() - 0.01) < 0.005


def test_bigger_budget_lower_threshold():
    cos = np.linspace(0.5, 0.99, 1000)
    strict = resolve_tau(cos, CalibParams(false_merge_pct=1))
    loose = resolve_tau(cos, CalibParams(false_merge_pct=25))
    assert loose < strict


def test_distinguishable_site_low_threshold():
    """ASHV measures p99 = 0.81; a floor would override the budget."""
    cos = np.concatenate([np.linspace(0.3, 0.78, 990), np.linspace(0.78, 0.82, 10)])
    tau = resolve_tau(cos, CalibParams(false_merge_pct=1.0))
    assert 0.75 < tau < 0.83
    assert abs((cos >= tau).mean() - 0.01) < 0.005


def test_too_few_far_pairs_no_threshold():
    assert resolve_tau(far(MIN_FAR - 1), CalibParams()) is None


def test_unmeasured_threshold_refuses_calibrated():
    with pytest.raises(ValueError, match="too few to set a threshold"):
        apply_rule(PipelineParams(), record(pairs([0.98] * 12),
                                            far(MIN_FAR - 1), CalibParams(), 0.04))


def test_fixed_rule_short_walk():
    p = replace(PipelineParams(), dedup=DedupParams(rule="fixed"))
    got = apply_rule(p, record(pairs([0.98] * 12), far(MIN_FAR - 1),
                               CalibParams(), 0.04))
    assert got.dedup.tau == p.dedup.tau


def test_calibrated_default_writes_record():
    """The calibrated value spans 0.8371 to 0.9438 across the walks on hand."""
    assert PipelineParams().dedup.rule == "calibrated"
    r = record(pairs([0.98] * 12), far(), CalibParams(), 0.04)
    got = apply_rule(PipelineParams(), r)
    assert got.dedup.tau == r["tau"] != DedupParams().tau


def test_record_sorts_worst_first():
    r = record(pairs([0.99, 0.90, 0.97, 0.95] * 3), far(), CalibParams(), 0.042)
    assert r["reference"]["n"] == 12
    assert r["reference"]["min"] == 0.90
    assert r["reference"]["median"] == 0.96
    assert [p["cosine"] for p in r["pairs"]][:3] == [0.90, 0.90, 0.90]
    assert r["far"]["n"] == MIN_FAR
    assert r["gapSeconds"] == 0.042
    assert r["falseMergePct"] == CalibParams().false_merge_pct


def test_low_reference_unhealthy():
    assert record(pairs([0.99] * 12), far(), CalibParams(), 0.04)["healthy"]
    assert not record(pairs([0.80] * 12), far(), CalibParams(), 0.04)["healthy"]


def test_unrun_health_check_not_unhealthy():
    r = record([], far(), CalibParams(), 0.04, "MediaSDK exited 125")
    assert r["healthy"] is False
    assert r["referenceError"] == "MediaSDK exited 125"
    assert r["tau"] is not None


def test_budget_reresolve_reuses_sketch():
    measured = record(pairs([0.98] * 12), far(200), CalibParams(), 0.04)
    again = record(measured["pairs"], np.array(measured["far"]["cosines"]),
                   CalibParams(false_merge_pct=25), 0.04)
    assert again["tau"] < measured["tau"]
    assert again["reference"] == measured["reference"]


def test_far_cosines_percentile_sketch():
    cos = np.linspace(0.4, 0.99, 400_000)
    kept = thin(cos)
    assert len(kept) < 6000
    assert kept == sorted(kept)
    for p in (50, 95, 99):
        assert abs(np.percentile(kept, p) - np.percentile(cos, p)) < 0.001


class MeanEmbedder:
    def embed(self, images):
        return l2_normalise(np.array([[img.mean(), 1.0] for img in images],
                                     dtype=np.float32))


def jpeg(whole=True) -> bytes:
    ok, arr = cv2.imencode(".jpg", np.full((32, 64, 3), 128, np.uint8))
    assert ok
    data = arr.tobytes()
    return data if whole else data[:-2]


def fake_sdk(monkeypatch, calib_dir, frames: dict[int, bytes]) -> list[list[str]]:
    """Records every subprocess call; `docker run` drops `frames` in calib_dir."""
    calls = []

    def run(cmd, **_kw):
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            for no, data in frames.items():
                (calib_dir / f"{no}.jpg").write_bytes(data)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(calibrate_mod.subprocess, "run", run)
    return calls


def one_pano(tmp_path, monkeypatch, frames):
    """One face at t=0.5 s of a 30 fps walk; its reference frame is 16."""
    out = tmp_path / "walk"
    faces = [(0, 0.5, 45, "y045_00000.jpg")]
    emb = np.array([[0.6, 0.8]], dtype=np.float32)
    params = PipelineParams(faces=FaceParams(size=32, yaws=(45,)))
    calls = fake_sdk(monkeypatch, out / CALIB_DIR, frames)
    rec = calibrate(tmp_path / "walk.insv", out, faces, emb, params,
                    MeanEmbedder(), native_fps=30.0)
    return rec, calls, out / CALIB_DIR


def test_reference_frame_measured_then_discarded(tmp_path, monkeypatch):
    rec, _calls, calib_dir = one_pano(tmp_path, monkeypatch, {16: jpeg()})
    assert rec["referenceError"] == ""
    assert rec["reference"]["n"] == 1
    assert rec["pairs"][0]["neighbour"] == "n00000_y045.jpg"
    assert (calib_dir / "n00000_y045.jpg").exists()
    assert not (calib_dir / "16.jpg").exists()


def test_container_name_freed_before_reference(tmp_path, monkeypatch):
    _rec, calls, _d = one_pano(tmp_path, monkeypatch, {16: jpeg()})
    assert calls[0] == ["docker", "rm", "-f", "calib-walk"]
    assert calls[1][:2] == ["docker", "run"]


def test_torn_reference_frame_skipped(tmp_path, monkeypatch):
    """A killed container leaves truncated JPEGs; cv2.imread answers None."""
    rec, _calls, _d = one_pano(tmp_path, monkeypatch, {16: jpeg(whole=False)})
    assert rec["reference"]["n"] == 0
    assert rec["referenceError"].startswith("MediaSDK exported none of 1")


def test_undecodable_reference_frame_skipped(tmp_path, monkeypatch):
    junk = b"\xff\xd8" + b"\x00" * 400 + JPEG_EOI
    rec, _calls, calib_dir = one_pano(tmp_path, monkeypatch, {16: junk})
    assert rec["reference"]["n"] == 0
    assert not (calib_dir / "16.jpg").exists()
