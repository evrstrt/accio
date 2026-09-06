"""The run history, kept so two configurations can be compared."""

import json

from accio.core.params import CalibParams, DedupParams, EmbedParams, PipelineParams
from accio.jobs.pipeline import RUNS_FILE, RUNS_KEPT, log_run, read_runs

CALIB = {"reference": {"median": 0.9815, "n": 40}}


def dinov2() -> PipelineParams:
    return PipelineParams(
        embed=EmbedParams(model_name="vit_base_patch14_dinov2.lvd142m",
                          img_size=392),
        dedup=DedupParams(tau=0.9577, rule="calibrated"),
        calib=CalibParams())


def test_run_records_params_and_result(tmp_path):
    log_run(tmp_path, PipelineParams(), CALIB, "gate", 148, 103)
    (r,) = read_runs(tmp_path)
    assert r["backbone"] == PipelineParams().embed.model_name
    assert (r["tau"], r["rule"]) == (0.94, "calibrated")
    assert (r["reference"], r["pairs"]) == (0.9815, 40)
    assert (r["faces"], r["anchors"], r["absorbed"]) == (148, 103, 45)
    assert r["from"] == "gate"
    assert r["at"]


def test_runs_accumulate_newest_first(tmp_path):
    log_run(tmp_path, PipelineParams(), CALIB, "gate", 148, 103)
    log_run(tmp_path, dinov2(), CALIB, "embed", 148, 107)
    runs = read_runs(tmp_path)
    assert [r["anchors"] for r in runs] == [107, 103]
    assert runs[0]["backbone"].endswith("dinov2.lvd142m")


def test_history_capped_keeps_recent(tmp_path):
    for i in range(30):
        log_run(tmp_path, PipelineParams(), CALIB, "select", 148, i)
    runs = read_runs(tmp_path, limit=20)
    assert len(runs) == 20
    assert [r["anchors"] for r in runs[:3]] == [29, 28, 27]


def test_file_capped(tmp_path):
    """Capping only the read lets a walk re-run for months grow the file forever."""
    for i in range(RUNS_KEPT + 7):
        log_run(tmp_path, PipelineParams(), CALIB, "select", 148, i)
    lines = (tmp_path / RUNS_FILE).read_text().splitlines()
    assert len(lines) == RUNS_KEPT
    assert json.loads(lines[-1])["anchors"] == RUNS_KEPT + 6
    assert json.loads(lines[0])["anchors"] == 7


def test_torn_line_skipped(tmp_path):
    log_run(tmp_path, PipelineParams(), CALIB, "select", 148, 1)
    with open(tmp_path / RUNS_FILE, "a") as f:
        f.write('{"anchors": 2, "tau"')
    (r,) = read_runs(tmp_path)
    assert r["anchors"] == 1
    log_run(tmp_path, PipelineParams(), CALIB, "select", 148, 3)
    assert [r["anchors"] for r in read_runs(tmp_path)] == [3, 1]


def test_run_without_calibration(tmp_path):
    log_run(tmp_path, PipelineParams(), None, "select", 148, 79)
    (r,) = read_runs(tmp_path)
    assert r["reference"] == 0 and r["pairs"] == 0
    assert r["anchors"] == 79


def test_no_history_empty(tmp_path):
    assert read_runs(tmp_path) == []


def test_run_appended_after_earlier(tmp_path):
    log_run(tmp_path, PipelineParams(), CALIB, "gate", 148, 103)
    first = (tmp_path / RUNS_FILE).read_text()
    log_run(tmp_path, dinov2(), CALIB, "embed", 148, 107)
    after = (tmp_path / RUNS_FILE).read_text()
    assert after.startswith(first)
    assert len(after.splitlines()) == 2
    assert all(json.loads(line) for line in after.splitlines())
