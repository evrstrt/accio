"""The run history: what each configuration gave, kept so two can be compared.

A walk on disk only ever shows the settings that ran last. Without this,
comparing two backbones means writing the numbers down by hand.
"""

import json

from accio.core.params import CalibParams, DedupParams, EmbedParams, PipelineParams
from accio.jobs.pipeline import RUNS_FILE, log_run, read_runs

CALIB = {"reference": {"median": 0.9815, "n": 40}}


def dinov2() -> PipelineParams:
    return PipelineParams(
        embed=EmbedParams(model_name="vit_base_patch14_dinov2.lvd142m",
                          img_size=392),
        dedup=DedupParams(tau=0.9577, rule="calibrated"),
        calib=CalibParams())


def test_a_run_records_what_it_took_and_what_it_gave(tmp_path):
    log_run(tmp_path, PipelineParams(), CALIB, "gate", 148, 103)
    (r,) = read_runs(tmp_path)
    assert r["backbone"] == PipelineParams().embed.model_name
    assert (r["tau"], r["rule"]) == (0.94, "fixed")
    assert (r["reference"], r["pairs"]) == (0.9815, 40)
    assert (r["faces"], r["anchors"], r["absorbed"]) == (148, 103, 45)
    assert r["from"] == "gate"
    assert r["at"]                                   # stamped, not asserted on


def test_runs_accumulate_newest_first(tmp_path):
    log_run(tmp_path, PipelineParams(), CALIB, "gate", 148, 103)
    log_run(tmp_path, dinov2(), CALIB, "embed", 148, 107)
    runs = read_runs(tmp_path)
    assert [r["anchors"] for r in runs] == [107, 103]
    assert runs[0]["backbone"].endswith("dinov2.lvd142m")


def test_the_history_is_capped_but_keeps_the_recent_end(tmp_path):
    for i in range(30):
        log_run(tmp_path, PipelineParams(), CALIB, "select", 148, i)
    runs = read_runs(tmp_path, limit=20)
    assert len(runs) == 20
    assert [r["anchors"] for r in runs[:3]] == [29, 28, 27]


def test_a_run_without_a_calibration_still_records(tmp_path):
    """A fixed threshold does not need the reference, but the row is the same
    shape either way or the history cannot be read as a table."""
    log_run(tmp_path, PipelineParams(), None, "select", 148, 79)
    (r,) = read_runs(tmp_path)
    assert r["reference"] == 0 and r["pairs"] == 0
    assert r["anchors"] == 79


def test_a_walk_with_no_history_reads_as_empty(tmp_path):
    assert read_runs(tmp_path) == []


def test_the_file_is_append_only(tmp_path):
    log_run(tmp_path, PipelineParams(), CALIB, "gate", 148, 103)
    first = (tmp_path / RUNS_FILE).read_text()
    log_run(tmp_path, dinov2(), CALIB, "embed", 148, 107)
    after = (tmp_path / RUNS_FILE).read_text()
    assert after.startswith(first)
    assert len(after.splitlines()) == 2
    assert all(json.loads(line) for line in after.splitlines())
