"""Folding staged settings into a walk's params, and naming what that costs.

The stage a change re-runs from is the whole cost model: get it wrong and
either the frames go stale or a threshold tweak re-stitches the walk.
"""

import pytest
from dataclasses import replace
from fastapi import HTTPException

from accio.core.params import PipelineParams
from accio.server.app import Rerun, check, merge


def patch(**sections) -> Rerun:
    return Rerun(**sections)


def test_a_threshold_change_only_reselects():
    params, first = merge(PipelineParams(), patch(dedup={"tau": 0.96}))
    assert first == "select"
    assert params.dedup.tau == 0.96


def test_the_earliest_stage_wins():
    _, first = merge(PipelineParams(),
                     patch(dedup={"tau": 0.96}, faces={"fov_deg": 120},
                           gate={"window": 8}))
    assert first == "gate"


def test_setting_a_value_back_is_not_a_change():
    p = PipelineParams()
    params, first = merge(p, patch(gate={"window": p.gate.window},
                                   dedup={"tau": p.dedup.tau}))
    assert first is None
    assert params == p


def test_only_the_edited_fields_move():
    p = PipelineParams()
    params, first = merge(p, patch(faces={"fov_deg": 120}))
    assert first == "faces"
    assert params.faces == replace(p.faces, fov_deg=120)


def test_json_lists_become_the_tuples_the_params_use():
    params, first = merge(PipelineParams(),
                          patch(faces={"yaws": [0, 120, 240]},
                                gate={"band": [0.2, 0.8]}))
    assert first == "gate"
    assert params.faces.yaws == (0, 120, 240)
    assert params.gate.band == (0.2, 0.8)


def test_a_band_the_same_as_the_saved_one_is_not_a_change():
    p = PipelineParams()
    _, first = merge(p, patch(gate={"band": list(p.gate.band)}))
    assert first is None


@pytest.mark.parametrize("bad", [
    {"gate": {"band": [0.8, 0.2]}},          # bottom above top
    {"gate": {"band": [-0.1, 0.8]}},         # outside the panorama
    {"faces": {"yaws": []}},                 # no faces at all
    {"faces": {"yaws": [45, 45]}},           # the same heading twice
    {"faces": {"yaws": [45, 400]}},          # not a heading
    {"dedup": {"rule": "vibes"}},
])
def test_settings_that_would_not_survive_the_pipeline_are_refused(bad):
    with pytest.raises(HTTPException) as e:
        check(patch(**bad))
    assert e.value.status_code == 422
