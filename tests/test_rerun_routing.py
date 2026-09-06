"""Which stage a settings change re-runs from, as the server routes it."""

import pytest

from accio.core.params import BACKBONES, PipelineParams
from accio.server.app import Rerun, check, merge

DINOV2 = "vit_base_patch14_dinov2.lvd142m"


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


def test_budget_change_visible():
    _, _f, changed = merge(PipelineParams(), Rerun(calib={"false_merge_pct": 10}))
    assert changed == {"calib.false_merge_pct"}


def test_window_change_remeasures():
    _, _f, changed = merge(PipelineParams(), Rerun(calib={"far_seconds": 40}))
    assert changed == {"calib.far_seconds"}


def test_segmentation_reruns_alone():
    _p, first, changed = merge(PipelineParams(), Rerun(segment={"enabled": True}))
    assert first == "segment"
    assert changed == {"segment.enabled"}


def test_faces_change_reruns_downstream():
    _p, first, _c = merge(PipelineParams(),
                          Rerun(faces={"fov_deg": 120}, segment={"enabled": True}))
    assert first == "faces"


def test_unlisted_segmenter_refused():
    with pytest.raises(Exception) as e:
        check(Rerun(segment={"model_name": "some/other-model"}))
    assert e.value.status_code == 422


def test_backbone_change_reruns_from_embed():
    params, first, changed = merge(PipelineParams(),
                                   Rerun(embed={"model_name": DINOV2}))
    assert first == "embed"
    assert changed == {"embed.model_name"}
    assert params.embed.model_name == DINOV2


def test_input_size_follows_backbone():
    """384 is not a multiple of 14; timm would refuse the model outright."""
    params, _f, _c = merge(PipelineParams(), Rerun(embed={"model_name": DINOV2}))
    assert params.embed.img_size == BACKBONES[DINOV2] == 392


def test_batch_size_change_keeps_input_size():
    p = PipelineParams()
    params, first, _c = merge(p, Rerun(embed={"batch_size": 4}))
    assert first == "embed"
    assert params.embed.img_size == p.embed.img_size


def test_unsized_backbone_refused():
    with pytest.raises(Exception) as e:
        check(Rerun(embed={"model_name": "resnet50"}))
    assert e.value.status_code == 422
