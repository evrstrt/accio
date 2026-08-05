"""Swapping the backbone: the two ways it could quietly do the wrong thing.

A backbone change has to drag its input size along, because the size must
divide by the model's patch size. And it has to actually reach the embedder,
or a walk gets vectors from one model while recording the name of another.
"""

import pytest

from accio.core.params import BACKBONES, PipelineParams
from accio.jobs.runner import Runner
from accio.server.app import Rerun, check, merge

DINOV2 = "vit_base_patch14_dinov2.lvd142m"


def test_every_backbone_has_a_size_its_patches_divide():
    for name, size in BACKBONES.items():
        patch = int(name.split("patch")[1].split("_")[0])
        assert size % patch == 0, f"{name}: {size} is not a multiple of {patch}"


def test_swapping_the_backbone_re_runs_from_embed():
    params, first, changed = merge(PipelineParams(),
                                   Rerun(embed={"model_name": DINOV2}))
    assert first == "embed"
    assert changed == {"embed.model_name"}
    assert params.embed.model_name == DINOV2


def test_the_input_size_follows_the_model():
    """384 is not a multiple of 14; timm would refuse the model outright."""
    params, _f, _c = merge(PipelineParams(), Rerun(embed={"model_name": DINOV2}))
    assert params.embed.img_size == BACKBONES[DINOV2] == 392


def test_a_batch_size_change_leaves_the_input_size_alone():
    p = PipelineParams()
    params, first, _c = merge(p, Rerun(embed={"batch_size": 4}))
    assert first == "embed"
    assert params.embed.img_size == p.embed.img_size


def test_a_backbone_we_cannot_size_is_refused():
    with pytest.raises(Exception) as e:
        check(Rerun(embed={"model_name": "resnet50"}))
    assert e.value.status_code == 422


def test_the_runner_keeps_one_embedder_per_backbone(tmp_path):
    """Reusing a loaded embedder for a job that asked for another model would
    record a name the vectors did not come from."""
    r = Runner(tmp_path)
    default = PipelineParams()
    other = PipelineParams(embed=type(default.embed)(
        model_name=DINOV2, img_size=BACKBONES[DINOV2]))

    a, b = r.embedder(default), r.embedder(other)
    assert a is not b
    assert a is r.embedder(default)                    # cached, not rebuilt
    assert a.params.model_name == default.embed.model_name
    assert b.params.model_name == DINOV2
    assert b.params.img_size == 392
