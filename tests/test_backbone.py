"""A backbone change must carry its input size along and reach the embedder."""

from accio.core.params import BACKBONES, EmbedParams, PipelineParams
from accio.jobs.runner import Runner

DINOV2 = "vit_base_patch14_dinov2.lvd142m"


def test_backbone_size_divisible_by_patch():
    for name, size in BACKBONES.items():
        patch = int(name.split("patch")[1].split("_")[0])
        assert size % patch == 0, f"{name}: {size} is not a multiple of {patch}"


def test_input_size_follows_backbone():
    """384 is not a multiple of 14; timm would refuse the model outright."""
    assert EmbedParams(model_name=DINOV2).img_size == BACKBONES[DINOV2] == 392
    assert EmbedParams(model_name=DINOV2, img_size=384).img_size == 392
    assert EmbedParams(model_name="resnet50", img_size=224).img_size == 224


def test_one_embedder_per_backbone(tmp_path):
    r = Runner(tmp_path)
    default = PipelineParams()
    other = PipelineParams(embed=type(default.embed)(
        model_name=DINOV2, img_size=BACKBONES[DINOV2]))

    a, b = r.embedder(default), r.embedder(other)
    assert a is not b
    assert a is r.embedder(default)
    assert a.params.model_name == default.embed.model_name
    assert b.params.model_name == DINOV2
    assert b.params.img_size == 392
