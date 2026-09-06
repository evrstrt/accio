"""Pipeline parameters. Defaults come from measurements on real walks; change
them with evidence."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractParams:
    fps: float = 2.0
    pano_width: int = 3840    # height is width / 2
    sdk_image: str = "insta360-mediasdk:3.1.1"

    @property
    def pano_height(self) -> int:
        return self.pano_width // 2


@dataclass(frozen=True)
class FaceParams:
    fov_deg: float = 110.0    # 4 x 110 at 90 spacing = 20 deg overlap per seam
    size: int = 1024
    # offset 45 keeps the stitch seams (yaw +-90) inside the overlap. No
    # top/bottom faces: zenith is soffit edge-on, nadir is the helmet.
    yaws: tuple[int, ...] = (45, 135, 225, 315)


@dataclass(frozen=True)
class GateParams:
    """Drops panoramas that arrived broken. Quality is Select's job.

    On both reference walks this vetoes nothing (lowest panorama at 0.204 of
    the median). Raising it deletes coverage, not blur.
    """

    dead: float = 0.15        # fraction of the walk's median sharpness
    band: tuple[float, float] = (0.25, 0.75)  # latitude band scored; poles are
                                              # projection stretch + helmet


# input size has to divide by the patch size, so it travels with the model
BACKBONES: dict[str, int] = {
    "vit_base_patch16_dinov3.lvd1689m": 384,
    "vit_small_patch16_dinov3.lvd1689m": 384,
    "vit_base_patch14_dinov2.lvd142m": 392,
    "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k": 384,
}


@dataclass(frozen=True)
class EmbedParams:
    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    img_size: int = 384
    batch_size: int = 8
    device: str | None = None  # None: cuda, then mps, then cpu

    def __post_init__(self):
        if self.model_name in BACKBONES:
            object.__setattr__(self, "img_size", BACKBONES[self.model_name])


# "semantic" models label every pixel from ADE20K's fixed list (wall, floor,
# ceiling, door, window; no rebar, formwork or conduit). Mask2Former traces the
# wall/floor junction on bare RCC where SegFormer-B4 reads floor as wall, hence
# the default. "open" models take the classes as text and find instances
# rather than covering the frame.
SEGMENTERS: dict[str, str] = {
    "facebook/mask2former-swin-large-ade-semantic": "semantic",
    "shi-labs/oneformer_ade20k_swin_large": "semantic",
    "nvidia/segformer-b4-finetuned-ade-512-512": "semantic",
    "nvidia/segformer-b0-finetuned-ade-512-512": "semantic",
    "IDEA-Research/grounding-dino-base": "open",
}

SITE_CLASSES: tuple[str, ...] = (
    "rebar", "formwork", "scaffolding", "concrete column", "concrete beam",
    "concrete slab", "block wall", "electrical conduit", "pipe", "duct",
    "window opening", "door opening", "staircase", "debris", "plaster",
)


@dataclass(frozen=True)
class SegmentParams:
    """Runs on the kept frames only, after Select, and never drops a frame."""

    enabled: bool = False
    model_name: str = "facebook/mask2former-swin-large-ade-semantic"
    classes: tuple[str, ...] = SITE_CLASSES   # open-vocabulary models only
    threshold: float = 0.2                    # detection confidence, same
    device: str | None = None

    @property
    def kind(self) -> str:
        return SEGMENTERS.get(self.model_name, "semantic")


@dataclass(frozen=True)
class CalibParams:
    """Sets tau from the walk's own far-apart pairs.

    Faces at one heading more than far_seconds apart are somewhere else, so
    the fraction of those a threshold merges is its false-merge risk. tau is
    the (100 - false_merge_pct) percentile of their cosines.

    The reference (a kept face against the raw frame straight after it) is
    measured as a health check on the stitch and backbone but does not set
    tau: it sits at 0.966-0.984 while adjacent gated faces score 0.70-0.91,
    so a threshold hung off it merges almost nothing.

    The budget is generous because an over-cut walk is recoverable (every face
    stays on disk) and an under-cut one has already been labelled twice.
    """

    samples: int = 10             # panoramas the reference is measured on
    far_seconds: float = 20.0
    false_merge_pct: float = 5.0  # cuts 44-59% on real walks


@dataclass(frozen=True)
class DedupParams:
    """Calibrated by default: the fixed 0.94 shipped 1637 of 2632 faces on one
    walk with 98% of them holding a twin above its own threshold. Calibrated
    values seen so far run 0.837 to 0.944."""

    tau: float = 0.94         # used by the fixed rule; calibration overwrites it
    rule: str = "calibrated"  # or "fixed", the fallback for a walk too short
                              # to have a far distribution

    # a group of one has no sharper twin, so it is the one route by which a
    # smear reaches a labeller. 7th Floor: 17 of 224 exports are alone, and
    # this floor drops 2 of them.
    solo_floor: float = 0.50  # of its heading's norm; 0 keeps every singleton
    solo_span: float = 45.0   # seconds either side the norm is taken over


@dataclass(frozen=True)
class PipelineParams:
    extract: ExtractParams = field(default_factory=ExtractParams)
    faces: FaceParams = field(default_factory=FaceParams)
    gate: GateParams = field(default_factory=GateParams)
    embed: EmbedParams = field(default_factory=EmbedParams)
    calib: CalibParams = field(default_factory=CalibParams)
    dedup: DedupParams = field(default_factory=DedupParams)
    segment: SegmentParams = field(default_factory=SegmentParams)


def from_dict(d: dict) -> PipelineParams:
    """Rebuild params saved next to a walk; JSON turned the tuples into lists."""
    faces = dict(d.get("faces", {}))
    if "yaws" in faces:
        faces["yaws"] = tuple(faces["yaws"])
    gate = dict(d.get("gate", {}))
    if "band" in gate:
        gate["band"] = tuple(gate["band"])
    segment = dict(d.get("segment", {}))
    if "classes" in segment:
        segment["classes"] = tuple(segment["classes"])
    return PipelineParams(
        extract=ExtractParams(**d.get("extract", {})),
        faces=FaceParams(**faces),
        gate=GateParams(**gate),
        embed=EmbedParams(**d.get("embed", {})),
        calib=CalibParams(**d.get("calib", {})),
        dedup=DedupParams(**d.get("dedup", {})),
        segment=SegmentParams(**segment),
    )
