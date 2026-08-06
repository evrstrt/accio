"""Pipeline parameters with the defaults validated in the dedup PoC (June 2026).

Every stage takes its params object explicitly; nothing reads globals. Change a
default here only with evidence (the PoC's borderline sheets, or an override
pattern from review).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractParams:
    """Dual-fisheye .insv -> decimated equirect panoramas via the Insta360
    MediaSDK container (optflow stitch + flowstate horizon levelling; replaces
    ffmpeg v360, which left parallax seams and tilted horizons)."""

    fps: float = 2.0          # walking pace + full sphere per frame
    pano_width: int = 3840    # native resolution; height is width / 2
    sdk_image: str = "insta360-mediasdk:3.1.1"

    @property
    def pano_height(self) -> int:
        return self.pano_width // 2


@dataclass(frozen=True)
class FaceParams:
    """Equirect panorama -> gnomonic side faces."""

    fov_deg: float = 110.0    # 4 x 110 at 90 spacing = 20 deg overlap per seam
    size: int = 1024
    yaws: tuple[int, ...] = (45, 135, 225, 315)
    # yaw offset 45 keeps the stitch seams (yaw +-90) inside the overlapped
    # face margins; no top/bottom faces: zenith is slab soffit seen edge-on,
    # nadir is the helmet


@dataclass(frozen=True)
class GateParams:
    """Motion-blur veto on panoramas.

    It does not thin the walk. Select does that, against a false-merge budget
    somebody chose and can restate. This removes only what no labeller could
    use, and a frame it passes still has to earn its place downstream.
    """

    window: int = 9           # frames the local reference max spans, centred:
                              # at 2 fps, the two seconds either side
    floor: float = 0.40       # below this fraction of the local max a frame is
                              # smeared rather than merely flat. Deliberately
                              # under 1/2.14, the gap between the sharpest and
                              # dullest frame of a typical stretch, because
                              # Select already picks the sharpest of each group:
                              # this only has to catch unusable, not rank.
    band: tuple[float, float] = (0.25, 0.75)  # central latitude band; poles are
                              # projection stretch + helmet


# The backbones a walk can be embedded with. Input size is not free: it has to
# divide by the model's patch size, so each one carries its own. Every entry
# is a frozen ViT read at its CLS token, which keeps the cosines comparable in
# kind even though their scales differ (which is what calibration measures).
BACKBONES: dict[str, int] = {
    "vit_base_patch16_dinov3.lvd1689m": 384,    # 24x24 patches
    "vit_small_patch16_dinov3.lvd1689m": 384,   # same grid, a third the cost
    "vit_base_patch14_dinov2.lvd142m": 392,     # 28x28, patch 14
    "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k": 384,
}


@dataclass(frozen=True)
class EmbedParams:
    """Frozen ViT embeddings, CLS token (patch-mean collapses on bare
    concrete: median random-pair cosine 0.95)."""

    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    img_size: int = 384       # 24x24 patch grid, enough for grey-on-grey
    batch_size: int = 8
    device: str | None = None # None = auto: cuda, then mps, then cpu


# The segmenters, and which kind each one is.
#
# "semantic" models label every pixel from a fixed list. All the ones here are
# ADE20K, the only public label set with an interior's vocabulary, and they are
# bounded by it: it has wall, floor, ceiling, door and window, and no rebar,
# formwork or conduit at all. Measured on ASHV bare RCC (Aug 2026), Mask2Former
# traces the wall/floor junction down a corridor that SegFormer-B4 reads as
# wall, which is why it is the default.
#
# "open" models take the classes as text, so the vocabulary is whatever you
# write. That is the only way to ask for the things a site is actually made of,
# at the cost of finding instances rather than covering the frame.
SEGMENTERS: dict[str, str] = {
    "facebook/mask2former-swin-large-ade-semantic": "semantic",
    "shi-labs/oneformer_ade20k_swin_large": "semantic",
    "nvidia/segformer-b4-finetuned-ade-512-512": "semantic",
    "nvidia/segformer-b0-finetuned-ade-512-512": "semantic",
    "IDEA-Research/grounding-dino-base": "open",
}

# What a site is made of, in the words someone would use for it. Only the open
# models read this; a semantic one is stuck with the list it was trained on.
SITE_CLASSES: tuple[str, ...] = (
    "rebar", "formwork", "scaffolding", "concrete column", "concrete beam",
    "concrete slab", "block wall", "electrical conduit", "pipe", "duct",
    "window opening", "door opening", "staircase", "debris", "plaster",
)


@dataclass(frozen=True)
class SegmentParams:
    """What is in each kept frame, annotated before anyone labels it.

    Runs on the frames Select kept rather than every face: it is the export
    that gets annotated, and inference is the expensive part. It never drops a
    frame, so a wrong mask costs a correction and not a candidate.
    """

    enabled: bool = False     # opt-in: it pulls a model and takes real time
    model_name: str = "facebook/mask2former-swin-large-ade-semantic"
    classes: tuple[str, ...] = SITE_CLASSES   # open-vocabulary models only
    threshold: float = 0.2    # detection confidence, same
    device: str | None = None

    @property
    def kind(self) -> str:
        return SEGMENTERS.get(self.model_name, "semantic")


@dataclass(frozen=True)
class CalibParams:
    """Two measurements per walk, so the threshold is a stated risk.

    The reference is a kept face against the raw frame straight after it: the
    ceiling for "identical", and a health check on the stitch, the exposure and
    the backbone. It does not set the threshold. Measured on GCMR, ASHV and
    114811 (Aug 2026) it sits at 0.966 to 0.984, while faces one gated interval
    apart score 0.70 to 0.91, so a threshold hung off the reference merges 4%
    to 15% of genuinely adjacent pairs and select does almost nothing.

    The far distribution sets it instead. Faces at one heading more than
    far_seconds apart are somewhere else, so how many of those a threshold
    merges is the risk it carries, and the budget names it directly.

    The budget is generous on purpose. Merging two frames costs nothing
    permanent: render_faces writes every face and the manifest only marks which
    ones were kept, so an over-cut walk comes back with a lower budget. An
    under-cut one has already been labelled, and that money does not come back.
    Duplicates are also the expensive kind of wrong: an operator standing in a
    stairwell for 30 seconds puts fifteen near-identical stairwell frames into
    training and skews the model, not just the invoice.

    What it cannot do is rescue a self-similar walk. Measured on 114811 (Aug
    2026), cutting 148 faces to 67 moves the mean nearest-neighbour cosine in
    the kept set from 0.949 to 0.904: the frames go, the redundancy stays,
    because everything there genuinely looks alike. Sampling on distance is the
    instrument for that. This one only decides what to keep of what was taken.
    """

    samples: int = 10             # panoramas the reference is measured on; x4
    far_seconds: float = 20.0     # apart enough to be a different place
    false_merge_pct: float = 5.0  # of far pairs allowed to merge; tau is the
                                  # (100 - this) percentile of them. Cuts 44-59%
                                  # on real walks, and every frame stays on disk


@dataclass(frozen=True)
class DedupParams:
    """Greedy cosine dedup, per walk (cross-walk near-duplicates are different
    walls that look alike)."""

    tau: float = 0.94         # a starting point, not a measurement: what it
                              # merges depends entirely on the site
    rule: str = "fixed"       # "fixed" uses tau as given; "calibrated" takes it
                              # from the walk's own far-apart pairs, at the
                              # false-merge budget, and writes it back into tau


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
    """Rebuild params saved next to a walk. JSON has no tuples, so the fields
    that are tuples come back as lists and are converted here."""
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
