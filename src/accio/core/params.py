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
    """Relative motion-blur gate on panoramas."""

    window: int = 4           # at 2 fps: keep the sharpest pano per 2 s of walk
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


@dataclass(frozen=True)
class CalibParams:
    """Measuring what "identical" scores on this walk, to set the threshold by.

    A kept frame and the raw frame straight after it are the same scene, so
    what they score is the ceiling for "the same thing" with this camera, this
    stitch and this backbone. The threshold sits at a low percentile of that,
    so a merge needs two frames about as alike as a pair taken a frame apart.
    """

    samples: int = 10         # panoramas spread through the walk; x4 faces each
    quantile: float = 5.0     # merge what is as alike as 95% of identical pairs


@dataclass(frozen=True)
class DedupParams:
    """Greedy cosine dedup, per walk (cross-walk near-duplicates are different
    walls that look alike)."""

    tau: float = 0.94         # sits in the measured gap: different walls 0.936,
                              # same wall one step later 0.965
    rule: str = "fixed"       # "fixed" uses tau as given; "calibrated" takes it
                              # from the walk's measured identical-content
                              # reference and writes that value back into tau


@dataclass(frozen=True)
class PipelineParams:
    extract: ExtractParams = field(default_factory=ExtractParams)
    faces: FaceParams = field(default_factory=FaceParams)
    gate: GateParams = field(default_factory=GateParams)
    embed: EmbedParams = field(default_factory=EmbedParams)
    calib: CalibParams = field(default_factory=CalibParams)
    dedup: DedupParams = field(default_factory=DedupParams)


def from_dict(d: dict) -> PipelineParams:
    """Rebuild params saved next to a walk. JSON has no tuples, so the fields
    that are tuples come back as lists and are converted here."""
    faces = dict(d.get("faces", {}))
    if "yaws" in faces:
        faces["yaws"] = tuple(faces["yaws"])
    gate = dict(d.get("gate", {}))
    if "band" in gate:
        gate["band"] = tuple(gate["band"])
    return PipelineParams(
        extract=ExtractParams(**d.get("extract", {})),
        faces=FaceParams(**faces),
        gate=GateParams(**gate),
        embed=EmbedParams(**d.get("embed", {})),
        calib=CalibParams(**d.get("calib", {})),
        dedup=DedupParams(**d.get("dedup", {})),
    )
