"""Pipeline parameters with the defaults validated in the dedup PoC (June 2026).

Every stage takes its params object explicitly; nothing reads globals. Change a
default here only with evidence (the PoC's borderline sheets, or an override
pattern from review).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractParams:
    """Dual-fisheye .insv -> decimated equirect panoramas via ffmpeg v360."""

    fps: float = 2.0          # walking pace + full sphere per frame
    pano_width: int = 3840    # native resolution; height is width / 2
    lens_fov: float = 195.0   # per-lens FOV; calibrated by seam-continuity sweep
                              # on GCMR footage, residual error is lens parallax
    jpeg_quality: int = 2     # ffmpeg -q:v, 2 is near-lossless

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


@dataclass(frozen=True)
class EmbedParams:
    """Frozen DINOv3 embeddings, CLS token (patch-mean collapses on bare
    concrete: median random-pair cosine 0.95)."""

    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    img_size: int = 384       # 24x24 patch grid, enough for grey-on-grey
    batch_size: int = 8
    device: str | None = None # None = auto: cuda, then mps, then cpu


@dataclass(frozen=True)
class DedupParams:
    """Greedy cosine dedup, per walk (cross-walk near-duplicates are different
    walls that look alike)."""

    tau: float = 0.94         # sits in the measured gap: different walls 0.936,
                              # same wall one step later 0.965


@dataclass(frozen=True)
class PipelineParams:
    extract: ExtractParams = field(default_factory=ExtractParams)
    faces: FaceParams = field(default_factory=FaceParams)
    gate: GateParams = field(default_factory=GateParams)
    embed: EmbedParams = field(default_factory=EmbedParams)
    dedup: DedupParams = field(default_factory=DedupParams)
