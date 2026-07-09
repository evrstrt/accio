"""Stage 1: dual-fisheye .insv -> decimated equirect panorama JPEGs.

ffmpeg's v360 filter does the stitch; we only decimate to walking pace and
scale to native width. The subprocess call is isolated in stitch() so the
rest of the module (naming, timestamps) is pure and testable.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .params import ExtractParams

PANO_PATTERN = "pano_%05d.jpg"


@dataclass(frozen=True)
class PanoFrame:
    index: int
    t_sec: float
    path: Path


class FfmpegNotFound(RuntimeError):
    pass


def stitch(video: Path, pano_dir: Path, params: ExtractParams) -> list[PanoFrame]:
    """Run ffmpeg over the .insv and return the extracted panoramas in order."""
    if shutil.which("ffmpeg") is None:
        raise FfmpegNotFound("ffmpeg not on PATH; install it (brew install ffmpeg)")
    pano_dir.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={params.fps},"
        f"v360=input=dfisheye:output=equirect"
        f":ih_fov={params.lens_fov}:iv_fov={params.lens_fov},"
        f"scale={params.pano_width}:{params.pano_height}"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vf", vf, "-start_number", "0", "-q:v", "2",
         str(pano_dir / PANO_PATTERN)],
        check=True,
    )
    return pano_frames(pano_dir, params.fps)


def pano_frames(pano_dir: Path, fps: float) -> list[PanoFrame]:
    """List extracted panoramas with their timestamps, in walk order."""
    frames = []
    for path in sorted(pano_dir.glob("pano_*.jpg")):
        index = int(path.stem.split("_")[1])
        frames.append(PanoFrame(index=index, t_sec=index / fps, path=path))
    return frames
