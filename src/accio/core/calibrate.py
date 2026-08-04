"""What "identical" scores on this walk, measured rather than assumed.

A cosine of 0.94 means nothing on its own. It only means something next to
the score of two frames that are the same scene: take a kept frame and the
raw frame immediately after it, a fraction of a second later, and whatever
they score is this walk's ceiling for "the same thing", with this camera,
this stitch and this backbone.

Measured on GCMR and ASHV footage (Aug 2026) that ceiling sat at 0.985 on
both, while ordinary walking steps half a second apart ranged from 0.70 to
0.90 between the two. So the reference is the stable quantity and the
redundancy is not, which is why the threshold hangs off the reference.

It is measured on every run whether or not the threshold uses it: a walk
whose reference comes back low is a walk where the stitch, the exposure or
the backbone is misbehaving, and a fixed threshold above the ceiling would
silently merge nothing at all.
"""

import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import extract, faces as faces_mod
from .params import PipelineParams

SAMPLE_PANOS = 10        # ~40 pairs at four yaws, enough for a 5th percentile
QUANTILE = 5.0           # merge what is as alike as 95% of identical pairs
FLOOR, CEILING = 0.85, 0.995
HEALTHY_REFERENCE = 0.95  # a median below this means the reference is suspect
CALIB_DIR = "calib"


def sample_panos(pano_idx: list[int], n: int = SAMPLE_PANOS) -> list[int]:
    """Evenly spread through the walk, so one static stretch cannot dominate."""
    uniq = sorted(set(pano_idx))
    if len(uniq) <= n:
        return uniq
    return [uniq[round(i * (len(uniq) - 1) / (n - 1))] for i in range(n)]


def resolve_tau(cosines: np.ndarray) -> float:
    if len(cosines) < 8:
        return PipelineParams().dedup.tau
    return round(float(np.clip(np.percentile(cosines, QUANTILE), FLOOR, CEILING)), 4)


def calibrate(video: Path, out: Path, faces: list[tuple[int, float, int, str]],
              embeddings: np.ndarray, params: PipelineParams, embedder,
              native_fps: float) -> dict:
    """Stitch the frame straight after a sample of kept panoramas and compare.

    faces is (pano_idx, t_sec, yaw, file name) in embedding order. Returns the
    record the UI shows: every pair with its cosine, the resolved threshold,
    and whether the reference looks healthy.
    """
    t_of: dict[int, float] = {}
    row_of: dict[tuple[int, int], int] = {}
    for i, (pano, t_sec, yaw, _) in enumerate(faces):
        t_of[pano] = t_sec
        row_of[(pano, yaw)] = i

    chosen = sample_panos([p for p, _, _, _ in faces])
    frame_nos = [round(t_of[p] * native_fps) + 1 for p in chosen]

    calib_dir = out / CALIB_DIR
    calib_dir.mkdir(parents=True, exist_ok=True)
    cname = "calib-" + re.sub(r"[^a-zA-Z0-9_.-]", "", out.name)[:40]
    try:
        subprocess.run(
            extract.sdk_cmd(video, calib_dir, frame_nos, cname, params.extract),
            check=True, capture_output=True, timeout=120 + 8 * len(frame_nos))
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        subprocess.run(["docker", "kill", cname], capture_output=True)

    exported = {int(p.stem): p for p in calib_dir.glob("*.jpg") if p.stem.isdigit()}
    pairs = []
    for pano, frame_no in zip(chosen, frame_nos):
        stitched = exported.get(frame_no)
        if stitched is None:
            continue
        pano_img = cv2.imread(str(stitched))
        for yaw, img in faces_mod.render_faces(pano_img, params.faces).items():
            row = row_of.get((pano, yaw))
            if row is None:
                continue
            name = f"n{pano:05d}_y{yaw:03d}.jpg"
            cv2.imwrite(str(calib_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            vec = embedder.embed([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)])[0]
            pairs.append({
                "pano": pano, "yaw": yaw, "tSec": round(t_of[pano], 1),
                "face": faces[row][3], "neighbour": name,
                "cosine": round(float(vec @ embeddings[row]), 4),
            })
        stitched.unlink()          # the panorama was scratch; the faces are not

    cos = np.array([p["cosine"] for p in pairs])
    median = round(float(np.median(cos)), 4) if len(cos) else 0.0
    return {
        "tau": resolve_tau(cos),
        "quantile": QUANTILE,
        "pairs": sorted(pairs, key=lambda p: p["cosine"]),
        "reference": {
            "median": median,
            "p05": round(float(np.percentile(cos, 5)), 4) if len(cos) else 0.0,
            "min": round(float(cos.min()), 4) if len(cos) else 0.0,
            "n": len(cos),
        },
        "healthy": median >= HEALTHY_REFERENCE,
        "gapSeconds": round(1.0 / native_fps, 4),
    }
