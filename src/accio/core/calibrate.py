"""Per-walk dedup threshold from a false-merge budget.

Faces at one heading more than far_seconds apart are taken to be different
places; tau is the percentile of their cosines that merges false_merge_pct of
them. Measured on GCMR, ASHV and 114811 (Aug 2026): a kept face against the
raw frame after it scores 0.966-0.984, faces one gated interval apart 0.70-0.91,
so a threshold set from the near end merges 4-15% of adjacent pairs.

The reference (kept face vs the re-stitched neighbouring frame) is a health
check only: a low median means the stitch, exposure or backbone is off. If the
SDK will not run, the record says so and tau is unaffected.
"""

import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import extract, faces as faces_mod
from .params import CalibParams, PipelineParams

# tau is not clipped: ASHV measures far p99 = 0.81, and a floor of 0.85 would
# override the budget
HEALTHY_REFERENCE = 0.95  # a reference median below this is suspect
MIN_FAR = 50              # fewer far pairs and a percentile is noise
FAR_KEPT = 5000           # a sorted subsample this size holds every percentile
                          # to about 0.0002
CALIB_DIR = "calib"


def sample_panos(pano_idx: list[int], n: int) -> list[int]:
    """n panorama indices evenly spread through the walk."""
    if n <= 0:
        return []
    uniq = sorted(set(pano_idx))
    if len(uniq) <= n:
        return uniq
    if n == 1:
        return uniq[:1]
    return [uniq[round(i * (len(uniq) - 1) / (n - 1))] for i in range(n)]


def far_cosines(faces: list[tuple[int, float, int, str]], embeddings: np.ndarray,
                far_seconds: float) -> np.ndarray:
    """Cosines between same-yaw faces more than far_seconds apart.

    Same yaw only: two yaws of one station look unalike for other reasons and
    would drag the distribution down.
    """
    by_yaw: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for i, (_pano, t_sec, yaw, _name) in enumerate(faces):
        by_yaw[yaw].append((t_sec, i))

    out = []
    for rows in by_yaw.values():
        rows.sort()
        t = np.array([r[0] for r in rows])
        v = embeddings[[r[1] for r in rows]]
        apart = np.triu(np.abs(t[:, None] - t[None, :]) > far_seconds, k=1)
        out.append((v @ v.T)[apart])
    return np.concatenate(out) if out else np.array([])


def thin(cos: np.ndarray, n: int = FAR_KEPT) -> list[float]:
    """Sorted and evenly subsampled to n."""
    cos = np.sort(cos)
    if len(cos) > n:
        cos = cos[np.linspace(0, len(cos) - 1, n).round().astype(int)]
    return [round(float(c), 4) for c in cos]


def resolve_tau(far: np.ndarray, params: CalibParams) -> float | None:
    """The far-cosine percentile meeting the budget; None with too few far pairs."""
    if len(far) < MIN_FAR:
        return None
    return round(float(np.percentile(far, 100.0 - params.false_merge_pct)), 4)


def record(pairs: list[dict], far: np.ndarray, params: CalibParams,
           gap_seconds: float, reference_error: str = "") -> dict:
    """The calibration record the UI shows."""
    cos = np.array([p["cosine"] for p in pairs])
    median = round(float(np.median(cos)), 4) if len(cos) else 0.0
    far = np.asarray(far, dtype=float)
    pct = (lambda p: round(float(np.percentile(far, p)), 4)) if len(far) \
        else (lambda p: 0.0)
    return {
        "tau": resolve_tau(far, params),
        "falseMergePct": params.false_merge_pct,
        "farSeconds": params.far_seconds,
        "samples": params.samples,
        "pairs": sorted(pairs, key=lambda p: p["cosine"]),
        "far": {
            "n": int(len(far)),
            "median": pct(50),
            "p95": pct(95),
            "p99": pct(99),
            "max": round(float(far.max()), 4) if len(far) else 0.0,
            "cosines": thin(far),
        },
        "reference": {
            "median": median,
            "p05": round(float(np.percentile(cos, 5)), 4) if len(cos) else 0.0,
            "min": round(float(cos.min()), 4) if len(cos) else 0.0,
            "n": len(cos),
        },
        # a check that did not run is not healthy
        "healthy": bool(len(cos)) and median >= HEALTHY_REFERENCE,
        "referenceError": reference_error,
        "gapSeconds": round(gap_seconds, 4),
    }


def calibrate(video: Path, out: Path, faces: list[tuple[int, float, int, str]],
              embeddings: np.ndarray, params: PipelineParams, embedder,
              native_fps: float) -> dict:
    """faces is (pano_idx, t_sec, yaw, file name) in embedding order.

    Only the reference needs the SDK; if that fails the record says so.
    """
    t_of: dict[int, float] = {}
    row_of: dict[tuple[int, int], int] = {}
    for i, (pano, t_sec, yaw, _) in enumerate(faces):
        t_of[pano] = t_sec
        row_of[(pano, yaw)] = i

    far = far_cosines(faces, embeddings, params.calib.far_seconds)

    chosen = sample_panos([p for p, _, _, _ in faces], params.calib.samples)
    frame_nos = [round(t_of[p] * native_fps) + 1 for p in chosen]

    calib_dir = out / CALIB_DIR
    calib_dir.mkdir(parents=True, exist_ok=True)
    # stale frames from a run with different samples would be read back as this run's
    for old in calib_dir.glob("*.jpg"):
        old.unlink()
    cname = "calib-" + re.sub(r"[^a-zA-Z0-9_.-]", "", out.name)[:40]
    failed = ""
    # a container still dying holds the name
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
    try:
        subprocess.run(
            extract.sdk_cmd(video, calib_dir, frame_nos, cname, params.extract),
            check=True, capture_output=True, timeout=120 + 8 * len(frame_nos))
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", cname], capture_output=True)
        failed = "the MediaSDK container timed out"
    except subprocess.CalledProcessError as e:
        subprocess.run(["docker", "kill", cname], capture_output=True)
        tail = (e.stderr or b"").decode("utf-8", "replace").strip()[-400:]
        failed = f"MediaSDK exited {e.returncode}: {tail}" if tail \
            else f"MediaSDK exited {e.returncode}"

    exported = {int(p.stem): p for p in calib_dir.glob("*.jpg")
                if p.stem.isdigit() and extract.whole_jpeg(p)}
    pairs = []
    for pano, frame_no in zip(chosen, frame_nos):
        stitched = exported.get(frame_no)
        if stitched is None:
            continue
        pano_img = cv2.imread(str(stitched))
        if pano_img is None:
            stitched.unlink()
            continue
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
        stitched.unlink()          # the panorama was scratch; the faces stay

    if not pairs and not failed:
        failed = f"MediaSDK exported none of {len(frame_nos)} reference frames"
    return record(pairs, far, params.calib, 1.0 / native_fps, failed)
