"""What this walk's own frames score, so the threshold is a stated risk.

A cosine of 0.94 means nothing on its own, and neither does the score of two
frames a fraction of a second apart. Measured on GCMR, ASHV and 114811 (Aug
2026), a kept face against the raw frame straight after it scores 0.966 to
0.984, while faces one gated interval apart, which is what dedup is actually
asked to judge, score 0.70 to 0.91. A threshold set just under the first
number merges 4% to 15% of genuinely adjacent pairs. It cuts nothing.

So the threshold hangs off the other end: pairs at one heading far enough
apart in time to be somewhere else. How many of those a threshold merges is
the risk it carries, so that is the knob, as a budget, and tau is whatever
meets it here. It needs no stitching, because those faces are already
rendered and already embedded.

The budget runs generous. Merging costs nothing permanent, since every face
stays in faces/ and the manifest only records which were kept, so a walk cut
too hard comes back with a lower budget. A walk cut too softly has already
been labelled twice over, and that is the money and the class balance both.

The reference is still measured every run, as the health check it always was:
a walk whose identical-content ceiling comes back low is a walk where the
stitch, the exposure or the backbone is misbehaving. It no longer decides
anything, so when the SDK will not run for it, the run says so and carries on
rather than inventing a threshold.
"""

import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import extract, faces as faces_mod
from .params import CalibParams, PipelineParams

# No guard rail on the threshold itself. It used to be clipped to [0.85, 0.995]
# on the reasoning that a value outside that was a symptom rather than a
# measurement, which was true when it came from the identical-content ceiling
# and always landed near 0.95. Placed against the far pairs it is legitimately
# much lower on a site whose bays are distinguishable: ASHV measures p99 = 0.81,
# and clipping that to 0.85 silently overrode the budget with a number nobody
# asked for. The budget is the guard rail now, and it holds by construction. A
# broken backbone shows up in the reference, which is what it is for.
HEALTHY_REFERENCE = 0.95  # a median below this means the reference is suspect
MIN_FAR = 50              # fewer far pairs than this and a percentile is noise
FAR_KEPT = 5000           # far cosines stored, so the budget can be re-explored
                          # without re-measuring; a sorted subsample of this
                          # size holds every percentile to about 0.0002
CALIB_DIR = "calib"


def sample_panos(pano_idx: list[int], n: int) -> list[int]:
    """Evenly spread through the walk, so one static stretch cannot dominate."""
    uniq = sorted(set(pano_idx))
    if len(uniq) <= n:
        return uniq
    if n < 2:
        return uniq[:1]
    return [uniq[round(i * (len(uniq) - 1) / (n - 1))] for i in range(n)]


def far_cosines(faces: list[tuple[int, float, int, str]], embeddings: np.ndarray,
                far_seconds: float) -> np.ndarray:
    """What two different places score, from vectors select already holds.

    One heading at a time: two yaws of the same station look unalike for
    reasons that have nothing to do with being somewhere else, and mixing them
    in would drag the distribution down and the threshold with it.
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
    """Sorted and evenly subsampled: a percentile sketch, not the raw cloud.

    A long walk has hundreds of thousands of far pairs and the record only has
    to answer percentile questions about them.
    """
    cos = np.sort(cos)
    if len(cos) > n:
        cos = cos[np.linspace(0, len(cos) - 1, n).round().astype(int)]
    return [round(float(c), 4) for c in cos]


def resolve_tau(far: np.ndarray, params: CalibParams) -> float | None:
    """The threshold that meets the false-merge budget on this walk.

    None when there is not enough of a far distribution to place one, which is
    a real answer for a short walk: it means use the fixed rule, not that some
    default is fine here.
    """
    if len(far) < MIN_FAR:
        return None
    return round(float(np.percentile(far, 100.0 - params.false_merge_pct)), 4)


def record(pairs: list[dict], far: np.ndarray, params: CalibParams,
           gap_seconds: float, reference_error: str = "") -> dict:
    """The measurement as the UI shows it: the reference pairs, the far
    distribution the threshold comes from, and what it resolves to."""
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
        # the reference is a health check now, so "unhealthy" has to mean the
        # stitch looked wrong, not that the check could not be run at all
        "healthy": bool(len(cos)) and median >= HEALTHY_REFERENCE,
        "referenceError": reference_error,
        "gapSeconds": round(gap_seconds, 4),
    }


def calibrate(video: Path, out: Path, faces: list[tuple[int, float, int, str]],
              embeddings: np.ndarray, params: PipelineParams, embedder,
              native_fps: float) -> dict:
    """Measure both ends: the identical-content ceiling, and what elsewhere
    scores.

    faces is (pano_idx, t_sec, yaw, file name) in embedding order. The far
    distribution comes straight off the embeddings; only the reference needs
    the SDK, and if that will not run the record says so and the threshold is
    unaffected.
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
    # a stale set from a run with different samples would otherwise be read
    # back as though this run had measured it
    for old in calib_dir.glob("*.jpg"):
        old.unlink()
    cname = "calib-" + re.sub(r"[^a-zA-Z0-9_.-]", "", out.name)[:40]
    failed = ""
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

    if not pairs and not failed:
        failed = f"MediaSDK exported none of {len(frame_nos)} reference frames"
    return record(pairs, far, params.calib, 1.0 / native_fps, failed)
