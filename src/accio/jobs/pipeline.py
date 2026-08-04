"""The pipeline end to end: one .insv walk -> deduplicated label candidates.

run_walk() writes out/<walk>/pano/, out/<walk>/faces/,
out/<walk>/manifest.csv with a kept flag plus anchor/cosine for every
absorbed face, and out/<walk>/embeddings.npz with one row per manifest
row. The ingest job calls this; there is no other entry point.
"""

import csv
import json
from dataclasses import asdict, replace
from itertools import batched
from pathlib import Path

import cv2
import numpy as np

from ..core import blur, extract, faces
from ..core.calibrate import calibrate
from ..core.dedup import greedy_dedup
from ..core.embed import Embedder
from ..core.params import PipelineParams

EMBED_CHUNK = 64  # faces held in memory at once on the way to the embedder
MANIFEST_COLUMNS = ["pano_idx", "t_sec", "yaw", "path", "kept", "anchor", "cosine"]


def face_name(pano_idx: int, yaw: int) -> str:
    return f"y{yaw:03d}_{pano_idx:05d}.jpg"


def run_walk(video: Path, out_root: Path, params: PipelineParams,
             embedder: Embedder, progress=None) -> dict:
    """progress(stage, status, **counts) is called around every stage, so the
    caller can show the run advancing instead of a single opaque wait."""
    say = progress or (lambda *a, **k: None)
    walk = video.stem
    out = out_root / walk
    out.mkdir(parents=True, exist_ok=True)

    # what the camera actually recorded, kept next to the derived frames: it is
    # the denominator the rest of the funnel is a fraction of
    native_fps, frames = extract.probe_fps_nframes(video)
    (out / "source.json").write_text(json.dumps(
        {"frames": frames, "fps": native_fps, "seconds": round(frames / native_fps, 1),
         "files": [p.name for p in extract.lens_files(video)]}))
    say("video", "done", frames=frames, seconds=round(frames / native_fps))

    say("stitch", "running")
    panos = extract.stitch(video, out / "pano", params.extract)
    say("stitch", "done", panos=len(panos))

    say("gate", "running")
    scores = [blur.score_pano(cv2.imread(str(p.path)), params.gate) for p in panos]
    sharp = [p for p, keep in zip(panos, blur.windowed_keep(scores, params.gate.window))
             if keep]
    say("gate", "done", sharp=len(sharp))

    say("faces", "running")
    faces_dir = out / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    records = []  # (pano, yaw, path) in capture order
    for p in sharp:
        pano_img = cv2.imread(str(p.path))
        for yaw, img in faces.render_faces(pano_img, params.faces).items():
            path = faces_dir / face_name(p.index, yaw)
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append((p, yaw, path))
    say("faces", "done", faces=len(records))

    say("embed", "running")
    embeddings = np.concatenate([
        embedder.embed([cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                        for _, _, path in chunk])
        for chunk in batched(records, EMBED_CHUNK)
    ])
    # embeddings outlive the walk: registry dedup against re-walks, leak checks
    # and future selection all reuse them. Cosines are only comparable between
    # walks embedded by the same model, so the model name travels with the rows.
    np.savez(out / "embeddings.npz",
             embeddings=embeddings, model=params.embed.model_name)
    say("embed", "done")

    say("select", "running")
    # what "identical" scores on this walk, measured every run: the threshold
    # only uses it under the calibrated rule, but a low reference is a warning
    # worth having either way
    face_rows = [(p.index, p.t_sec, yaw, path.name) for p, yaw, path in records]
    calib = calibrate(video, out, face_rows, embeddings, params, embedder,
                      native_fps)
    (out / "calibration.json").write_text(json.dumps(calib, indent=1))
    params = apply_rule(params, calib)
    result = greedy_dedup(embeddings, params.dedup)
    write_manifest(out / "manifest.csv",
                   [(p.index, p.t_sec, yaw, path.name) for p, yaw, path in records],
                   result)
    save_params(out, params)
    kept = len(result.kept)
    say("select", "done", anchors=kept, absorbed=len(records) - kept)

    return dict(walk=walk, panos=len(panos), sharp=len(sharp),
                faces=len(records), kept=kept)


def apply_rule(params: PipelineParams, calib: dict | None) -> PipelineParams:
    """Under the calibrated rule the threshold comes from the measurement, so
    tau always holds the number that actually ran."""
    if params.dedup.rule != "calibrated" or not calib:
        return params
    return replace(params, dedup=replace(params.dedup, tau=calib["tau"]))


def write_manifest(path: Path, faces_out: list[tuple[int, float, int, str]],
                   result) -> None:
    """One row per face in capture order, with the group it landed in.

    faces_out is (pano_idx, t_sec, yaw, file name); the dedup result supplies
    kept / anchor / cosine. Shared by the full run and by a re-select, so the
    two can never disagree about the manifest's shape.
    """
    kept = set(result.kept)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLUMNS)
        for i, (pano_idx, t_sec, yaw, name) in enumerate(faces_out):
            anchor, cos = ("", "")
            if i in result.anchor_of:
                a, c = result.anchor_of[i]
                anchor, cos = str(a), f"{c:.4f}"
            w.writerow([pano_idx, f"{t_sec:.1f}", yaw, name, int(i in kept),
                        anchor, cos])


def save_params(out: Path, params: PipelineParams) -> None:
    """The settings this walk's frames were built with, next to the frames."""
    (out / "params.json").write_text(json.dumps(asdict(params), indent=1))


def reselect(walk_out: Path, params: PipelineParams) -> dict:
    """Re-run selection alone, from the embeddings already on disk.

    Everything upstream of Select is unchanged by a threshold, so this costs
    a dedup pass and a manifest rewrite rather than a stitch and a GPU pass.
    Face files and their names are untouched, which is what lets review
    decisions (keyed by name) survive the change.
    """
    with np.load(walk_out / "embeddings.npz") as data:
        embeddings = data["embeddings"]
    with open(walk_out / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(embeddings):
        raise ValueError(f"manifest has {len(rows)} rows but "
                         f"{len(embeddings)} embeddings")

    calib_file = walk_out / "calibration.json"
    calib = json.loads(calib_file.read_text()) if calib_file.exists() else None
    params = apply_rule(params, calib)
    result = greedy_dedup(embeddings, params.dedup)
    write_manifest(walk_out / "manifest.csv",
                   [(int(r["pano_idx"]), float(r["t_sec"]), int(r["yaw"]), r["path"])
                    for r in rows], result)
    save_params(walk_out, params)
    kept = len(result.kept)
    return dict(faces=len(rows), anchors=kept, absorbed=len(rows) - kept,
                tau=params.dedup.tau, rule=params.dedup.rule)
