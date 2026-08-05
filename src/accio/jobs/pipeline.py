"""The pipeline end to end: one .insv walk -> deduplicated label candidates.

run_walk() writes out/<walk>/pano/, out/<walk>/faces/,
out/<walk>/manifest.csv with a kept flag plus anchor/cosine for every
absorbed face, and out/<walk>/embeddings.npz with one row per manifest
row. The ingest job calls this; there is no other entry point.

Changing a setting re-runs from the stage that setting belongs to, not from
the start: rerun() enters the same chain partway down and reuses whatever is
already on disk above it. That is why each stage is its own function and why
the stages that persist their output (stitch, gate, faces) can all be read
back as well as written.
"""

import csv
import json
import shutil
from dataclasses import asdict, replace
from datetime import datetime
from itertools import batched
from pathlib import Path

import cv2
import numpy as np

from ..core import blur, calibrate as calibrate_mod, extract, faces
from ..core.calibrate import calibrate
from ..core.dedup import greedy_dedup
from ..core.embed import Embedder
from ..core.params import PipelineParams

EMBED_CHUNK = 64  # faces held in memory at once on the way to the embedder
MANIFEST_COLUMNS = ["pano_idx", "t_sec", "yaw", "path", "kept", "anchor", "cosine"]
STAGES = ("stitch", "gate", "faces", "embed", "calibrate", "select")

# a face on its way through the pipeline: (pano index, t, yaw, file)
FaceRow = tuple[int, float, int, Path]


def face_name(pano_idx: int, yaw: int) -> str:
    return f"y{yaw:03d}_{pano_idx:05d}.jpg"


def run_walk(video: Path, out_root: Path, params: PipelineParams,
             embedder: Embedder, progress=None) -> dict:
    """progress(stage, status, **counts) is called around every stage, so the
    caller can show the run advancing instead of a single opaque wait."""
    say = progress or (lambda *a, **k: None)
    out = out_root / video.stem
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

    return _tail(video, out, params, embedder, native_fps, "gate", say,
                 panos=panos)


def rerun(video: Path, out: Path, params: PipelineParams, embedder: Embedder,
          first: str, progress=None) -> dict:
    """Re-run from `first` down, reusing the stages above it as they are.

    The stages above are reported done rather than skipped silently, so the
    canvas shows the same blocks either way and the ones that were reused are
    visibly not running.
    """
    say = progress or (lambda *a, **k: None)
    if first not in STAGES:
        raise ValueError(f"unknown stage {first!r}")
    if first == "stitch":
        raise ValueError("re-stitching is not wired up yet; re-ingest the walk")
    native_fps = json.loads((out / "source.json").read_text())["fps"]
    say("video", "done")
    for stage in STAGES[:STAGES.index(first)]:
        say(stage, "done")
    return _tail(video, out, params, embedder, native_fps, first, say)


def _tail(video: Path, out: Path, params: PipelineParams, embedder: Embedder,
          native_fps: float, first: str, say, panos=None) -> dict:
    """The pipeline below the stitch, entered at `first`."""
    start = STAGES.index(first)

    if start <= STAGES.index("faces"):
        if start <= STAGES.index("gate"):
            say("gate", "running")
            sharp = gate(panos if panos is not None
                         else extract.load_panos(out / "pano"), params)
            say("gate", "done", sharp=len(sharp))
        else:
            sharp = sharp_panos(out)   # unchanged by this re-run, read back
        say("faces", "running")
        records = render_faces(out, sharp, params)
        say("faces", "done", faces=len(records))
    else:
        records = face_rows(out)

    if start <= STAGES.index("embed"):
        say("embed", "running")
        embeddings = embed_faces(out, records, params, embedder)
        say("embed", "done")
    else:
        with np.load(out / "embeddings.npz") as data:
            embeddings = data["embeddings"]

    named = [(i, t, yaw, path.name) for i, t, yaw, path in records]
    # what "identical" scores on this walk. Measured whether or not the
    # threshold uses it: a low reference means the stitch, the exposure or the
    # backbone is misbehaving, which is worth knowing either way.
    if start <= STAGES.index("calibrate"):
        say("calibrate", "running")
        calib = calibrate(video, out, named, embeddings, params, embedder,
                          native_fps)
        save_calibration(out, calib)
        say("calibrate", "done", pairs=calib["reference"]["n"],
            reference=calib["reference"]["median"], calibTau=calib["tau"])
    else:
        calib = read_calibration(out)

    say("select", "running")
    params = apply_rule(params, calib)
    result = greedy_dedup(embeddings, params.dedup)
    write_manifest(out / "manifest.csv", named, result)
    save_params(out, params)
    kept = len(result.kept)
    say("select", "done", anchors=kept, absorbed=len(records) - kept)

    log_run(out, params, calib, first, len(records), kept)
    clear_failure(out)     # this walk ran through; whatever broke before is past
    return dict(walk=out.name, faces=len(records), anchors=kept,
                absorbed=len(records) - kept, tau=params.dedup.tau,
                rule=params.dedup.rule)


def gate(panos: list, params: PipelineParams) -> list:
    """The sharpest panorama per window; the rest were walked through."""
    scores = [blur.score_pano(cv2.imread(str(p.path)), params.gate) for p in panos]
    return [p for p, keep in zip(panos, blur.windowed_keep(scores, params.gate.window))
            if keep]


def render_faces(out: Path, sharp: list, params: PipelineParams) -> list[FaceRow]:
    """Project every sharp panorama into its faces, replacing any earlier set.

    The directory is cleared first: a change to the yaws renames every face, and
    leaving the old ones behind would put frames in the export that no manifest
    row points at. Review decisions are keyed by face name, so they survive a
    re-render that keeps the names and go stale on one that does not.
    """
    faces_dir = out / "faces"
    shutil.rmtree(faces_dir, ignore_errors=True)
    faces_dir.mkdir(parents=True, exist_ok=True)
    records: list[FaceRow] = []
    for p in sharp:
        pano_img = cv2.imread(str(p.path))
        for yaw, img in faces.render_faces(pano_img, params.faces).items():
            path = faces_dir / face_name(p.index, yaw)
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append((p.index, p.t_sec, yaw, path))
    return records


def embed_faces(out: Path, records: list[FaceRow], params: PipelineParams,
                embedder: Embedder) -> np.ndarray:
    embeddings = np.concatenate([
        embedder.embed([cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                        for _, _, _, path in chunk])
        for chunk in batched(records, EMBED_CHUNK)
    ])
    # embeddings outlive the walk: registry dedup against re-walks, leak checks
    # and future selection all reuse them. Cosines are only comparable between
    # walks embedded by the same model, so the model name travels with the rows.
    np.savez(out / "embeddings.npz",
             embeddings=embeddings, model=params.embed.model_name)
    return embeddings


def manifest_rows(out: Path) -> list[dict]:
    with open(out / "manifest.csv") as f:
        return list(csv.DictReader(f))


def sharp_panos(out: Path) -> list:
    """The panoramas the gate kept last time, read back rather than re-scored."""
    keep = {int(r["pano_idx"]) for r in manifest_rows(out)}
    return [p for p in extract.load_panos(out / "pano") if p.index in keep]


def face_rows(out: Path) -> list[FaceRow]:
    """The faces already on disk, in the capture order the embeddings are in."""
    faces_dir = out / "faces"
    return [(int(r["pano_idx"]), float(r["t_sec"]), int(r["yaw"]),
             faces_dir / r["path"]) for r in manifest_rows(out)]


def read_calibration(out: Path) -> dict | None:
    path = out / "calibration.json"
    return json.loads(path.read_text()) if path.exists() else None


RUNS_FILE = "runs.jsonl"
RUNS_KEPT = 20
ERROR_FILE = "error.json"


def save_failure(out: Path, stage: str, message: str, detail: str = "") -> None:
    """Record that a run broke, next to whatever it managed to produce.

    The runner holds jobs in memory, so without this a failed ingest leaves a
    directory and a multi-gigabyte video that nothing in the app can explain
    or remove once the server restarts.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / ERROR_FILE).write_text(json.dumps({
        "stage": stage, "message": message, "detail": detail,
        "at": datetime.now().isoformat(timespec="seconds")}, indent=1))


def clear_failure(out: Path) -> None:
    (out / ERROR_FILE).unlink(missing_ok=True)


def read_failure(out: Path) -> dict | None:
    path = out / ERROR_FILE
    return json.loads(path.read_text()) if path.exists() else None


def log_run(out: Path, params: PipelineParams, calib: dict | None, first: str,
            faces: int, anchors: int) -> None:
    """Append what this configuration produced.

    Every run already knows its backbone, its threshold, what identical frames
    scored under it and how much that absorbed. Without writing it down,
    comparing two backbones means keeping the numbers on paper: the walk on
    disk only ever shows the last one that ran.
    """
    ref = (calib or {}).get("reference", {})
    row = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "from": first,
        "backbone": params.embed.model_name,
        "tau": params.dedup.tau,
        "rule": params.dedup.rule,
        "reference": ref.get("median", 0),
        "pairs": ref.get("n", 0),
        "faces": faces,
        "anchors": anchors,
        "absorbed": faces - anchors,
    }
    with open(out / RUNS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_runs(out: Path, limit: int = RUNS_KEPT) -> list[dict]:
    """The most recent runs, newest first."""
    path = out / RUNS_FILE
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return rows[::-1][:limit]


def save_calibration(out: Path, calib: dict) -> None:
    (out / "calibration.json").write_text(json.dumps(calib, indent=1))


def requantile(out: Path, params: PipelineParams) -> dict:
    """Move the threshold's percentile without measuring anything again.

    The pairs and their cosines are already on disk; the percentile is a
    statistic over them, not a new measurement. So this costs a sort, and the
    knob can be explored instead of committed to.
    """
    calib = read_calibration(out)
    if calib is None:
        raise FileNotFoundError("this walk has no calibration record")
    calib = calibrate_mod.record(calib["pairs"], params.calib,
                                 calib.get("gapSeconds", 0.0))
    save_calibration(out, calib)
    return calib


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
    rows = manifest_rows(walk_out)
    if len(rows) != len(embeddings):
        raise ValueError(f"manifest has {len(rows)} rows but "
                         f"{len(embeddings)} embeddings")

    calib = read_calibration(walk_out)
    params = apply_rule(params, calib)
    result = greedy_dedup(embeddings, params.dedup)
    write_manifest(walk_out / "manifest.csv",
                   [(int(r["pano_idx"]), float(r["t_sec"]), int(r["yaw"]), r["path"])
                    for r in rows], result)
    save_params(walk_out, params)
    kept = len(result.kept)
    log_run(walk_out, params, calib, "select", len(rows), kept)
    return dict(faces=len(rows), anchors=kept, absorbed=len(rows) - kept,
                tau=params.dedup.tau, rule=params.dedup.rule)
