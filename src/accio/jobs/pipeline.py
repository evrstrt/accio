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
import time
from dataclasses import asdict, replace
from datetime import datetime
from itertools import batched
from pathlib import Path

import cv2
import numpy as np

from ..core import (atomic, blur, calibrate as calibrate_mod, extract, faces,
                    segment as segment_mod)
from ..core.calibrate import calibrate
from ..core.dedup import greedy_dedup, sharpest
from ..core.embed import Embedder
from ..core.params import PipelineParams

EMBED_CHUNK = 64  # faces held in memory at once on the way to the embedder
MANIFEST_COLUMNS = ["pano_idx", "t_sec", "yaw", "path", "kept", "anchor",
                    "cosine", "sharpness", "pick"]
STAGES = ("stitch", "gate", "faces", "embed", "calibrate", "select", "segment")

# a face on its way through the pipeline: (pano index, t, yaw, file)
FaceRow = tuple[int, float, int, Path]


def face_name(pano_idx: int, yaw: int) -> str:
    return f"y{yaw:03d}_{pano_idx:05d}.jpg"


class Clock:
    """Wall clock per stage, wrapped around the progress callback.

    Nothing recorded timing before this, so every capacity question had to be
    answered by reading file mtimes off disk and guessing which write belonged
    to which stage. The run record already says what a configuration produced;
    what it cost is the other half, and it is the half that decides whether a
    backbone or a segmenter is affordable on a real walk.
    """

    def __init__(self, say):
        self._say = say
        self.seconds: dict[str, float] = {}
        self._started: dict[str, float] = {}

    def __call__(self, stage: str, status: str, **counts) -> None:
        now = time.monotonic()
        if status == "running":
            self._started[stage] = now
        elif status == "done" and stage in self._started:
            self.seconds[stage] = round(now - self._started.pop(stage), 1)
        self._say(stage, status, **counts)


def run_walk(video: Path, out_root: Path, params: PipelineParams,
             embedder: Embedder, progress=None, segmenter=None) -> dict:
    """progress(stage, status, **counts) is called around every stage, so the
    caller can show the run advancing instead of a single opaque wait."""
    say = Clock(progress or (lambda *a, **k: None))
    out = out_root / video.stem
    out.mkdir(parents=True, exist_ok=True)

    # what the camera actually recorded, kept next to the derived frames: it is
    # the denominator the rest of the funnel is a fraction of
    native_fps, frames, width, height = extract.probe(video)
    files = extract.lens_files(video)
    atomic.write_json(
        out / "source.json",
        {"frames": frames, "fps": native_fps,
         "seconds": round(frames / native_fps, 1),
         "files": [p.name for p in files], "width": width, "height": height,
         # both packings are dual-fisheye; the count says whether this walk
         # actually had both circles to stitch from
         "lenses": extract.lenses_in_frame(width, height) * len(files)})
    say("video", "done", frames=frames, seconds=round(frames / native_fps))

    say("stitch", "running")
    panos = extract.stitch(video, out / "pano", params.extract)
    say("stitch", "done", panos=len(panos))

    return _tail(video, out, params, embedder, native_fps, "gate", say,
                 panos=panos, segmenter=segmenter)


def rerun(video: Path, out: Path, params: PipelineParams, embedder: Embedder,
          first: str, progress=None, segmenter=None) -> dict:
    """Re-run from `first` down, reusing the stages above it as they are.

    The stages above are reported done rather than skipped silently, so the
    canvas shows the same blocks either way and the ones that were reused are
    visibly not running.
    """
    say = Clock(progress or (lambda *a, **k: None))
    if first not in STAGES:
        raise ValueError(f"unknown stage {first!r}")
    if first == "stitch":
        raise ValueError("re-stitching is not wired up yet; re-ingest the walk")
    native_fps = json.loads((out / "source.json").read_text())["fps"]
    say("video", "done")
    for stage in STAGES[:STAGES.index(first)]:
        say(stage, "done")
    return _tail(video, out, params, embedder, native_fps, first, say,
                 segmenter=segmenter)


def _tail(video: Path, out: Path, params: PipelineParams, embedder: Embedder,
          native_fps: float, first: str, say, panos=None, segmenter=None) -> dict:
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
        records, sharpness = render_faces(out, sharp, params)
        say("faces", "done", faces=len(records))
    else:
        records, sharpness = face_rows(out)

    if start <= STAGES.index("embed"):
        say("embed", "running")
        embeddings = embed_faces(out, records, params, embedder)
        say("embed", "done")
    else:
        with np.load(out / "embeddings.npz") as data:
            embeddings = data["embeddings"]

    named = [(i, t, yaw, path.name) for i, t, yaw, path in records]
    # what elsewhere scores on this walk, which is what the threshold has to
    # stay above, plus the identical-content reference as a health check: a low
    # one means the stitch, the exposure or the backbone is misbehaving.
    if start <= STAGES.index("calibrate"):
        say("calibrate", "running")
        calib = calibrate(video, out, named, embeddings, params, embedder,
                          native_fps)
        save_calibration(out, calib)
        say("calibrate", "done", pairs=calib["reference"]["n"],
            reference=calib["reference"]["median"],
            farPairs=calib["far"]["n"], calibTau=calib["tau"] or 0)
    else:
        calib = read_calibration(out)

    say("select", "running")
    params = apply_rule(params, calib)
    result = greedy_dedup(embeddings, params.dedup,
                          np.array([p for p, _, _, _ in named]))
    picks = sharpest(result, np.asarray(sharpness, dtype=float))
    write_manifest(out / "manifest.csv", named, result, sharpness, picks)
    save_params(out, params)
    kept = len(result.kept)
    say("select", "done", anchors=kept, absorbed=len(records) - kept)

    # what is in the frames that survived. Annotation only: it runs after the
    # set is decided and never changes it, so a wrong mask costs a correction
    # rather than a candidate.
    #
    # On the picks, not the anchors: those are the frames the export ships, and
    # they differ from their anchor in about half of all groups now that the
    # sharpest member wins. Segmenting anchors would leave most exported frames
    # with no mask and a blank class row, and build_zip's `if mask.exists()`
    # would swallow it.
    if params.segment.enabled:
        say("segment", "running")
        named_kept = [named[picks[i]][3] for i in result.kept]
        classes = run_segment(out, named_kept, params, segmenter)
        say("segment", "done", segmented=len(classes))
    elif start <= STAGES.index("segment"):
        clear_segmentation(out)

    log_run(out, params, calib, first, len(records), kept,
            getattr(say, "seconds", {}))
    clear_failure(out)     # this walk ran through; whatever broke before is past
    return dict(walk=out.name, faces=len(records), anchors=kept,
                absorbed=len(records) - kept, tau=params.dedup.tau,
                rule=params.dedup.rule)


def imread(path: Path):
    """cv2.imread answers None for anything it cannot decode, and every caller
    here immediately takes .shape of it. Naming the file that failed beats an
    AttributeError attributed to whichever stage happened to touch it first."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"{path.name} is not a readable image; "
                           "re-run from the stitch to write it again")
    return img


def gate(panos: list, params: PipelineParams) -> list:
    """Every panorama a labeller could work with, which is nearly all of them.

    This used to keep one panorama in four and call it a blur gate. What
    survives here is the walk; what ships is decided by Select.
    """
    g = params.gate
    scores = [blur.score_pano(imread(p.path), g) for p in panos]
    ok = blur.legible(scores, g.window, g.floor, g.dead)
    return [p for p, keep in zip(panos, ok) if keep]


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
    sharpness: list[float] = []
    for p in sharp:
        pano_img = imread(p.path)
        for yaw, img in faces.render_faces(pano_img, params.faces).items():
            path = faces_dir / face_name(p.index, yaw)
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append((p.index, p.t_sec, yaw, path))
            # scored here because the face is already in memory, and per face
            # rather than per panorama: an operator turning their head smears
            # the leading face while the trailing one stays sharp, and one
            # score for the whole sphere cannot see that
            sharpness.append(blur.vol_score(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    return records, sharpness


def embed_faces(out: Path, records: list[FaceRow], params: PipelineParams,
                embedder: Embedder) -> np.ndarray:
    embeddings = np.concatenate([
        embedder.embed([cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)
                        for _, _, _, path in chunk])
        for chunk in batched(records, EMBED_CHUNK)
    ])
    # embeddings outlive the walk: registry dedup against re-walks, leak checks
    # and future selection all reuse them. Cosines are only comparable between
    # walks embedded by the same model, so the model name travels with the rows.
    atomic.atomically(out / "embeddings.npz", lambda tmp: np.savez(
        tmp, embeddings=embeddings, model=params.embed.model_name))
    return embeddings


def manifest_rows(out: Path) -> list[dict]:
    with open(out / "manifest.csv") as f:
        return list(csv.DictReader(f))


def sharp_panos(out: Path) -> list:
    """The panoramas the gate kept last time, read back rather than re-scored."""
    keep = {int(r["pano_idx"]) for r in manifest_rows(out)}
    return [p for p in extract.load_panos(out / "pano") if p.index in keep]


def face_rows(out: Path) -> tuple[list[FaceRow], list[float]]:
    """The faces already on disk, in the capture order the embeddings are in,
    with the sharpness scored when they were rendered."""
    faces_dir = out / "faces"
    rows = manifest_rows(out)
    return ([(int(r["pano_idx"]), float(r["t_sec"]), int(r["yaw"]),
              faces_dir / r["path"]) for r in rows],
            [float(r["sharpness"]) for r in rows])


def read_calibration(out: Path) -> dict | None:
    path = out / "calibration.json"
    return json.loads(path.read_text()) if path.exists() else None


SEGMENT_FILE = "segmentation.json"


def run_segment(out: Path, names: list[str], params: PipelineParams,
                segmenter) -> dict[str, dict[str, float]]:
    """Mask every kept face and record what each one is made of.

    The mask is written as class indices, not colour, so the file is the label
    and a palette stays a rendering choice. Only the kept faces are done: it is
    the export that gets annotated, and inference is the expensive part.
    """
    masks = out / segment_mod.MASK_DIR
    shutil.rmtree(masks, ignore_errors=True)
    masks.mkdir(parents=True, exist_ok=True)
    faces_dir = out / "faces"

    classes: dict[str, dict[str, float]] = {}
    seen: dict[str, str] = {}
    for chunk in batched(names, EMBED_CHUNK):
        images = [cv2.cvtColor(imread(faces_dir / n), cv2.COLOR_BGR2RGB)
                  for n in chunk]
        segs, labels = segmenter.segment(images)
        for name, seg in zip(chunk, segs):
            segment_mod.write_mask(masks / f"{Path(name).stem}.png", seg)
            classes[name] = segment_mod.shares(seg, labels)
            # the mask holds indices, so whatever reads it needs their names;
            # only the ones that actually turned up are worth carrying
            seen.update({str(i): labels[i] for i in np.unique(seg).tolist()
                         if i in labels})

    atomic.write_json(out / SEGMENT_FILE,
                      {"model": params.segment.model_name, "labels": seen,
                       "classes": classes}, indent=1)
    return classes


def clear_segmentation(out: Path) -> None:
    """Turning the stage off takes its output with it, or the export would
    carry masks of a frame set that no longer exists."""
    shutil.rmtree(out / segment_mod.MASK_DIR, ignore_errors=True)
    (out / SEGMENT_FILE).unlink(missing_ok=True)


def read_segmentation(out: Path) -> dict:
    path = out / SEGMENT_FILE
    return json.loads(path.read_text()) if path.exists() else {}


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
    atomic.write_json(out / ERROR_FILE, {
        "stage": stage, "message": message, "detail": detail,
        "at": datetime.now().isoformat(timespec="seconds")}, indent=1)


def clear_failure(out: Path) -> None:
    (out / ERROR_FILE).unlink(missing_ok=True)


def read_failure(out: Path) -> dict | None:
    path = out / ERROR_FILE
    return json.loads(path.read_text()) if path.exists() else None


def log_run(out: Path, params: PipelineParams, calib: dict | None, first: str,
            faces: int, anchors: int,
            seconds: dict[str, float] | None = None) -> None:
    """Append what this configuration produced.

    Every run already knows its backbone, its threshold, what identical frames
    scored under it and how much that absorbed. Without writing it down,
    comparing two backbones means keeping the numbers on paper: the walk on
    disk only ever shows the last one that ran.
    """
    ref = (calib or {}).get("reference", {})
    far = (calib or {}).get("far", {})
    row = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "from": first,
        "backbone": params.embed.model_name,
        "tau": params.dedup.tau,
        "rule": params.dedup.rule,
        "reference": ref.get("median", 0),
        "pairs": ref.get("n", 0),
        # what the threshold was actually placed against, and the risk it was
        # placed at: without these a row cannot say why tau was that number
        "farPairs": far.get("n", 0),
        "farP99": far.get("p99", 0),
        "falseMergePct": params.calib.false_merge_pct,
        "faces": faces,
        "anchors": anchors,
        "absorbed": faces - anchors,
        # what it cost, per stage: the other half of what a run produced
        "seconds": seconds or {},
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
    atomic.write_json(out / "calibration.json", calib, indent=1)


def can_requantile(out: Path) -> bool:
    """Whether the budget can be moved by re-reading rather than re-measuring.

    It needs the far cosines the record stores. A walk whose calibration was
    written without them has nothing to re-read, and re-resolving from an
    empty array would quietly answer "no threshold" for a walk that has one.
    """
    calib = read_calibration(out)
    return bool(calib and calib.get("far", {}).get("cosines"))


def requantile(out: Path, params: PipelineParams) -> dict:
    """Move the false-merge budget without measuring anything again.

    Both distributions are already on disk; the budget is a percentile over
    the far cosines, not a new measurement. So this costs a sort, and the knob
    can be explored instead of committed to. Changing far_seconds is not this:
    that redraws which pairs count as elsewhere, and needs the embeddings.
    """
    calib = read_calibration(out)
    if calib is None:
        raise FileNotFoundError("this walk has no calibration record")
    calib = calibrate_mod.record(calib["pairs"],
                                 np.array(calib.get("far", {}).get("cosines", [])),
                                 params.calib, calib.get("gapSeconds", 0.0),
                                 calib.get("referenceError", ""))
    save_calibration(out, calib)
    return calib


def apply_rule(params: PipelineParams, calib: dict | None) -> PipelineParams:
    """Under the calibrated rule the threshold comes from the measurement, so
    tau always holds the number that actually ran.

    A walk too short to have a far distribution has no measured threshold, and
    saying so beats writing the default in and calling it calibrated.
    """
    if params.dedup.rule != "calibrated" or not calib:
        return params
    tau = calib.get("tau")
    if tau is None:
        raise ValueError(
            f"this walk has {calib.get('far', {}).get('n', 0)} pairs more than "
            f"{calib.get('farSeconds', 0)}s apart, too few to set a threshold "
            f"from. Use the fixed rule, or a longer walk.")
    return replace(params, dedup=replace(params.dedup, tau=tau))


def write_manifest(path: Path, faces_out: list[tuple[int, float, int, str]],
                   result, sharpness: list[float],
                   picks: dict[int, int] | None = None) -> None:
    """One row per face in capture order, with the group it landed in.

    faces_out is (pano_idx, t_sec, yaw, file name); the dedup result supplies
    kept / anchor / cosine. Shared by the full run and by a re-select, so the
    two can never disagree about the manifest's shape.

    `pick` is only on anchor rows: the member this group exports, which is the
    sharpest of it rather than whichever arrived first. A reviewer can still
    override it, and that override is stored separately, so this column stays
    the machine's answer and never records a human's.
    """
    kept = set(result.kept)
    if picks is None:
        picks = sharpest(result, np.asarray(sharpness, dtype=float))

    def rows(target: Path) -> None:
        with open(target, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(MANIFEST_COLUMNS)
            for i, (pano_idx, t_sec, yaw, name) in enumerate(faces_out):
                anchor, cos = ("", "")
                if i in result.anchor_of:
                    a, c = result.anchor_of[i]
                    anchor, cos = str(a), f"{c:.4f}"
                pick = faces_out[picks[i]][3] if i in picks else ""
                w.writerow([pano_idx, f"{t_sec:.1f}", yaw, name, int(i in kept),
                            anchor, cos, f"{sharpness[i]:.2f}", pick])

    # this file is the walk: sharp_panos and face_rows both rebuild the stages
    # above from it, so a half-written one does not lose the selection, it
    # loses everything the stitch paid for
    atomic.atomically(path, rows)


def save_params(out: Path, params: PipelineParams) -> None:
    """The settings this walk's frames were built with, next to the frames."""
    atomic.write_json(out / "params.json", asdict(params), indent=1)


def reselect(walk_out: Path, params: PipelineParams) -> dict:
    """Re-run selection alone, from the embeddings already on disk.

    Everything upstream of Select is unchanged by a threshold, so this costs
    a dedup pass and a manifest rewrite rather than a stitch and a GPU pass.
    Face files and their names are untouched, which is what lets review
    decisions (keyed by name) survive the change.

    Downstream is a different matter: the masks belong to the frames the last
    run picked, and this changes which those are. Leaving them made the record
    describe a set that no longer exists, so the masks page listed frames that
    were no longer kept and 404ed on every one, `segmented` counted frames with
    no mask behind it, and build_zip's `if mask.exists()` shipped a partly
    annotated dataset without saying so. Clearing is honest and re-running
    segment is cheap next to being wrong.
    """
    with np.load(walk_out / "embeddings.npz") as data:
        embeddings = data["embeddings"]
    rows = manifest_rows(walk_out)
    if len(rows) != len(embeddings):
        raise ValueError(f"manifest has {len(rows)} rows but "
                         f"{len(embeddings)} embeddings")

    calib = read_calibration(walk_out)
    params = apply_rule(params, calib)
    result = greedy_dedup(embeddings, params.dedup,
                          np.array([int(r["pano_idx"]) for r in rows]))
    write_manifest(walk_out / "manifest.csv",
                   [(int(r["pano_idx"]), float(r["t_sec"]), int(r["yaw"]), r["path"])
                    for r in rows], result,
                   [float(r["sharpness"]) for r in rows])
    save_params(walk_out, params)
    clear_segmentation(walk_out)
    kept = len(result.kept)
    log_run(walk_out, params, calib, "select", len(rows), kept)
    return dict(faces=len(rows), anchors=kept, absorbed=len(rows) - kept,
                tau=params.dedup.tau, rule=params.dedup.rule)
