"""One .insv walk -> deduplicated label candidates.

run_walk() writes out/<walk>/pano/, out/<walk>/faces/, manifest.csv and
embeddings.npz (one row per manifest row). rerun() enters the same chain at
a later stage and reads the stages above it back from disk, so stitch, gate
and faces each have a read-back counterpart.
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
from ..core.dedup import drop_solo, greedy_dedup
from ..core.embed import Embedder
from ..core.export import review_counts
from ..core.params import (BACKBONES, SEGMENTERS, SITE_CLASSES, PipelineParams,
                           from_dict)

EMBED_CHUNK = 64  # faces held in memory at once on the way to the embedder
MANIFEST_COLUMNS = ["pano_idx", "t_sec", "yaw", "path", "kept", "anchor",
                    "cosine", "sharpness"]
STAGES = ("stitch", "gate", "faces", "embed", "calibrate", "select", "segment")

# (pano index, t, yaw, file)
FaceRow = tuple[int, float, int, Path]


def face_name(pano_idx: int, yaw: int) -> str:
    return f"y{yaw:03d}_{pano_idx:05d}.jpg"


class Clock:
    """Per-stage wall time, wrapped around the progress callback."""

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
    """progress(stage, status, **counts) is called around every stage."""
    say = Clock(progress or (lambda *a, **k: None))
    out = out_root / video.stem
    out.mkdir(parents=True, exist_ok=True)

    native_fps, frames, width, height = extract.probe(video)
    files = extract.lens_files(video)
    atomic.write_json(
        out / "source.json",
        {"frames": frames, "fps": native_fps,
         "seconds": round(frames / native_fps, 1),
         "files": [p.name for p in files], "width": width, "height": height,
         "lenses": extract.lenses_in_frame(width, height) * len(files)})
    say("video", "done", frames=frames, seconds=round(frames / native_fps))

    say("stitch", "running")
    panos = extract.stitch(video, out / "pano", params.extract)
    say("stitch", "done", panos=len(panos))

    return _tail(video, out, params, embedder, native_fps, "gate", say,
                 panos=panos, segmenter=segmenter)


def rerun(video: Path | None, out: Path, params: PipelineParams,
          embedder: Embedder, first: str, progress=None, segmenter=None) -> dict:
    """Re-run from `first` down; the stages above are reported done.

    `video` is only needed when `first` is at or before the calibrate stage.
    """
    say = Clock(progress or (lambda *a, **k: None))
    if first not in STAGES:
        raise ValueError(f"unknown stage {first!r}")
    if first == "stitch":
        raise ValueError("re-stitching is not wired up yet; re-ingest the walk")
    if video is None and STAGES.index(first) <= STAGES.index("calibrate"):
        raise ValueError(f"re-running from {first} needs the original video")
    source = read_json(out / "source.json")
    if not isinstance(source, dict) or "fps" not in source:
        raise RuntimeError(f"{out.name} has no readable source.json; "
                           "re-ingest the walk")
    say("video", "done")
    for stage in STAGES[:STAGES.index(first)]:
        say(stage, "done")
    return _tail(video, out, params, embedder, source["fps"], first, say,
                 segmenter=segmenter)


def _tail(video: Path | None, out: Path, params: PipelineParams,
          embedder: Embedder, native_fps: float, first: str, say, panos=None,
          segmenter=None) -> dict:
    start = STAGES.index(first)

    if start <= STAGES.index("faces"):
        if start <= STAGES.index("gate"):
            say("gate", "running")
            sharp = gate(panos if panos is not None
                         else extract.load_panos(out / "pano"), params)
            say("gate", "done", sharp=len(sharp))
        else:
            sharp = sharp_panos(out)
        say("faces", "running")
        records, sharpness = render_faces(out, sharp, params)
        say("faces", "done", faces=len(records))
    else:
        records, sharpness = face_rows(out)
    if not records:
        raise RuntimeError(f"{out.name} has no faces to work with; nothing "
                           "survived the gate, or the manifest is empty")

    if start <= STAGES.index("embed"):
        say("embed", "running")
        embeddings = embed_faces(out, records, params, embedder)
        say("embed", "done")
    else:
        with np.load(out / "embeddings.npz") as data:
            embeddings = data["embeddings"]

    named = [(i, t, yaw, path.name) for i, t, yaw, path in records]
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
    result, solo = select(embeddings, named, sharpness, params)
    write_manifest(out / "manifest.csv", named, result, sharpness)
    save_params(out, params)
    kept = len(result.kept)
    say("select", "done", anchors=kept, absorbed=len(records) - kept, solo=solo)

    # annotation only: runs after the set is decided and never changes it
    if params.segment.enabled:
        say("segment", "running")
        named_kept = [named[i][3] for i in result.kept]
        classes = run_segment(out, named_kept, params, segmenter)
        say("segment", "done", segmented=len(classes))
    else:
        clear_segmentation(out)

    log_run(out, params, calib, first, len(records), kept, solo, say.seconds)
    clear_failure(out)
    return dict(walk=out.name, faces=len(records), anchors=kept,
                absorbed=len(records) - kept, tau=params.dedup.tau,
                rule=params.dedup.rule)


def imread(path: Path):
    """cv2.imread returns None for an undecodable file; name it instead."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"{path.name} is not a readable image; "
                           "re-run from the stitch to write it again")
    return img


def gate(panos: list, params: PipelineParams) -> list:
    """Drop only panoramas a labeller could not work with."""
    g = params.gate
    scores = [blur.score_pano(imread(p.path), g) for p in panos]
    sharp = [p for p, keep in zip(panos, blur.legible(scores, g.dead)) if keep]
    if not sharp:
        raise RuntimeError(
            f"the gate kept none of {len(panos)} panoramas"
            if panos else "the stitch produced no panoramas")
    return sharp


def render_faces(out: Path, sharp: list, params: PipelineParams
                 ) -> tuple[list[FaceRow], list[float]]:
    """Project every sharp panorama into its faces, replacing the earlier set.

    A yaw change renames every face; stale files would land in the export
    with no manifest row.
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
            # per face, not per panorama: a head turn smears one face and not the next
            sharpness.append(blur.vol_score(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    return records, sharpness


def embed_faces(out: Path, records: list[FaceRow], params: PipelineParams,
                embedder: Embedder) -> np.ndarray:
    embeddings = np.concatenate([
        embedder.embed([cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)
                        for _, _, _, path in chunk])
        for chunk in batched(records, EMBED_CHUNK)
    ])
    # cosines only compare across walks embedded by the same model
    atomic.atomically(out / "embeddings.npz", lambda tmp: np.savez(
        tmp, embeddings=embeddings, model=params.embed.model_name))
    return embeddings


def manifest_rows(out: Path) -> list[dict]:
    with open(out / "manifest.csv") as f:
        return list(csv.DictReader(f))


def sharp_panos(out: Path) -> list:
    """The panoramas the last gate kept, read back from the manifest."""
    keep = {int(r["pano_idx"]) for r in manifest_rows(out)}
    return [p for p in extract.load_panos(out / "pano") if p.index in keep]


def face_rows(out: Path) -> tuple[list[FaceRow], list[float]]:
    """The faces on disk in manifest order, with their recorded sharpness."""
    faces_dir = out / "faces"
    rows = manifest_rows(out)
    return ([(int(r["pano_idx"]), float(r["t_sec"]), int(r["yaw"]),
              faces_dir / r["path"]) for r in rows],
            [float(r["sharpness"]) for r in rows])


def read_json(path: Path) -> dict | list | None:
    """None when the file is missing, unreadable or not JSON."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_calibration(out: Path) -> dict | None:
    calib = read_json(out / "calibration.json")
    return calib if isinstance(calib, dict) else None


SEGMENT_FILE = "segmentation.json"


def run_segment(out: Path, names: list[str], params: PipelineParams,
                segmenter) -> dict[str, dict[str, float]]:
    """Mask every kept face; masks hold class indices, not colours."""
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
            seen.update({str(i): labels[i] for i in np.unique(seg).tolist()
                         if i in labels})

    atomic.write_json(out / SEGMENT_FILE,
                      {"model": params.segment.model_name, "labels": seen,
                       "classes": classes}, indent=1)
    return classes


def clear_segmentation(out: Path) -> None:
    shutil.rmtree(out / segment_mod.MASK_DIR, ignore_errors=True)
    (out / SEGMENT_FILE).unlink(missing_ok=True)


def read_segmentation(out: Path) -> dict:
    seg = read_json(out / SEGMENT_FILE)
    return seg if isinstance(seg, dict) else {}


RUNS_FILE = "runs.jsonl"
RUNS_KEPT = 20
ERROR_FILE = "error.json"
JOB_FILE = "job.json"


def save_job(out: Path, first: str | None) -> None:
    """Sentinel written at submit and removed when the run resolves.

    Jobs live in the runner's memory; this is what recover() finds after a
    process death, for a queued job with no directory yet or a re-run killed
    mid-stage behind a stale manifest.
    """
    out.mkdir(parents=True, exist_ok=True)
    atomic.write_json(out / JOB_FILE, {
        "stage": first or "",
        "at": datetime.now().isoformat(timespec="seconds")}, indent=1)


def clear_job(out: Path) -> None:
    (out / JOB_FILE).unlink(missing_ok=True)


def read_job(out: Path) -> dict | None:
    job = read_json(out / JOB_FILE)
    return job if isinstance(job, dict) else None


def save_failure(out: Path, stage: str, message: str, detail: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    atomic.write_json(out / ERROR_FILE, {
        "stage": stage, "message": message, "detail": detail,
        "at": datetime.now().isoformat(timespec="seconds")}, indent=1)


def clear_failure(out: Path) -> None:
    (out / ERROR_FILE).unlink(missing_ok=True)


def read_failure(out: Path) -> dict | None:
    failure = read_json(out / ERROR_FILE)
    return failure if isinstance(failure, dict) else None


def log_run(out: Path, params: PipelineParams, calib: dict | None, first: str,
            faces: int, anchors: int, solo: int = 0,
            seconds: dict[str, float] | None = None) -> None:
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
        "farPairs": far.get("n", 0),
        "farP99": far.get("p99", 0),
        "falseMergePct": params.calib.false_merge_pct,
        "faces": faces,
        "anchors": anchors,
        "absorbed": faces - anchors,
        "solo": solo,
        "seconds": seconds or {},
    }
    path = out / RUNS_FILE
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [ln for ln in lines if ln] + [json.dumps(row)]
    atomic.write_text(path, "\n".join(lines[-RUNS_KEPT:]) + "\n")


def read_runs(out: Path, limit: int = RUNS_KEPT) -> list[dict]:
    """Newest first; undecodable lines are skipped."""
    path = out / RUNS_FILE
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[::-1][:limit]


def save_calibration(out: Path, calib: dict) -> None:
    atomic.write_json(out / "calibration.json", calib, indent=1)


def can_requantile(out: Path) -> bool:
    """True if the record stores the far cosines needed to re-derive tau."""
    calib = read_calibration(out)
    return bool(calib and calib.get("far", {}).get("cosines"))


def requantile(out: Path, params: PipelineParams) -> dict:
    """Re-derive tau from the stored far cosines under a new false-merge budget.

    Changing far_seconds is not covered: that redraws the pair set and needs
    the embeddings.
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
    """Under the calibrated rule, tau is the measured threshold."""
    if params.dedup.rule != "calibrated":
        return params
    if not calib:
        raise ValueError("this walk has no calibration record to take a "
                         "threshold from. Use the fixed rule, or re-run from "
                         "calibrate.")
    tau = calib.get("tau")
    if tau is None:
        raise ValueError(
            f"this walk has {calib.get('far', {}).get('n', 0)} pairs more than "
            f"{calib.get('farSeconds', 0)}s apart, too few to set a threshold "
            f"from. Use the fixed rule, or a longer walk.")
    return replace(params, dedup=replace(params.dedup, tau=tau))


def write_manifest(path: Path, faces_out: list[tuple[int, float, int, str]],
                   result, sharpness: list[float]) -> None:
    """One row per face in capture order; faces_out is (pano_idx, t_sec, yaw, name).

    The anchor is the group's exported frame (dedup visits sharpest first);
    reviewer overrides live in the store, not here.
    """
    kept = set(result.kept)

    def rows(target: Path) -> None:
        with open(target, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(MANIFEST_COLUMNS)
            for i, (pano_idx, t_sec, yaw, name) in enumerate(faces_out):
                anchor, cos = ("", "")
                if i in result.anchor_of:
                    a, c = result.anchor_of[i]
                    anchor, cos = str(a), f"{c:.4f}"
                w.writerow([pano_idx, f"{t_sec:.1f}", yaw, name, int(i in kept),
                            anchor, cos, f"{sharpness[i]:.2f}"])

    # sharp_panos and face_rows rebuild the upstream stages from this file
    atomic.atomically(path, rows)


def save_params(out: Path, params: PipelineParams) -> None:
    atomic.write_json(out / "params.json", asdict(params), indent=1)


# params section -> the stage a change to it re-runs from
STAGE_OF = {"gate": "gate", "faces": "faces", "embed": "embed",
            "calib": "calibrate", "dedup": "select", "segment": "segment"}


def merge(params: PipelineParams, patch: dict[str, dict]
          ) -> tuple[PipelineParams, str | None, set[str]]:
    """Fold {section: {field: value}} into params.

    Returns (new params, earliest stage invalidated, "section.field" names
    that differ from the walk's).
    """
    changed: set[str] = set()
    for section, fields in patch.items():
        current = getattr(params, section)
        # JSON has no tuples; the params dataclasses do
        fields = {k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in fields.items() if v is not None}
        fields = {k: v for k, v in fields.items() if getattr(current, k) != v}
        if not fields:
            continue
        params = replace(params, **{section: replace(current, **fields)})
        changed.update(f"{section}.{k}" for k in fields)
    stages = {STAGE_OF[f.split(".")[0]] for f in changed}
    return params, min(stages, key=STAGES.index, default=None), changed


def walk_params(out: Path) -> PipelineParams:
    """The params the walk was built with; defaults if none were saved."""
    saved = read_json(out / "params.json")
    return from_dict(saved) if isinstance(saved, dict) else PipelineParams()


def embed_model_used(out: Path) -> str:
    npz = out / "embeddings.npz"
    if not npz.exists():
        return ""
    with np.load(npz) as data:
        return str(data["model"])


def pipeline_spec(out: Path) -> dict:
    """The walk's params plus the embedder recorded in the npz."""
    return dict(asdict(walk_params(out)), embed_model_used=embed_model_used(out),
                backbones=list(BACKBONES), segmenters=SEGMENTERS,
                site_classes=list(SITE_CLASSES))


def top_classes(classes: dict[str, dict], n: int = 6) -> list[dict]:
    """Each class's mean share over all segmented frames, largest first."""
    if not classes:
        return []
    totals: dict[str, float] = {}
    for shares in classes.values():
        for name, share in shares.items():
            totals[name] = totals.get(name, 0.0) + share
    frames = len(classes)
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])[:n]
    return [{"name": k, "share": round(v / frames, 3)} for k, v in ranked]


def stage_counts(out: Path, rows: list[dict], state: dict,
                 runs: list[dict]) -> dict:
    """Per-stage counts for the pipeline view; `runs` is read_runs(out)."""
    src = read_json(out / "source.json")
    src = src if isinstance(src, dict) else {}
    anchors = sum(1 for r in rows if r["kept"] == "1")
    dropped, overridden = review_counts(rows, state)
    calib = read_calibration(out) or {}
    seg_classes = read_segmentation(out).get("classes", {})
    return {
        "frames": src.get("frames", 0),
        "seconds": src.get("seconds", 0),
        "source": f"{src['width']}x{src['height']}" if "width" in src else "",
        "lenses": src.get("lenses", 0),
        "panos": len(list((out / "pano").glob(
            f"{extract.PANO_PREFIX}*{extract.PANO_EXT}"))),
        "sharp": len({r["pano_idx"] for r in rows}),
        "faces": len(rows),
        "pairs": calib.get("reference", {}).get("n", 0),
        "reference": calib.get("reference", {}).get("median", 0),
        "farPairs": calib.get("far", {}).get("n", 0),
        "calibTau": calib.get("tau") or 0,
        "segmented": len(seg_classes),
        "classMix": top_classes(seg_classes),
        "anchors": anchors,
        "absorbed": len(rows) - anchors,
        # a dropped solo leaves no manifest row, so this comes from the run record
        "solo": (runs or [{}])[0].get("solo", 0),
        "dropped": dropped,
        "overridden": overridden,
        "kept": anchors - dropped,
    }


def select(embeddings: np.ndarray, named: list, sharpness: list,
           params: PipelineParams) -> tuple:
    """Dedup plus the solo drop. Returns (result, groups of one dropped)."""
    scores = np.asarray(sharpness, dtype=float)
    result = greedy_dedup(embeddings, params.dedup, scores,
                          np.array([p for p, _, _, _ in named]))
    # a group of one has no sharper member to stand in for it
    ratio = blur.heading_ratio(scores,
                               np.array([y for _, _, y, _ in named]),
                               np.array([t for _, t, _, _ in named]),
                               params.dedup.solo_span)
    before = len(result.kept)
    result = drop_solo(result, ratio, params.dedup.solo_floor)
    return result, before - len(result.kept)


def reselect(walk_out: Path, params: PipelineParams) -> dict:
    """Re-run selection from the embeddings on disk.

    Face files keep their names, so review decisions survive. Masks belong
    to the previous kept set and are cleared.
    """
    with np.load(walk_out / "embeddings.npz") as data:
        embeddings = data["embeddings"]
    rows = manifest_rows(walk_out)
    if len(rows) != len(embeddings):
        raise ValueError(f"manifest has {len(rows)} rows but "
                         f"{len(embeddings)} embeddings")

    calib = read_calibration(walk_out)
    params = apply_rule(params, calib)
    named = [(int(r["pano_idx"]), float(r["t_sec"]), int(r["yaw"]), r["path"])
             for r in rows]
    sharpness = [float(r["sharpness"]) for r in rows]
    result, solo = select(embeddings, named, sharpness, params)
    write_manifest(walk_out / "manifest.csv", named, result, sharpness)
    save_params(walk_out, params)
    clear_segmentation(walk_out)
    kept = len(result.kept)
    log_run(walk_out, params, calib, "select", len(rows), kept, solo)
    return dict(faces=len(rows), anchors=kept, absorbed=len(rows) - kept,
                solo=solo, tau=params.dedup.tau, rule=params.dedup.rule)
