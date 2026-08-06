"""FastAPI server: pipeline output (immutable) + review decisions (store).

No science here. Routes translate manifest.csv and the decisions store into
JSON and serve images; the pipeline stays runnable headless and the frontend
stays swappable.

Run:  uv run uvicorn accio.server.app:app --reload
The app owns one data root (default ./data, override with ACCIO_DATA):
videos/ holds the original .insv files (the raw ground truth, kept so walks
can be re-stitched when parameters improve), walks/<id>/ the derived frames,
accio.db the decisions and walk metadata.
"""

import csv
import json
import os
import shutil
import tempfile
import threading
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .. import settings
from ..core import atomic, export, extract
from ..core.params import (BACKBONES, SEGMENTERS, SITE_CLASSES,
                           PipelineParams)
from ..core.params import from_dict as params_from_dict
from ..jobs import pipeline
from ..jobs.runner import Runner
from ..store import db

DATA_ROOT = settings.DATA_ROOT
VIDEO_DIR = DATA_ROOT / "videos"
WALKS_ROOT = DATA_ROOT / "walks"
# Exports are assembled here rather than in /tmp: under the data root they sit
# on the volume that already holds the frames, so a hundreds-of-megabytes
# archive cannot fill a container's writable layer instead.
EXPORT_TMP = DATA_ROOT / ".exports"

app = FastAPI(title="accio")

_local = threading.local()
_runner: Runner | None = None
_runner_lock = threading.Lock()


def conn():
    if not hasattr(_local, "conn"):
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        _local.conn = db.connect(DATA_ROOT / "accio.db")
    return _local.conn


def runner() -> Runner:
    global _runner
    with _runner_lock:
        if _runner is None:
            reap_incoming()
            _runner = Runner(WALKS_ROOT)
    return _runner


def reap_incoming() -> None:
    """Nothing under .incoming survives a restart, by construction.

    An upload is staged there and moved out once it is known to be stitchable,
    with the directory removed in a finally. Process death skips the finally,
    and what it leaves is a partial multi-gigabyte file in a directory no route
    reads and no operator has reason to look in.
    """
    staging = VIDEO_DIR / ".incoming"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    # half-built exports are the same story: the background task that deletes
    # one runs after the response, which a killed process never sends
    shutil.rmtree(EXPORT_TMP, ignore_errors=True)


def walk_dir(walk_id: str) -> Path:
    """A walk is a directory under the root, whether or not it made frames.

    A run that broke leaves the directory and the video behind. Requiring a
    manifest here would make that walk invisible to the list and impossible to
    delete, which is how a failed ingest turns into a few hundred megabytes
    nothing in the app can account for.
    """
    d = (WALKS_ROOT / walk_id).resolve()
    if not d.is_relative_to(WALKS_ROOT) or not d.is_dir():
        raise HTTPException(404, f"unknown walk {walk_id!r}")
    return d


def finished(walk_id: str) -> Path:
    """A walk that got as far as a manifest; anything reading one needs this."""
    d = walk_dir(walk_id)
    if not (d / "manifest.csv").exists():
        raise HTTPException(409, f"{walk_id} has no frames; its run did not finish")
    return d


def read_manifest(walk_id: str) -> list[dict]:
    path = walk_dir(walk_id) / "manifest.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# walk id -> (manifest mtime, face count, anchor names)
_summary: dict[str, tuple[int, int, set[str]]] = {}


def walk_summary(walk_id: str) -> tuple[int, set[str]]:
    """How many faces a walk has and which of them are anchors.

    Cached on the manifest's mtime. The list endpoint is polled and touches
    every walk, so parsing every manifest each time made it cost every frame
    of every walk per request: benchmarked at 1.8 s of CPU at 500 walks, which
    several reviewers polling would never catch up with. The manifest is
    written atomically now, so its mtime changes exactly when the answer does
    and never mid-write.
    """
    path = walk_dir(walk_id) / "manifest.csv"
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        _summary.pop(walk_id, None)
        return 0, set()
    hit = _summary.get(walk_id)
    if hit and hit[0] == stamp:
        return hit[1], hit[2]
    rows = read_manifest(walk_id)
    anchors = {r["path"] for r in rows if r["kept"] == "1"}
    _summary[walk_id] = (stamp, len(rows), anchors)
    return len(rows), anchors


@app.get("/api/walks")
def walks() -> list[dict]:
    out = []
    for d in sorted(WALKS_ROOT.iterdir()) if WALKS_ROOT.exists() else []:
        if not d.is_dir():
            continue
        faces, anchors = walk_summary(d.name)
        state = db.effective_state(conn(), d.name)
        # decisions on faces that are no longer anchors stay in the log but do
        # not count against this run, same rule as export.review_counts
        dropped = sum(1 for a, s in state.items()
                      if a in anchors and s["dropped"])
        out.append({
            "id": d.name,
            "faces": faces,
            "kept": len(anchors) - dropped,
            "meta": db.walk_meta(conn(), d.name),
            # a walk with no manifest never finished: it is listed so it can be
            # seen and removed, not because there is anything to review
            "ready": bool(faces),
            "error": pipeline.read_failure(d),
        })
    return out


@app.post("/api/ingest")
# Deliberately `def`, not `async def`. The body copies gigabytes with
# shutil.copyfileobj and then shells out to ffprobe, none of which yields, so
# on the event loop it stopped the whole server for the length of an upload:
# no job polling, no images, a second operator's upload stalled behind it, and
# it read as a crash. FastAPI runs a plain def in the threadpool, where a
# blocking call only costs its own thread, and UploadFile works the same there.
def ingest(
    files: list[UploadFile] = File([]),
    site: str = Form(""),
    building: str = Form(""),
    floor: str = Form(""),
    stage: str = Form(""),
    operator: str = Form(""),
    mount_height_cm: int | None = Form(None),
    shot_date: str = Form(""),
    shot_time: str = Form(""),
) -> dict:
    # Upload is the only ingest: this is a web app, the server has no access
    # to client paths. Dual-file recordings (_00_ front + _10_ back) arrive
    # as two uploads; the app owns its copy of the originals.
    if not files or not files[0].filename:
        raise HTTPException(422, "upload the .insv file(s)")
    # The capture metadata is the whole point of ingesting through the app
    # rather than running the pipeline by hand: a frame with no site on it is
    # not a labelling candidate, it is an orphan. The date and time are absent
    # from this list because the file answers them.
    need = {"site": site, "building": building, "floor": floor, "stage": stage,
            "operator": operator,
            "mount height": "" if mount_height_cm is None else "set"}
    blank = [k for k, v in need.items() if not str(v).strip()]
    if blank:
        raise HTTPException(422, f"still needed: {', '.join(blank)}")
    # a _10_ on its own is the back lens of a dual-file recording: the SDK
    # stitches nothing from it, so say so now rather than after the upload
    names = [Path(f.filename or "").name for f in files]
    if any("_10_" in n for n in names) and not any("_00_" in n for n in names):
        raise HTTPException(422, "that is the back lens only; upload the _00_ "
                                 "file alongside it")
    # Land the upload beside the store, not in it, and only move it across
    # once it is known to be stitchable. Writing first and cleaning up after
    # would delete whatever it overwrote on the way in.
    staging = VIDEO_DIR / ".incoming"
    staging.mkdir(parents=True, exist_ok=True)
    hold = Path(tempfile.mkdtemp(dir=staging))
    try:
        saved = []
        for up in files:
            dst = hold / Path(up.filename).name
            with open(dst, "wb") as f:
                shutil.copyfileobj(up.file, f)
            saved.append(dst)
        video = extract.front_lens(saved)

        # The other half of the lone-_10_ mistake: a square frame is one
        # fisheye circle, so a _00_ on its own is half a sphere. Stitching it
        # anyway smears the front hemisphere across the back, which reads as a
        # bad stitch rather than the missing file it is.
        _fps, _n, width, height = extract.probe(video)
        if extract.lenses_in_frame(width, height) == 1 and len(saved) < 2:
            raise HTTPException(422, f"{video.name} is {width}x{height}: one lens "
                                     "of a two-file recording. Upload its _10_ "
                                     "file alongside it.")
        # the walk id is the file name, so a second upload of it would take
        # over the first walk's directory, its original, and its review
        # decisions, which are keyed by face name and would silently re-attach
        # to whatever frame now carries it
        if (WALKS_ROOT / video.stem).exists():
            raise HTTPException(409, f"{video.stem} is already a walk. Delete it "
                                     "first if you mean to replace it.")
        video = shutil.move(str(video), VIDEO_DIR / video.name)
        for p in saved:
            if p.exists():
                shutil.move(str(p), VIDEO_DIR / p.name)
        video = Path(video)
    finally:
        shutil.rmtree(hold, ignore_errors=True)

    # The camera and the moment are in the file; only the operator knows the
    # site, so only that is asked for. A form value still wins, since a walk
    # can be uploaded long after it was shot from a camera clock nobody set.
    cam = extract.camera(video)
    stamp = extract.recorded_at(video)
    db.save_walk_meta(conn(), video.stem, video.name, site=site,
                      building=building, floor=floor, stage=stage,
                      operator=operator, mount_height_cm=mount_height_cm,
                      shot_date=shot_date or stamp[:10],
                      shot_time=shot_time or stamp[11:16],
                      camera=cam.get("model", ""))
    job = runner().submit(video)
    return job.public()


@app.get("/api/jobs")
def jobs() -> list[dict]:
    return runner().list() if _runner is not None else []


@app.get("/api/walks/{walk_id}")
def walk_detail(walk_id: str) -> dict:
    rows = read_manifest(walk_id)
    state = db.effective_state(conn(), walk_id)
    faces = [{
        "idx": i,
        "panoIdx": int(r["pano_idx"]),
        "tSec": float(r["t_sec"]),
        "yaw": int(r["yaw"]),
        "url": f"/api/walks/{walk_id}/faces/{r['path']}",
        "kept": r["kept"] == "1",
        "anchor": int(r["anchor"]) if r["anchor"] else None,
        "cosine": float(r["cosine"]) if r["cosine"] else None,
        "sharpness": float(r["sharpness"]),
    } for i, r in enumerate(rows)]
    # decisions are stored by face name; the API speaks manifest indices, so
    # translate at the boundary and the stored log survives a re-run
    idx_of = {r["path"]: i for i, r in enumerate(rows)}
    groups = []
    for f in faces:
        if not f["kept"]:
            continue
        s = state.get(rows[f["idx"]]["path"], {"pick": None, "dropped": False})
        pick = idx_of.get(s["pick"]) if s["pick"] else None
        # the machine's choice is the sharpest member; the human's overrides it
        auto = idx_of.get(rows[f["idx"]]["pick"], f["idx"])
        groups.append({
            "anchor": f,
            "members": sorted((m for m in faces if m["anchor"] == f["idx"]),
                              key=lambda m: -m["cosine"]),
            "auto": auto,
            "pick": pick if pick is not None else auto,
            "dropped": s["dropped"],
        })
    return {"id": walk_id, "faces": len(faces), "groups": groups,
            "meta": db.walk_meta(conn(), walk_id),
            "stages": stage_counts(walk_id, rows, state),
            "pipeline": pipeline_spec(walk_id),
            "runs": pipeline.read_runs(walk_dir(walk_id)),
            # a walk whose run broke still opens: the canvas shows where
            "error": pipeline.read_failure(walk_dir(walk_id))}


def stage_counts(walk_id: str, rows: list[dict], state: dict) -> dict:
    """What each stage emitted, for the pipeline view. The pre-gate pano count
    comes from the stitched files on disk; everything else from the manifest."""
    wdir = walk_dir(walk_id)
    pano_dir = wdir / "pano"
    src_file = wdir / "source.json"
    src = json.loads(src_file.read_text()) if src_file.exists() else {}
    anchors = sum(1 for r in rows if r["kept"] == "1")
    dropped, overridden = export.review_counts(rows, state)
    calib = pipeline.read_calibration(wdir) or {}
    seg = pipeline.read_segmentation(wdir)
    seg_classes = seg.get("classes", {})
    return {
        "frames": src.get("frames", 0),
        "seconds": src.get("seconds", 0),
        "source": f"{src['width']}x{src['height']}" if "width" in src else "",
        "lenses": src.get("lenses", 0),
        "panos": len(list(pano_dir.glob(f"{extract.PANO_PREFIX}*{extract.PANO_EXT}"))),
        "sharp": len({r["pano_idx"] for r in rows}),
        "faces": len(rows),
        # the calibration stage's own output: how alike identical frames were
        # (the health check), how many far-apart pairs the threshold was placed
        # against, and what it resolved to
        "pairs": calib.get("reference", {}).get("n", 0),
        "reference": calib.get("reference", {}).get("median", 0),
        "farPairs": calib.get("far", {}).get("n", 0),
        "calibTau": calib.get("tau") or 0,
        "segmented": len(seg_classes),
        # the classes this walk is made of, largest share first
        "classMix": top_classes(seg_classes),
        "anchors": anchors,
        "absorbed": len(rows) - anchors,
        # groups of one the last run refused as unusable. Read from the run
        # record rather than the manifest: a dropped solo leaves no row behind,
        # so the manifest cannot say it ever existed.
        "solo": (pipeline.read_runs(walk_dir(walk_id), 1) or [{}])[0].get("solo", 0),
        "dropped": dropped,
        "overridden": overridden,
        "kept": anchors - dropped,
    }


def top_classes(classes: dict[str, dict], n: int = 6) -> list[dict]:
    """What the walk as a whole is made of: each class's mean share across the
    frames it appears in, weighted by how much of each it covers."""
    if not classes:
        return []
    totals: dict[str, float] = {}
    for shares in classes.values():
        for name, share in shares.items():
            totals[name] = totals.get(name, 0.0) + share
    frames = len(classes)
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])[:n]
    return [{"name": k, "share": round(v / frames, 3)} for k, v in ranked]


def walk_params(walk_id: str) -> PipelineParams:
    """The settings a walk was actually built with, not today's defaults."""
    saved = walk_dir(walk_id) / "params.json"
    if saved.exists():
        return params_from_dict(json.loads(saved.read_text()))
    return PipelineParams()


def pipeline_spec(walk_id: str) -> dict:
    """The stage settings this walk was built with, plus the embedder that
    actually ran (the npz records it, so a model swap stays visible)."""
    npz = walk_dir(walk_id) / "embeddings.npz"
    model = str(np.load(npz)["model"]) if npz.exists() else ""
    return dict(asdict(walk_params(walk_id)), embed_model_used=model,
                backbones=list(BACKBONES), segmenters=SEGMENTERS,
                site_classes=list(SITE_CLASSES))


# which stage a section of the settings belongs to: editing it invalidates
# that stage and everything below it
STAGE_OF = {"extract": "stitch", "gate": "gate", "faces": "faces",
            "embed": "embed", "calib": "calibrate", "dedup": "select",
            "segment": "segment"}


class GatePatch(BaseModel):
    dead: float | None = Field(None, ge=0.0, lt=1.0)
    band: tuple[float, float] | None = None


class FacesPatch(BaseModel):
    fov_deg: float | None = Field(None, gt=10, lt=180)
    size: int | None = Field(None, ge=64, le=4096)
    yaws: list[int] | None = None


class EmbedPatch(BaseModel):
    model_name: str | None = None
    batch_size: int | None = Field(None, ge=1, le=64)


class CalibPatch(BaseModel):
    samples: int | None = Field(None, ge=2, le=200)
    far_seconds: float | None = Field(None, ge=2, le=600)
    false_merge_pct: float | None = Field(None, gt=0, le=50)


class SegmentPatch(BaseModel):
    enabled: bool | None = None
    model_name: str | None = None
    classes: list[str] | None = None
    threshold: float | None = Field(None, gt=0, lt=1)


class DedupPatch(BaseModel):
    tau: float | None = Field(None, gt=0, lt=1)
    rule: str | None = None      # 'fixed' | 'calibrated'
    solo_floor: float | None = Field(None, ge=0.0, lt=1.0)
    solo_span: float | None = Field(None, gt=0)


class Rerun(BaseModel):
    """Staged settings, by the section of the params they belong to."""
    gate: GatePatch | None = None
    faces: FacesPatch | None = None
    embed: EmbedPatch | None = None
    calib: CalibPatch | None = None
    dedup: DedupPatch | None = None
    segment: SegmentPatch | None = None


def check(r: Rerun) -> None:
    """The constraints that are about more than one field."""
    if r.gate and r.gate.band is not None:
        lo, hi = r.gate.band
        if not 0 <= lo < hi <= 1:
            raise HTTPException(422, "the band must be a top below a bottom, "
                                     "both within the panorama")
    # gate.dead and dedup.solo_floor are single-field bounds, so their Field()
    # constraints hold them; this function is for the rules that span fields
    if r.faces and r.faces.yaws is not None:
        y = r.faces.yaws
        if not y or len(set(y)) != len(y) or any(not 0 <= v < 360 for v in y):
            raise HTTPException(422, "yaws must be distinct headings under 360")
    if r.dedup and r.dedup.rule is not None and r.dedup.rule not in (
            "fixed", "calibrated"):
        raise HTTPException(422, f"unknown threshold rule {r.dedup.rule!r}")
    if r.embed and r.embed.model_name is not None \
            and r.embed.model_name not in BACKBONES:
        raise HTTPException(422, f"unknown backbone {r.embed.model_name!r}")
    if r.segment and r.segment.model_name is not None \
            and r.segment.model_name not in SEGMENTERS:
        raise HTTPException(422, f"unknown segmenter {r.segment.model_name!r}")
    if r.segment and r.segment.classes is not None:
        named = [c.strip() for c in r.segment.classes if c.strip()]
        if not named:
            raise HTTPException(422, "name at least one class to look for")
        if len(named) > 40:
            raise HTTPException(422, "too many classes; a long prompt makes the "
                                     "detector merge neighbouring phrases")


def merge(params: PipelineParams, r: Rerun
          ) -> tuple[PipelineParams, str | None, set[str]]:
    """Fold the patch into the walk's params, name the earliest stage it
    invalidates, and say which fields actually moved. Fields set back to what
    the walk already ran are not changes."""
    changed: set[str] = set()
    for section in STAGE_OF:
        patch = getattr(r, section, None)
        if patch is None:
            continue
        current = getattr(params, section)
        # JSON has no tuples; the params dataclasses do
        fields = {k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in patch.model_dump().items() if v is not None}
        fields = {k: v for k, v in fields.items() if getattr(current, k) != v}
        if not fields:
            continue
        params = replace(params, **{section: replace(current, **fields)})
        changed.update(f"{section}.{k}" for k in fields)
    # the input size is not independent of the backbone: it has to divide by
    # the patch size, so it follows the model rather than being chosen
    if "embed.model_name" in changed:
        params = replace(params, embed=replace(
            params.embed, img_size=BACKBONES[params.embed.model_name]))
    stages = {STAGE_OF[f.split(".")[0]] for f in changed}
    first = min(stages, key=pipeline.STAGES.index, default=None)
    return params, first, changed


@app.post("/api/walks/{walk_id}/rerun")
def rerun(walk_id: str, r: Rerun) -> dict:
    """Apply staged settings, re-running from the earliest stage they touch.

    Two of them cost nothing to redo: a threshold re-selects from the cached
    embeddings, and the calibration percentile is a statistic over pairs already
    measured. Both answer with the new counts straight away. Anything that has
    to render or measure again queues on the worker and the canvas follows it.
    """
    out = finished(walk_id)
    check(r)
    if runner().busy(walk_id):
        raise HTTPException(409, "this walk is already running")
    params, first, changed = merge(walk_params(walk_id), r)
    if first is None:
        return {"changed": False}
    # a different budget over the same far pairs is not a new measurement.
    # far_seconds is: it redraws which pairs count as elsewhere. And a record
    # with no far cosines stored has nothing to re-read, so that falls through
    # to the stage rather than resolving a threshold from an empty array.
    if changed == {"calib.false_merge_pct"} and pipeline.can_requantile(out):
        pipeline.requantile(out, params)
        first = "select"
    if first == "select":
        try:
            return dict(pipeline.reselect(out, params), changed=True)
        except ValueError as e:
            # an unmeasurable threshold is the operator's problem to act on,
            # not a server fault, and the message says what to do about it
            raise HTTPException(422, str(e)) from e
    # only the stages that read the original need it: Calibrate stitches the
    # neighbour frames it measures against, and everything below works from
    # the faces already on disk
    needs_video = pipeline.STAGES.index(first) <= pipeline.STAGES.index("calibrate")
    video = walk_video(walk_id) if needs_video else WALKS_ROOT / walk_id
    return dict(runner().submit(video, first=first, params=params,
                                walk_id=walk_id).public(), changed=True)


def walk_video(walk_id: str) -> Path:
    """The original the walk was made from; a re-render reads it again."""
    meta = db.walk_meta(conn(), walk_id) or {}
    video = VIDEO_DIR / Path(meta.get("videoFile") or f"{walk_id}.insv").name
    if not video.exists():
        raise HTTPException(409, f"the original video for {walk_id} is gone")
    return video


# the API speaks camelCase, the table snake_case
META_COLUMNS = {"site": "site", "building": "building", "floor": "floor",
                "stage": "stage", "operator": "operator",
                "mountHeightCm": "mount_height_cm", "shotDate": "shot_date",
                "shotTime": "shot_time", "camera": "camera"}


class MetaPatch(BaseModel):
    """Capture metadata. Absent means unchanged; "" means cleared."""
    site: str | None = Field(None, max_length=200)
    building: str | None = Field(None, max_length=200)
    floor: str | None = Field(None, max_length=200)
    stage: str | None = Field(None, max_length=200)
    operator: str | None = Field(None, max_length=200)
    mountHeightCm: int | None = Field(None, ge=0, le=1000)
    shotDate: str | None = Field(None, max_length=10)
    shotTime: str | None = Field(None, max_length=5)
    camera: str | None = Field(None, max_length=100)


@app.patch("/api/walks/{walk_id}/meta")
def patch_meta(walk_id: str, m: MetaPatch) -> dict:
    """Correct a walk's capture metadata. Nothing derived depends on it, so
    this changes a row and no frames."""
    walk_dir(walk_id)
    fields = {META_COLUMNS[k]: v for k, v in m.model_dump().items()
              if v is not None}
    if fields:
        try:
            db.update_walk_meta(conn(), walk_id, **fields)
        except KeyError:
            raise HTTPException(404, f"no metadata recorded for {walk_id}")
    return db.walk_meta(conn(), walk_id) or {}


class Decision(BaseModel):
    anchorIdx: int
    action: str  # 'pick' | 'drop' | 'restore'
    pickIdx: int | None = None


@app.post("/api/walks/{walk_id}/decisions")
def post_decision(walk_id: str, d: Decision) -> dict:
    finished(walk_id)
    rows = read_manifest(walk_id)
    if d.action not in ("pick", "drop", "restore"):
        raise HTTPException(422, f"unknown action {d.action!r}")
    if not (0 <= d.anchorIdx < len(rows)) or rows[d.anchorIdx]["kept"] != "1":
        raise HTTPException(422, f"{d.anchorIdx} is not a kept anchor")
    if d.action == "pick":
        ok = d.pickIdx is not None and 0 <= d.pickIdx < len(rows) and (
            d.pickIdx == d.anchorIdx
            or rows[d.pickIdx]["anchor"] == str(d.anchorIdx))
        if not ok:
            raise HTTPException(422, f"{d.pickIdx} is not in group {d.anchorIdx}")
    db.log_decision(conn(), walk_id, rows[d.anchorIdx]["path"], d.action,
                    rows[d.pickIdx]["path"] if d.pickIdx is not None else None)
    return {"ok": True}


@app.post("/api/walks/{walk_id}/retry")
def retry(walk_id: str) -> dict:
    """Run the failed stage again with the settings the walk already has.

    A failure clears only when a run succeeds, and re-applying the same
    settings is correctly a no-op, so without this a walk that broke on
    something transient wears the mark until a setting is changed to shake it
    loose.
    """
    out = walk_dir(walk_id)
    failure = pipeline.read_failure(out)
    if failure is None:
        raise HTTPException(409, f"{walk_id} has no failure to retry")
    if runner().busy(walk_id):
        raise HTTPException(409, "this walk is already running")
    first = failure.get("stage") or ""
    video = walk_video(walk_id)
    # a walk that broke at the stitch has nothing to re-enter: start it over
    if first not in pipeline.STAGES or first == "stitch":
        return runner().submit(video).public()
    return runner().submit(video, first=first, params=walk_params(walk_id),
                           walk_id=walk_id).public()


@app.delete("/api/walks/{walk_id}")
def delete_walk(walk_id: str) -> dict:
    """Remove a walk: its frames, its decisions, and the video it came from.

    The video goes too. This is a processing tool, not a store: leaving a
    multi-gigabyte original with nothing in the app pointing at it is not
    keeping the raw ground truth, it is leaking disk.
    """
    wdir = walk_dir(walk_id)
    if runner().busy(walk_id):
        raise HTTPException(409, "this walk is still running")
    videos = [VIDEO_DIR / p.name for p in
              extract.lens_files(VIDEO_DIR / _video_name(walk_id))]
    shutil.rmtree(wdir)
    for v in videos:
        v.unlink(missing_ok=True)
    db.forget_walk(conn(), walk_id)
    return {"deleted": walk_id, "videos": [v.name for v in videos]}


def _video_name(walk_id: str) -> str:
    meta = db.walk_meta(conn(), walk_id) or {}
    return Path(meta.get("videoFile") or f"{walk_id}.insv").name


@app.get("/api/walks/{walk_id}/overrides")
def overrides(walk_id: str) -> list[dict]:
    walk_dir(walk_id)
    return db.override_log(conn(), walk_id)


@app.get("/api/walks/{walk_id}/calibration")
def calibration(walk_id: str) -> dict:
    """How this walk's identical-content reference was measured."""
    calib = pipeline.read_calibration(walk_dir(walk_id))
    if calib is None:
        raise HTTPException(404, f"no calibration for {walk_id}")
    return calib


@app.get("/api/walks/{walk_id}/calib/{name}")
def calib_image(walk_id: str, name: str) -> FileResponse:
    path = (walk_dir(walk_id) / "calib" / name).resolve()
    if not path.is_relative_to(WALKS_ROOT) or not path.exists():
        raise HTTPException(404, "no such calibration face")
    return FileResponse(path)


@app.get("/api/walks/{walk_id}/export")
def export_walk(walk_id: str) -> Response:
    wdir = finished(walk_id)
    rows = read_manifest(walk_id)
    state = db.effective_state(conn(), walk_id)
    meta = db.walk_meta(conn(), walk_id) or {}
    npz = wdir / "embeddings.npz"
    model = str(np.load(npz)["model"]) if npz.exists() else ""
    # Assembled on disk rather than in memory. The archive is the whole export
    # and it scales with the walk, so holding it cost the server ~235 MB for a
    # half-hour walk per reviewer pressing the button. A temp file also means
    # a missing frame fails as a clean 500 before any bytes are sent, where a
    # streamed response would already have committed to 200 and delivered a
    # truncated zip, which is the worst possible outcome for a dataset.
    #
    # the settings this walk was built with, not today's defaults: the EXIF is
    # the provenance record, and a wrong one is worse than none
    EXPORT_TMP.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=EXPORT_TMP)
    try:
        with os.fdopen(fd, "wb") as f:
            export.write_zip(f, walk_id, wdir, rows, state, meta, model,
                             walk_params(walk_id),
                             db.override_log(conn(), walk_id),
                             pipeline.read_segmentation(wdir))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return FileResponse(
        tmp, media_type="application/zip", filename=f"{walk_id}.zip",
        background=BackgroundTask(Path(tmp).unlink, missing_ok=True))


@app.get("/api/walks/{walk_id}/segmentation")
def segmentation(walk_id: str) -> dict:
    """Every kept frame's class mix, with the mask each one has."""
    wdir = walk_dir(walk_id)
    seg = pipeline.read_segmentation(wdir)
    if not seg:
        raise HTTPException(404, f"{walk_id} has not been segmented")
    rows = read_manifest(walk_id)
    t_of = {r["path"]: (float(r["t_sec"]), int(r["yaw"])) for r in rows}
    classes = seg.get("classes", {})
    frames = [{
        "face": name,
        "tSec": t_of.get(name, (0.0, 0))[0],
        "yaw": t_of.get(name, (0.0, 0))[1],
        "url": f"/api/walks/{walk_id}/faces/{name}",
        "mask": f"/api/walks/{walk_id}/masks/{Path(name).stem}.png",
        "classes": shares,
    } for name, shares in classes.items()]
    frames.sort(key=lambda f: (f["tSec"], f["yaw"]))
    return {"model": seg.get("model", ""), "frames": frames,
            # index -> name, so a reader of the masks knows what a pixel means
            "labels": seg.get("labels", {}),
            "classMix": top_classes(classes)}


@app.get("/api/walks/{walk_id}/masks/{name}")
def mask_image(walk_id: str, name: str) -> FileResponse:
    """The mask as written: class indices, so it is near-black on screen. The
    panel colours it; the file stays the label."""
    path = (walk_dir(walk_id) / "masks" / name).resolve()
    if not path.is_relative_to(WALKS_ROOT) or not path.exists():
        raise HTTPException(404, "no such mask")
    return FileResponse(path)


@app.get("/api/walks/{walk_id}/faces/{name}")
def face_image(walk_id: str, name: str, w: int | None = None) -> FileResponse:
    """A face, at its own size or at a thumbnail width.

    The grids show these at about 150 px and the strip at 132, while the file
    is 1024 square and around 280 KB. Sending the original meant a 132-group
    walk pulled ~38 MB to fill squares that need a couple of hundred kilobytes
    between them, and decoded a four-megabyte bitmap per tile; the review grid
    took seconds to populate and scrolling thrashed. A real walk is an order of
    magnitude worse.

    Built on demand rather than at render time, so it costs nothing for a walk
    nobody opens and needs no re-run for the walks that already exist.
    """
    faces = walk_dir(walk_id) / "faces"
    path = (faces / name).resolve()
    if not path.is_relative_to(faces.resolve()) or not path.exists():
        raise HTTPException(404, "no such face")
    if w is None:
        return FileResponse(path)
    if w not in THUMB_WIDTHS:
        raise HTTPException(422, f"thumbnail width must be one of "
                                 f"{sorted(THUMB_WIDTHS)}")
    return FileResponse(thumbnail(path, walk_dir(walk_id) / "thumbs", w))


# a fixed set, so the route cannot be asked to fill a disk with one cache
# entry per width somebody happened to type
THUMB_WIDTHS = {256}


def thumbnail(src: Path, cache: Path, width: int) -> Path:
    """The face at `width`, generated once and kept beside the walk."""
    out = cache / f"{width}" / src.name
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    img = cv2.imread(str(src))
    if img is None:
        raise HTTPException(404, "no such face")
    h = round(img.shape[0] * width / img.shape[1])
    # INTER_AREA is the right filter downwards: it averages the pixels it drops
    # rather than sampling past them, so thin structures survive as grey rather
    # than disappearing between samples
    small = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic.atomically(out, lambda tmp: cv2.imwrite(
        str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 82]))
    return out


# The built frontend, when there is one. Mounted last because "/" matches
# everything, so any route declared after it would be shadowed by a 404 from
# StaticFiles. In dev there is no dist/ and Vite serves the app itself, which
# is why this is a condition rather than a requirement.
if settings.WEB_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=settings.WEB_DIST, html=True),
              name="web")
