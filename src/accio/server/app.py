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
import threading
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core import export, extract
from ..core.params import BACKBONES, PipelineParams
from ..core.params import from_dict as params_from_dict
from ..jobs import pipeline
from ..jobs.runner import Runner
from ..store import db

DATA_ROOT = Path(os.environ.get("ACCIO_DATA", "data")).resolve()
VIDEO_DIR = DATA_ROOT / "videos"
WALKS_ROOT = DATA_ROOT / "walks"

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
            _runner = Runner(WALKS_ROOT)
    return _runner


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


@app.get("/api/walks")
def walks() -> list[dict]:
    out = []
    for d in sorted(WALKS_ROOT.iterdir()) if WALKS_ROOT.exists() else []:
        if not d.is_dir():
            continue
        rows = read_manifest(d.name)
        state = db.effective_state(conn(), d.name)
        n_dropped, _ = export.review_counts(rows, state)
        out.append({
            "id": d.name,
            "faces": len(rows),
            "kept": sum(r["kept"] == "1" for r in rows) - n_dropped,
            "meta": db.walk_meta(conn(), d.name),
            # a walk with no manifest never finished: it is listed so it can be
            # seen and removed, not because there is anything to review
            "ready": bool(rows),
            "error": pipeline.read_failure(d),
        })
    return out


@app.post("/api/ingest")
async def ingest(
    files: list[UploadFile] = File([]),
    site: str = Form(""),
    building: str = Form(""),
    stage: str = Form(""),
    operator: str = Form(""),
    mount_height_cm: int | None = Form(None),
    shot_date: str = Form(""),
) -> dict:
    # Upload is the only ingest: this is a web app, the server has no access
    # to client paths. Dual-file recordings (_00_ front + _10_ back) arrive
    # as two uploads; the app owns its copy of the originals.
    if not files or not files[0].filename:
        raise HTTPException(422, "upload the .insv file(s)")
    # a _10_ on its own is the back lens of a dual-file recording: the SDK
    # stitches nothing from it, so say so now rather than after the upload
    names = [Path(f.filename or "").name for f in files]
    if any("_10_" in n for n in names) and not any("_00_" in n for n in names):
        raise HTTPException(422, "that is the back lens only; upload the _00_ "
                                 "file alongside it")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for up in files:
        dst = VIDEO_DIR / Path(up.filename).name
        with open(dst, "wb") as f:
            shutil.copyfileobj(up.file, f)
        saved.append(dst)
    video = extract.front_lens(saved)

    db.save_walk_meta(conn(), video.stem, video.name, site=site,
                      building=building, stage=stage, operator=operator,
                      mount_height_cm=mount_height_cm, shot_date=shot_date)
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
        groups.append({
            "anchor": f,
            "members": sorted((m for m in faces if m["anchor"] == f["idx"]),
                              key=lambda m: -m["cosine"]),
            "pick": pick if pick is not None else f["idx"],
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
    return {
        "frames": src.get("frames", 0),
        "seconds": src.get("seconds", 0),
        "panos": len(list(pano_dir.glob(f"{extract.PANO_PREFIX}*{extract.PANO_EXT}"))),
        "sharp": len({r["pano_idx"] for r in rows}),
        "faces": len(rows),
        # the calibration stage's own output: how alike identical frames were,
        # over how many pairs, and the threshold that resolves to
        "pairs": calib.get("reference", {}).get("n", 0),
        "reference": calib.get("reference", {}).get("median", 0),
        "calibTau": calib.get("tau", 0),
        "anchors": anchors,
        "absorbed": len(rows) - anchors,
        "dropped": dropped,
        "overridden": overridden,
        "kept": anchors - dropped,
    }


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
                backbones=list(BACKBONES))


# which stage a section of the settings belongs to: editing it invalidates
# that stage and everything below it
STAGE_OF = {"extract": "stitch", "gate": "gate", "faces": "faces",
            "embed": "embed", "calib": "calibrate", "dedup": "select"}


class GatePatch(BaseModel):
    window: int | None = Field(None, ge=1, le=120)
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
    quantile: float | None = Field(None, gt=0, le=50)


class DedupPatch(BaseModel):
    tau: float | None = Field(None, gt=0, lt=1)
    rule: str | None = None      # 'fixed' | 'calibrated'


class Rerun(BaseModel):
    """Staged settings, by the section of the params they belong to."""
    gate: GatePatch | None = None
    faces: FacesPatch | None = None
    embed: EmbedPatch | None = None
    calib: CalibPatch | None = None
    dedup: DedupPatch | None = None


def check(r: Rerun) -> None:
    """The constraints that are about more than one field."""
    if r.gate and r.gate.band is not None:
        lo, hi = r.gate.band
        if not 0 <= lo < hi <= 1:
            raise HTTPException(422, "the band must be a top below a bottom, "
                                     "both within the panorama")
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
    # a different percentile of the same pairs is not a new measurement
    if changed == {"calib.quantile"}:
        pipeline.requantile(out, params)
        first = "select"
    if first == "select":
        return dict(pipeline.reselect(out, params), changed=True)
    video = walk_video(walk_id)
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
META_COLUMNS = {"site": "site", "building": "building", "stage": "stage",
                "operator": "operator", "mountHeightCm": "mount_height_cm",
                "shotDate": "shot_date"}


class MetaPatch(BaseModel):
    """Capture metadata. Absent means unchanged; "" means cleared."""
    site: str | None = Field(None, max_length=200)
    building: str | None = Field(None, max_length=200)
    stage: str | None = Field(None, max_length=200)
    operator: str | None = Field(None, max_length=200)
    mountHeightCm: int | None = Field(None, ge=0, le=1000)
    shotDate: str | None = Field(None, max_length=10)


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
    # the settings this walk was built with, not today's defaults: the EXIF is
    # the provenance record, and a wrong one is worse than none
    data = export.build_zip(walk_id, wdir, rows, state, meta, model,
                            walk_params(walk_id),
                            db.override_log(conn(), walk_id))
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{walk_id}.zip"'})


@app.get("/api/walks/{walk_id}/faces/{name}")
def face_image(walk_id: str, name: str) -> FileResponse:
    path = (walk_dir(walk_id) / "faces" / name).resolve()
    if not path.is_relative_to(WALKS_ROOT) or not path.exists():
        raise HTTPException(404, "no such face")
    return FileResponse(path)
