"""FastAPI server over the pipeline output and the decisions store.

Run:  uv run uvicorn accio.server.app:app --reload
Data root (ACCIO_DATA, default ./data): videos/ holds the original .insv
files, walks/<id>/ the derived frames, accio.db the decisions and metadata.
"""

import os
import shutil
import sqlite3
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask
from starlette.datastructures import Headers

from .. import settings
from ..core import export, extract
from ..core.params import BACKBONES, SEGMENTERS, PipelineParams
from ..jobs import pipeline, review, thumbs, upload
from ..jobs.runner import Runner
from ..store import db

DATA_ROOT = settings.DATA_ROOT
VIDEO_DIR = DATA_ROOT / "videos"
WALKS_ROOT = DATA_ROOT / "walks"
# under the data root, not /tmp, so uploads and archives cannot fill a
# container's writable layer
INCOMING = DATA_ROOT / ".incoming"
EXPORT_TMP = DATA_ROOT / ".exports"

# the store and every manifest write share the volume with uploads
FREE_HEADROOM = 5 * 2**30


# the runner's recover() must run at startup, not on the first mutating request
@asynccontextmanager
async def lifespan(app: FastAPI):
    reap_incoming()
    INCOMING.mkdir(parents=True, exist_ok=True)
    # Starlette spools a multipart body through tempfile before the route runs
    tempfile.tempdir = str(INCOMING)
    runner()
    yield


app = FastAPI(title="accio", lifespan=lifespan)

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


def reap_incoming() -> None:
    """Drop staged uploads and half-built exports a dead process left behind."""
    shutil.rmtree(INCOMING, ignore_errors=True)
    shutil.rmtree(VIDEO_DIR / ".incoming", ignore_errors=True)  # where older builds staged
    shutil.rmtree(EXPORT_TMP, ignore_errors=True)


def too_big() -> str:
    return (f"upload is over the {settings.MAX_UPLOAD_BYTES // 2**30} GB limit "
            "(ACCIO_MAX_UPLOAD_GB raises it)")


class UploadCap:
    """413 by Content-Length, before Starlette spools the body to disk."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/api/ingest":
            length = Headers(scope=scope).get("content-length", "")
            if length.isdigit() and int(length) > settings.MAX_UPLOAD_BYTES:
                await JSONResponse({"detail": too_big()}, 413)(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(UploadCap)


def walk_dir(walk_id: str) -> Path:
    """Any walk directory, manifest or not, so a failed run can be listed and deleted."""
    d = (WALKS_ROOT / walk_id).resolve()
    if not d.is_relative_to(WALKS_ROOT) or not d.is_dir():
        raise HTTPException(404, f"unknown walk {walk_id!r}")
    return d


def finished(walk_id: str) -> Path:
    """A walk that got as far as a manifest."""
    d = walk_dir(walk_id)
    if not (d / "manifest.csv").exists():
        raise HTTPException(409, f"{walk_id} has no frames; its run did not finish")
    return d


def read_manifest(walk_id: str) -> list[dict]:
    d = walk_dir(walk_id)
    return pipeline.manifest_rows(d) if (d / "manifest.csv").exists() else []


# walk id -> (manifest mtime, face count, anchor names)
_summary: dict[str, tuple[int, int, set[str]]] = {}


def walk_summary(walk_id: str) -> tuple[int, set[str]]:
    """(face count, anchor names), cached on the manifest's mtime.

    The list endpoint is polled; parsing every manifest per request measured
    1.8 s of CPU at 500 walks. The manifest is written atomically, so the
    mtime never changes mid-write.
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


def face_url(walk_id: str, name: str) -> str:
    return f"/api/walks/{quote(walk_id, safe='')}/faces/{quote(name, safe='')}"


def mask_url(walk_id: str, name: str) -> str:
    return f"/api/walks/{quote(walk_id, safe='')}/masks/{quote(name, safe='')}"


@app.get("/api/health")
def health() -> JSONResponse:
    worker = runner().alive()
    try:
        store = conn().execute("SELECT 1").fetchone() == (1,)
    except sqlite3.Error:
        store = False
    ok = worker and store
    return JSONResponse({"ok": ok, "worker": worker, "store": store},
                        200 if ok else 503)


@app.get("/api/walks")
def walks() -> list[dict]:
    out = []
    for d in sorted(WALKS_ROOT.iterdir()) if WALKS_ROOT.exists() else []:
        if not d.is_dir():
            continue
        faces, anchors = walk_summary(d.name)
        state = db.effective_state(conn(), d.name)
        out.append({
            "id": d.name,
            "faces": faces,
            "kept": len(anchors) - review.dropped_count(anchors, state),
            "meta": db.walk_meta(conn(), d.name),
            "ready": bool(faces),
            "error": pipeline.read_failure(d),
        })
    return out


@app.post("/api/ingest")
# plain def: the copy and ffprobe block, so FastAPI runs this in the threadpool
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
    # dual-file recordings arrive as two uploads: _00_ front + _10_ back
    names = [Path(f.filename or "").name for f in files]
    if not names or not all(names):
        raise HTTPException(422, "upload the .insv file(s)")
    # date and time are not required: the file carries them
    need = {"site": site, "building": building, "floor": floor, "stage": stage,
            "operator": operator,
            "mount height": "" if mount_height_cm is None else "set"}
    blank = [k for k, v in need.items() if not str(v).strip()]
    if blank:
        raise HTTPException(422, f"still needed: {', '.join(blank)}")
    # the SDK stitches nothing from a _10_ file alone
    if any("_10_" in n for n in names) and not any("_00_" in n for n in names):
        raise HTTPException(422, "that is the back lens only; upload the _00_ "
                                 "file alongside it")
    try:
        saved = upload.stage_upload([(n, f.file) for n, f in zip(names, files)],
                                    INCOMING, settings.MAX_UPLOAD_BYTES,
                                    FREE_HEADROOM)
    except upload.UploadTooLarge:
        raise HTTPException(413, too_big())
    except upload.VolumeFull:
        raise HTTPException(507, "the data volume is nearly full; the store and "
                                 "the walks live there too, so make room before "
                                 "uploading")
    try:
        try:
            video = upload.check_recording(saved)
        except upload.NotARecording as e:
            raise HTTPException(422, str(e)) from e
        # the walk id is the file stem; a re-upload would inherit the old decisions
        walk = WALKS_ROOT / video.stem
        try:
            walk.mkdir(parents=True)
        except FileExistsError:
            raise HTTPException(409, f"{video.stem} is already a walk. Delete "
                                     "it first if you mean to replace it.")
        try:
            VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            for p in saved:
                shutil.move(str(p), VIDEO_DIR / p.name)
            video = VIDEO_DIR / video.name
            # a form value wins over the file's timestamp: camera clocks go unset
            cam = extract.camera(video)
            stamp = extract.recorded_at(video)
            db.save_walk_meta(conn(), video.stem, video.name, site=site,
                              building=building, floor=floor, stage=stage,
                              operator=operator, mount_height_cm=mount_height_cm,
                              shot_date=shot_date or stamp[:10],
                              shot_time=shot_time or stamp[11:16],
                              camera=cam.get("model", ""))
            return runner().submit(video).public()
        except BaseException:
            shutil.rmtree(walk, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(saved[0].parent, ignore_errors=True)


@app.get("/api/jobs")
def jobs() -> list[dict]:
    return runner().list()


@app.get("/api/walks/{walk_id}")
def walk_detail(walk_id: str) -> dict:
    wdir = walk_dir(walk_id)
    rows = read_manifest(walk_id)
    state = db.effective_state(conn(), walk_id)
    faces, groups = review.groups(rows, state, lambda n: face_url(walk_id, n))
    runs = pipeline.read_runs(wdir)
    return {"id": walk_id, "faces": len(faces), "groups": groups,
            "meta": db.walk_meta(conn(), walk_id),
            "stages": pipeline.stage_counts(wdir, rows, state, runs),
            "pipeline": pipeline.pipeline_spec(wdir),
            "runs": runs,
            "error": pipeline.read_failure(wdir)}


class Patch(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GatePatch(Patch):
    dead: float | None = Field(None, ge=0.0, lt=1.0)
    band: tuple[float, float] | None = None


class FacesPatch(Patch):
    fov_deg: float | None = Field(None, gt=10, lt=180)
    size: int | None = Field(None, ge=64, le=4096)
    yaws: list[int] | None = None


class EmbedPatch(Patch):
    model_name: str | None = None
    batch_size: int | None = Field(None, ge=1, le=64)


class CalibPatch(Patch):
    samples: int | None = Field(None, ge=2, le=200)
    far_seconds: float | None = Field(None, ge=2, le=600)
    false_merge_pct: float | None = Field(None, gt=0, le=50)


class SegmentPatch(Patch):
    enabled: bool | None = None
    model_name: str | None = None
    classes: list[str] | None = None
    threshold: float | None = Field(None, gt=0, lt=1)


class DedupPatch(Patch):
    tau: float | None = Field(None, gt=0, lt=1)
    rule: str | None = None      # 'fixed' | 'calibrated'
    solo_floor: float | None = Field(None, ge=0.0, lt=1.0)
    solo_span: float | None = Field(None, gt=0)


class Rerun(Patch):
    """Settings patch, by params section."""
    gate: GatePatch | None = None
    faces: FacesPatch | None = None
    embed: EmbedPatch | None = None
    calib: CalibPatch | None = None
    dedup: DedupPatch | None = None
    segment: SegmentPatch | None = None


def check(r: Rerun) -> None:
    """Constraints Field() cannot express."""
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
    """(new params, earliest stage invalidated, fields that differ from the walk's)."""
    return pipeline.merge(params, r.model_dump(exclude_none=True))


@app.post("/api/walks/{walk_id}/rerun")
def rerun(walk_id: str, r: Rerun) -> dict:
    """Apply a settings patch from the earliest stage it touches.

    A threshold or budget change answers inline from what is on disk;
    anything else queues on the worker.
    """
    out = finished(walk_id)
    check(r)
    if runner().busy(walk_id):
        raise HTTPException(409, "this walk is already running")
    params, first, changed = merge(pipeline.walk_params(out), r)
    if first is None:
        return {"changed": False}
    # a record without far cosines falls through to the calibrate stage
    if changed == {"calib.false_merge_pct"} and pipeline.can_requantile(out):
        pipeline.requantile(out, params)
        first = "select"
    if first == "select":
        try:
            return dict(pipeline.reselect(out, params), changed=True)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    # calibrate stitches neighbour frames from the original; later stages do not
    needs_video = pipeline.STAGES.index(first) <= pipeline.STAGES.index("calibrate")
    video = walk_video(walk_id) if needs_video else None
    return dict(runner().submit(video, first=first, params=params,
                                walk_id=walk_id).public(), changed=True)


def video_path(walk_id: str) -> Path:
    meta = db.walk_meta(conn(), walk_id) or {}
    return VIDEO_DIR / Path(meta.get("videoFile") or f"{walk_id}.insv").name


def walk_video(walk_id: str) -> Path:
    video = video_path(walk_id)
    if not video.exists():
        raise HTTPException(409, f"the original video for {walk_id} is gone")
    return video


# the API speaks camelCase, the table snake_case
META_COLUMNS = {"site": "site", "building": "building", "floor": "floor",
                "stage": "stage", "operator": "operator",
                "mountHeightCm": "mount_height_cm", "shotDate": "shot_date",
                "shotTime": "shot_time", "camera": "camera"}


class MetaPatch(Patch):
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
    pick = None
    if d.action == "pick":
        ok = d.pickIdx is not None and 0 <= d.pickIdx < len(rows) and (
            d.pickIdx == d.anchorIdx
            or rows[d.pickIdx]["anchor"] == str(d.anchorIdx))
        if not ok:
            raise HTTPException(422, f"{d.pickIdx} is not in group {d.anchorIdx}")
        pick = rows[d.pickIdx]["path"]
    db.log_decision(conn(), walk_id, rows[d.anchorIdx]["path"], d.action, pick)
    return {"ok": True}


@app.post("/api/walks/{walk_id}/retry")
def retry(walk_id: str) -> dict:
    """Run the failed stage again with the walk's own settings."""
    out = walk_dir(walk_id)
    failure = pipeline.read_failure(out)
    if failure is None:
        raise HTTPException(409, f"{walk_id} has no failure to retry")
    if runner().busy(walk_id):
        raise HTTPException(409, "this walk is already running")
    first = failure.get("stage") or ""
    video = walk_video(walk_id)
    # rerun() cannot enter at the stitch; start over
    if first not in pipeline.STAGES or first == "stitch":
        return runner().submit(video).public()
    return runner().submit(video, first=first, params=pipeline.walk_params(out),
                           walk_id=walk_id).public()


@app.delete("/api/walks/{walk_id}")
def delete_walk(walk_id: str) -> dict:
    """Remove the frames, the decisions and the original video."""
    wdir = walk_dir(walk_id)
    if runner().busy(walk_id):
        raise HTTPException(409, "this walk is still running")
    videos = [VIDEO_DIR / p.name for p in extract.lens_files(video_path(walk_id))]
    shutil.rmtree(wdir)
    _summary.pop(walk_id, None)
    for v in videos:
        v.unlink(missing_ok=True)
    db.forget_walk(conn(), walk_id)
    return {"deleted": walk_id, "videos": [v.name for v in videos]}


@app.get("/api/walks/{walk_id}/overrides")
def overrides(walk_id: str) -> list[dict]:
    walk_dir(walk_id)
    return db.override_log(conn(), walk_id)


@app.get("/api/walks/{walk_id}/calibration")
def calibration(walk_id: str) -> dict:
    calib = pipeline.read_calibration(walk_dir(walk_id))
    if calib is None:
        raise HTTPException(404, f"no calibration for {walk_id}")
    return calib


@app.get("/api/walks/{walk_id}/export")
def export_walk(walk_id: str) -> Response:
    wdir = finished(walk_id)
    rows = read_manifest(walk_id)
    state = db.effective_state(conn(), walk_id)
    meta = db.walk_meta(conn(), walk_id) or {}
    # built on disk: in memory cost ~235 MB per request for a half-hour walk,
    # and a temp file lets a missing frame fail before any bytes are sent
    EXPORT_TMP.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=EXPORT_TMP)
    try:
        with os.fdopen(fd, "wb") as f:
            export.write_zip(f, walk_id, wdir, rows, state, meta,
                             pipeline.embed_model_used(wdir),
                             pipeline.walk_params(wdir),
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
    wdir = walk_dir(walk_id)
    seg = pipeline.read_segmentation(wdir)
    if not seg:
        raise HTTPException(404, f"{walk_id} has not been segmented")
    return review.segmentation(seg, read_manifest(walk_id),
                               lambda n: face_url(walk_id, n),
                               lambda n: mask_url(walk_id, n))


def image(folder: Path, name: str, cache: Path, w: int | None,
          what: str) -> FileResponse:
    """A file under `folder`, at full size or at a thumbnail width."""
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise HTTPException(404, f"no such {what}")
    try:
        return FileResponse(thumbs.sized(path, cache, w))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except FileNotFoundError:
        raise HTTPException(404, f"no such {what}")


@app.get("/api/walks/{walk_id}/calib/{name}")
def calib_image(walk_id: str, name: str, w: int | None = None) -> FileResponse:
    wdir = walk_dir(walk_id)
    return image(wdir / "calib", name, wdir / "thumbs" / "calib", w,
                 "calibration face")


@app.get("/api/walks/{walk_id}/masks/{name}")
def mask_image(walk_id: str, name: str, w: int | None = None) -> FileResponse:
    """Class indices, so near-black on screen; the panel colours it."""
    wdir = walk_dir(walk_id)
    return image(wdir / "masks", name, wdir / "thumbs" / "masks", w, "mask")


@app.get("/api/walks/{walk_id}/faces/{name}")
def face_image(walk_id: str, name: str, w: int | None = None) -> FileResponse:
    wdir = walk_dir(walk_id)
    return image(wdir / "faces", name, wdir / "thumbs", w, "face")


# mounted last: "/" shadows any route declared after it. In dev Vite serves the app
if settings.WEB_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=settings.WEB_DIST, html=True),
              name="web")
