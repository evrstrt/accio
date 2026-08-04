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
import os
import shutil
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..core import export, extract
from ..core.params import PipelineParams
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
    d = (WALKS_ROOT / walk_id).resolve()
    if not d.is_relative_to(WALKS_ROOT) or not (d / "manifest.csv").exists():
        raise HTTPException(404, f"unknown walk {walk_id!r}")
    return d


def read_manifest(walk_id: str) -> list[dict]:
    with open(walk_dir(walk_id) / "manifest.csv") as f:
        return list(csv.DictReader(f))


@app.get("/api/walks")
def walks() -> list[dict]:
    out = []
    for d in sorted(WALKS_ROOT.iterdir()) if WALKS_ROOT.exists() else []:
        if not (d / "manifest.csv").exists():
            continue
        rows = read_manifest(d.name)
        state = db.effective_state(conn(), d.name)
        n_dropped = sum(1 for s in state.values() if s["dropped"])
        out.append({
            "id": d.name,
            "faces": len(rows),
            "kept": sum(r["kept"] == "1" for r in rows) - n_dropped,
            "meta": db.walk_meta(conn(), d.name),
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
    groups = []
    for f in faces:
        if not f["kept"]:
            continue
        s = state.get(f["idx"], {"pick": None, "dropped": False})
        groups.append({
            "anchor": f,
            "members": sorted((m for m in faces if m["anchor"] == f["idx"]),
                              key=lambda m: -m["cosine"]),
            "pick": s["pick"] if s["pick"] is not None else f["idx"],
            "dropped": s["dropped"],
        })
    return {"id": walk_id, "faces": len(faces), "groups": groups,
            "meta": db.walk_meta(conn(), walk_id),
            "stages": stage_counts(walk_id, rows, state),
            "pipeline": pipeline_spec(walk_id)}


def stage_counts(walk_id: str, rows: list[dict], state: dict) -> dict:
    """What each stage emitted, for the pipeline view. The pre-gate pano count
    comes from the stitched files on disk; everything else from the manifest."""
    pano_dir = walk_dir(walk_id) / "pano"
    anchors = sum(1 for r in rows if r["kept"] == "1")
    dropped = sum(1 for s in state.values() if s["dropped"])
    return {
        "panos": len(list(pano_dir.glob(f"{extract.PANO_PREFIX}*{extract.PANO_EXT}"))),
        "sharp": len({r["pano_idx"] for r in rows}),
        "faces": len(rows),
        "anchors": anchors,
        "absorbed": len(rows) - anchors,
        "dropped": dropped,
        "kept": anchors - dropped,
    }


def pipeline_spec(walk_id: str) -> dict:
    """The stage settings this walk was built with, plus the embedder that
    actually ran (the npz records it, so a model swap stays visible)."""
    npz = walk_dir(walk_id) / "embeddings.npz"
    model = str(np.load(npz)["model"]) if npz.exists() else ""
    return dict(asdict(PipelineParams()), embed_model_used=model)


class Decision(BaseModel):
    anchorIdx: int
    action: str  # 'pick' | 'drop' | 'restore'
    pickIdx: int | None = None


@app.post("/api/walks/{walk_id}/decisions")
def post_decision(walk_id: str, d: Decision) -> dict:
    rows = read_manifest(walk_id)  # 404s on unknown walk
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
    db.log_decision(conn(), walk_id, d.anchorIdx, d.action, d.pickIdx)
    return {"ok": True}


@app.get("/api/walks/{walk_id}/overrides")
def overrides(walk_id: str) -> list[dict]:
    walk_dir(walk_id)
    return db.override_log(conn(), walk_id)


@app.get("/api/walks/{walk_id}/export")
def export_walk(walk_id: str) -> Response:
    wdir = walk_dir(walk_id)
    rows = read_manifest(walk_id)
    state = db.effective_state(conn(), walk_id)
    meta = db.walk_meta(conn(), walk_id) or {}
    npz = wdir / "embeddings.npz"
    model = str(np.load(npz)["model"]) if npz.exists() else ""
    data = export.build_zip(walk_id, wdir, rows, state, meta, model,
                            PipelineParams())
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{walk_id}.zip"'})


@app.get("/api/walks/{walk_id}/faces/{name}")
def face_image(walk_id: str, name: str) -> FileResponse:
    path = (walk_dir(walk_id) / "faces" / name).resolve()
    if not path.is_relative_to(WALKS_ROOT) or not path.exists():
        raise HTTPException(404, "no such face")
    return FileResponse(path)
