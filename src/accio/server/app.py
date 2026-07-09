"""FastAPI server: pipeline output (immutable) + review decisions (store).

No science here. Routes translate manifest.csv and the decisions store into
JSON and serve images; the pipeline stays runnable headless and the frontend
stays swappable.

Run:  uv run uvicorn accio.server.app:app --reload
Data root defaults to ./out; override with ACCIO_OUT. The decisions DB lives
at <data root>/accio.db.
"""

import csv
import os
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..jobs.runner import Runner
from ..store import db

OUT_ROOT = Path(os.environ.get("ACCIO_OUT", "out")).resolve()
VIDEO_DIR = OUT_ROOT / "_videos"  # leading underscore: never mistaken for a walk

app = FastAPI(title="accio")

_local = threading.local()
_runner: Runner | None = None
_runner_lock = threading.Lock()


def conn():
    if not hasattr(_local, "conn"):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        _local.conn = db.connect(OUT_ROOT / "accio.db")
    return _local.conn


def runner() -> Runner:
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = Runner(OUT_ROOT)
    return _runner


def walk_dir(walk_id: str) -> Path:
    d = (OUT_ROOT / walk_id).resolve()
    if not d.is_relative_to(OUT_ROOT) or not (d / "manifest.csv").exists():
        raise HTTPException(404, f"unknown walk {walk_id!r}")
    return d


def read_manifest(walk_id: str) -> list[dict]:
    with open(walk_dir(walk_id) / "manifest.csv") as f:
        return list(csv.DictReader(f))


@app.get("/api/walks")
def walks() -> list[dict]:
    out = []
    for d in sorted(OUT_ROOT.iterdir()) if OUT_ROOT.exists() else []:
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
    file: UploadFile | None = None,
    source_path: str = Form(""),
    site: str = Form(""),
    building: str = Form(""),
    stage: str = Form(""),
    operator: str = Form(""),
    mount_height_cm: int | None = Form(None),
    shot_date: str = Form(""),
) -> dict:
    if file is not None and file.filename:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        video = VIDEO_DIR / Path(file.filename).name
        with open(video, "wb") as f:
            shutil.copyfileobj(file.file, f)
    elif source_path:
        video = Path(source_path).expanduser()
        if not video.is_file():
            raise HTTPException(422, f"no such file: {video}")
    else:
        raise HTTPException(422, "provide a video file or a source_path")

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
            "meta": db.walk_meta(conn(), walk_id)}


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


@app.get("/api/walks/{walk_id}/faces/{name}")
def face_image(walk_id: str, name: str) -> FileResponse:
    path = (walk_dir(walk_id) / "faces" / name).resolve()
    if not path.is_relative_to(OUT_ROOT) or not path.exists():
        raise HTTPException(404, "no such face")
    return FileResponse(path)
