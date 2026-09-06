"""HTTP routes through the ASGI app: validation, limits, review state."""

import json
import shutil
import tempfile
import threading

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from accio import settings
from accio.core import extract
from accio.server import app as server
from accio.store import db

MANIFEST = (
    "pano_idx,t_sec,yaw,path,kept,anchor,cosine,sharpness\n"
    "0,0.5,45,y045_00000.jpg,1,,,90.0\n"
    "1,1.0,45,y045_00001.jpg,0,0,0.9700,40.0\n"
    "2,1.5,135,y135_00002.jpg,1,,,70.0\n")

META = {"site": "gcmr", "building": "t2", "floor": "3", "stage": "casco",
        "operator": "ed", "mount_height_cm": "180"}

FILE = [("files", ("VID_00_1.insv", b"0" * 64))]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(server, "VIDEO_DIR", tmp_path / "videos")
    monkeypatch.setattr(server, "WALKS_ROOT", tmp_path / "walks")
    monkeypatch.setattr(server, "INCOMING", tmp_path / ".incoming")
    monkeypatch.setattr(server, "EXPORT_TMP", tmp_path / ".exports")
    monkeypatch.setattr(server, "_summary", {})
    monkeypatch.setattr(server, "_runner", None)
    monkeypatch.setattr(server, "FREE_HEADROOM", 0)
    # startup points the process's tempdir at INCOMING; put it back afterwards
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    # per-thread like the real conn, whose thread-local cache outlives tmp_path
    conns: dict[int, object] = {}

    def conn():
        t = threading.get_ident()
        if t not in conns:
            conns[t] = db.connect(tmp_path / "accio.db")
        return conns[t]

    monkeypatch.setattr(server, "conn", conn)
    with TestClient(server.app) as c:
        yield c


@pytest.fixture
def probed(monkeypatch):
    """ffprobe answers a dual-lens frame, so any bytes pass as a recording;
    the worker thread is already running, so the job handler is stubbed."""
    monkeypatch.setattr(server.Runner, "_run_one", lambda self, job: None)
    monkeypatch.setattr(extract, "probe", lambda v: (29.97, 300, 3840, 1920))
    monkeypatch.setattr(extract, "camera", lambda v: {"model": "X4"})
    monkeypatch.setattr(extract, "recorded_at", lambda v: "2026-07-01T09:30:00")


def walk(tmp_path, name="w1", manifest=MANIFEST):
    d = tmp_path / "walks" / name
    d.mkdir(parents=True)
    (d / "manifest.csv").write_text(manifest)
    return d


def test_health_answers(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_health_503_without_worker(client):
    server._runner._thread = threading.Thread(target=lambda: None)
    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.json() == {"ok": False, "worker": False, "store": True}


def test_startup_runner_and_tempdir(
        client, tmp_path):
    assert server._runner is not None
    assert tempfile.tempdir == str(tmp_path / ".incoming")
    assert (tmp_path / ".incoming").is_dir()


def test_ingest_refuses_no_files(client):
    assert client.post("/api/ingest", data=META).status_code == 422


def test_chunked_upload_reaches_route(client):
    """The multipart helper must produce what Starlette parses."""
    ctype, parts = multipart({"site": "gcmr"}, [("VID_00_1.insv", b"x")])
    r = client.post("/api/ingest", content=iter(parts),
                    headers={"Content-Type": ctype})
    assert r.status_code == 422
    assert "building" in r.json()["detail"]


def test_ingest_refuses_unnamed_file(client):
    r = client.post("/api/ingest", data=META,
                    files=FILE + [("files", ("", b"x"))])
    assert r.status_code == 422


def test_ingest_names_missing_metadata(client):
    r = client.post("/api/ingest", data={"site": "gcmr"},
                    files=[("files", ("VID_00_1.insv", b"x"))])
    assert r.status_code == 422
    assert "building" in r.json()["detail"]


def test_ingest_refuses_back_lens_alone(client):
    r = client.post("/api/ingest", data=META,
                    files=[("files", ("VID_10_1.insv", b"x"))])
    assert r.status_code == 422
    assert "_00_" in r.json()["detail"]


def test_upload_over_cap_413_before_body(
        client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 16)

    def never(*a):
        raise AssertionError("the body was parsed")

    monkeypatch.setattr(server.upload, "stage_upload", never)
    r = client.post("/api/ingest", data=META, files=FILE)
    assert r.status_code == 413
    assert "GB limit" in r.json()["detail"]
    assert list((tmp_path / ".incoming").iterdir()) == []


def multipart(fields: dict, files: list) -> tuple[str, list[bytes]]:
    b = "boundary1234"
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
             .encode() for k, v in fields.items()]
    parts += [f'--{b}\r\nContent-Disposition: form-data; name="files"; '
              f'filename="{name}"\r\nContent-Type: application/octet-stream'
              f'\r\n\r\n'.encode() + data + b"\r\n" for name, data in files]
    parts.append(f"--{b}--\r\n".encode())
    return f"multipart/form-data; boundary={b}", parts


def test_chunked_upload_over_cap_413(
        client, tmp_path, monkeypatch):
    """Without a Content-Length the copy loop is the only guard."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 16)
    ctype, parts = multipart(META, [("VID_00_1.insv", b"0" * 64)])
    r = client.post("/api/ingest", content=iter(parts),
                    headers={"Content-Type": ctype})
    assert r.status_code == 413
    assert list((tmp_path / ".incoming").iterdir()) == []


def test_full_volume_507(client, monkeypatch):
    monkeypatch.setattr(server, "FREE_HEADROOM", 2**60)
    r = client.post("/api/ingest", data=META, files=FILE)
    assert r.status_code == 507


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")
def test_non_video_422(client):
    r = client.post("/api/ingest", data=META,
                    files=[("files", ("VID_00_1.insv", b"not a video"))])
    assert r.status_code == 422
    assert "VID_00_1.insv" in r.json()["detail"]


def test_ingest_claims_dir_and_queues(client, tmp_path,
                                                        probed):
    r = client.post("/api/ingest", data=META, files=FILE)
    assert r.status_code == 200, r.text
    assert r.json()["walkId"] == "VID_00_1"
    assert (tmp_path / "videos" / "VID_00_1.insv").read_bytes() == b"0" * 64
    assert (tmp_path / "walks" / "VID_00_1").is_dir()
    assert list((tmp_path / ".incoming").iterdir()) == []
    assert client.get("/api/walks/VID_00_1").json()["meta"]["camera"] == "X4"


def test_reupload_409(client, tmp_path, probed):
    walk(tmp_path, "VID_00_1")
    r = client.post("/api/ingest", data=META, files=FILE)
    assert r.status_code == 409
    assert not (tmp_path / "videos").exists()
    assert (tmp_path / "walks" / "VID_00_1" / "manifest.csv").exists()


def test_failed_ingest_releases_dir(
        client, tmp_path, probed, monkeypatch):
    def broken(video):
        raise RuntimeError("cannot read the camera model")

    monkeypatch.setattr(extract, "camera", broken)
    with pytest.raises(RuntimeError):
        client.post("/api/ingest", data=META, files=FILE)
    assert not (tmp_path / "walks" / "VID_00_1").exists()


def test_decision_outside_group_refused(client, tmp_path):
    walk(tmp_path)
    r = client.post("/api/walks/w1/decisions",
                    json={"anchorIdx": 0, "action": "pick", "pickIdx": 2})
    assert r.status_code == 422
    r = client.post("/api/walks/w1/decisions",
                    json={"anchorIdx": 0, "action": "pick", "pickIdx": -1})
    assert r.status_code == 422


def test_drop_ignores_pick_index(client, tmp_path):
    """Indexing rows[pickIdx] on a drop is a 500 at 999 and the last row at -1."""
    walk(tmp_path)
    r = client.post("/api/walks/w1/decisions",
                    json={"anchorIdx": 0, "action": "drop", "pickIdx": 999})
    assert r.status_code == 200
    (entry,) = client.get("/api/walks/w1/overrides").json()
    assert (entry["action"], entry["pick"]) == ("drop", None)


def test_pick_dropped_after_reselect(client, tmp_path):
    """A pick outside its group would export one frame under two names."""
    d = walk(tmp_path)
    ok = client.post("/api/walks/w1/decisions",
                     json={"anchorIdx": 0, "action": "pick", "pickIdx": 1})
    assert ok.status_code == 200
    picks = {g["anchor"]["idx"]: g["pick"]
             for g in client.get("/api/walks/w1").json()["groups"]}
    assert picks == {0: 1, 2: 2}

    # y045_00001 promoted to its own anchor
    (d / "manifest.csv").write_text(MANIFEST.replace(
        "1,1.0,45,y045_00001.jpg,0,0,0.9700,40.0",
        "1,1.0,45,y045_00001.jpg,1,,,40.0"))
    picks = {g["anchor"]["idx"]: g["pick"]
             for g in client.get("/api/walks/w1").json()["groups"]}
    assert picks == {0: 0, 1: 1, 2: 2}


def test_list_counts_drop(client, tmp_path):
    walk(tmp_path)
    client.post("/api/walks/w1/decisions", json={"anchorIdx": 0, "action": "drop"})
    (w,) = client.get("/api/walks").json()
    assert (w["faces"], w["kept"]) == (3, 1)


def test_delete_clears_summary_cache(client, tmp_path):
    walk(tmp_path)
    client.get("/api/walks")
    assert "w1" in server._summary
    assert client.delete("/api/walks/w1").status_code == 200
    assert "w1" not in server._summary
    assert client.get("/api/walks").json() == []


def test_rerun_unknown_field_422(client, tmp_path):
    walk(tmp_path)
    r = client.post("/api/walks/w1/rerun", json={"dedup": {"taus": 0.5}})
    assert r.status_code == 422
    r = client.patch("/api/walks/w1/meta", json={"sites": "x"})
    assert r.status_code == 422


def test_face_urls_are_escaped(client, tmp_path):
    walk(tmp_path, "a walk&b")
    groups = client.get("/api/walks/a%20walk%26b").json()["groups"]
    assert groups[0]["anchor"]["url"] == "/api/walks/a%20walk%26b/faces/y045_00000.jpg"


def png(path, value=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((512, 512), value, dtype=np.uint8))


def test_mask_thumbnail_png_indices(client, tmp_path):
    d = walk(tmp_path)
    png(d / "masks" / "y045_00000.png")
    r = client.get("/api/walks/w1/masks/y045_00000.png?w=256")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    small = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert small.shape == (256, 256) and set(np.unique(small).tolist()) == {5}
    assert client.get("/api/walks/w1/masks/y045_00000.png?w=300").status_code == 422


def test_image_routes_confined(client, tmp_path):
    """A calibration face is served from calib/, not from anywhere under the walk."""
    d = walk(tmp_path)
    png(d / "masks" / "m.png")
    png(d / "calib" / "c.png")
    assert client.get("/api/walks/w1/calib/c.png").status_code == 200
    for folder, name in (("calib", "../masks/m.png"), ("masks", "../calib/c.png"),
                         ("calib", "."), ("masks", "nope.png")):
        with pytest.raises(HTTPException) as e:
            server.image(d / folder, name, d / "thumbs", None, "x")
        assert e.value.status_code == 404


def test_torn_face_404(client, tmp_path):
    d = walk(tmp_path)
    (d / "faces").mkdir()
    (d / "faces" / "y045_00000.jpg").write_bytes(b"not a jpeg")
    assert client.get("/api/walks/w1/faces/y045_00000.jpg?w=256").status_code == 404


def test_corrupt_sidecar_list_survives(client, tmp_path):
    d = walk(tmp_path)
    (d / "error.json").write_text("{not json")
    (d / "params.json").write_text("{not json")
    (d / "runs.jsonl").write_text('{"anchors": 1}\n{oops\n')
    (w,) = client.get("/api/walks").json()
    assert w["error"] is None
    detail = client.get("/api/walks/w1").json()
    assert detail["runs"] == [{"anchors": 1}]
    assert detail["pipeline"]["dedup"]["rule"] == "calibrated"


def test_corrupt_job_file_startup(client, tmp_path):
    d = walk(tmp_path)
    (d / "job.json").write_text("{not json")
    assert client.get("/api/walks").status_code == 200


def test_unknown_walk_404(client):
    for r in (client.get("/api/walks/nope"),
              client.delete("/api/walks/nope"),
              client.get("/api/walks/nope/faces/y045_00000.jpg")):
        assert r.status_code == 404


def test_jobs_lists_runner_jobs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Runner, "_run_one", lambda self, job: None)
    assert client.get("/api/jobs").json() == []
    server.runner().submit(tmp_path / "videos" / "v.insv")
    assert [j["walkId"] for j in client.get("/api/jobs").json()] == ["v"]
    assert json.loads((tmp_path / "walks" / "v" / "job.json").read_text())
