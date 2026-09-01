"""The routes as HTTP: validation, limits, and review state through the app.

Everything else tests the functions routes call; nothing exercised the
composition until a restart bug lived exactly there (recovery only ran when a
mutating route happened to be hit first). These go through the ASGI app.
"""

import shutil
import threading

import pytest
from fastapi.testclient import TestClient

from accio import settings
from accio.server import app as server
from accio.store import db

MANIFEST = (
    "pano_idx,t_sec,yaw,path,kept,anchor,cosine,sharpness\n"
    "0,0.5,45,y045_00000.jpg,1,,,90.0\n"
    "1,1.0,45,y045_00001.jpg,0,0,0.9700,40.0\n"
    "2,1.5,135,y135_00002.jpg,1,,,70.0\n")

META = {"site": "gcmr", "building": "t2", "floor": "3", "stage": "casco",
        "operator": "ed", "mount_height_cm": "180"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(server, "VIDEO_DIR", tmp_path / "videos")
    monkeypatch.setattr(server, "WALKS_ROOT", tmp_path / "walks")
    monkeypatch.setattr(server, "EXPORT_TMP", tmp_path / ".exports")
    monkeypatch.setattr(server, "_summary", {})
    monkeypatch.setattr(server, "_runner", None)
    monkeypatch.setattr(server, "FREE_HEADROOM", 0)
    # per-thread like the real one, but against this test's DB; the real conn
    # caches on a thread-local that outlives the tmp_path
    conns: dict[int, object] = {}

    def conn():
        t = threading.get_ident()
        if t not in conns:
            conns[t] = db.connect(tmp_path / "accio.db")
        return conns[t]

    monkeypatch.setattr(server, "conn", conn)
    with TestClient(server.app) as c:
        yield c


def walk(tmp_path, name="w1", manifest=MANIFEST):
    d = tmp_path / "walks" / name
    d.mkdir(parents=True)
    (d / "manifest.csv").write_text(manifest)
    return d


def test_health_answers(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_startup_builds_the_runner(client):
    """The recovery pass runs because the server started, not because a
    mutating route happened to be hit first."""
    assert server._runner is not None


def test_ingest_refuses_no_files(client):
    assert client.post("/api/ingest", data=META).status_code == 422


def test_ingest_names_the_missing_metadata(client):
    r = client.post("/api/ingest", data={"site": "gcmr"},
                    files=[("files", ("VID_00_1.insv", b"x"))])
    assert r.status_code == 422
    assert "building" in r.json()["detail"]


def test_ingest_refuses_the_back_lens_alone(client):
    r = client.post("/api/ingest", data=META,
                    files=[("files", ("VID_10_1.insv", b"x"))])
    assert r.status_code == 422
    assert "_00_" in r.json()["detail"]


def test_an_upload_over_the_cap_is_413_and_leaves_nothing(client, tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 16)
    r = client.post("/api/ingest", data=META,
                    files=[("files", ("VID_00_1.insv", b"0" * 64))])
    assert r.status_code == 413
    assert list((tmp_path / "videos" / ".incoming").iterdir()) == []


def test_a_nearly_full_volume_is_507_not_a_broken_store(client, monkeypatch):
    monkeypatch.setattr(server, "FREE_HEADROOM", 2**60)
    r = client.post("/api/ingest", data=META,
                    files=[("files", ("VID_00_1.insv", b"0" * 64))])
    assert r.status_code == 507


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")
def test_a_file_that_is_not_a_video_is_422_naming_it(client):
    r = client.post("/api/ingest", data=META,
                    files=[("files", ("VID_00_1.insv", b"not a video"))])
    assert r.status_code == 422
    assert "VID_00_1.insv" in r.json()["detail"]


def test_a_decision_outside_the_group_is_refused(client, tmp_path):
    walk(tmp_path)
    r = client.post("/api/walks/w1/decisions",
                    json={"anchorIdx": 0, "action": "pick", "pickIdx": 2})
    assert r.status_code == 422


def test_a_pick_shows_until_a_reselect_moves_the_face_out(client, tmp_path):
    """The stored pick survives by name, but only while the picked face is
    still in the group: after a regroup it falls back to the anchor rather
    than exporting one frame under two names."""
    d = walk(tmp_path)
    ok = client.post("/api/walks/w1/decisions",
                     json={"anchorIdx": 0, "action": "pick", "pickIdx": 1})
    assert ok.status_code == 200
    picks = {g["anchor"]["idx"]: g["pick"]
             for g in client.get("/api/walks/w1").json()["groups"]}
    assert picks == {0: 1, 2: 2}

    # the reselect: y045_00001 is promoted to its own anchor
    (d / "manifest.csv").write_text(MANIFEST.replace(
        "1,1.0,45,y045_00001.jpg,0,0,0.9700,40.0",
        "1,1.0,45,y045_00001.jpg,1,,,40.0"))
    picks = {g["anchor"]["idx"]: g["pick"]
             for g in client.get("/api/walks/w1").json()["groups"]}
    assert picks == {0: 0, 1: 1, 2: 2}


def test_an_unknown_walk_is_404_everywhere(client):
    for r in (client.get("/api/walks/nope"),
              client.delete("/api/walks/nope"),
              client.get("/api/walks/nope/faces/y045_00000.jpg")):
        assert r.status_code == 404
