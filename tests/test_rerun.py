"""Re-running from a stage rebuilds it and everything below; the stitch stands."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from accio.core import extract
from accio.core.extract import PanoFrame, pano_name
from accio.core.params import (DedupParams, FaceParams, GateParams,
                               PipelineParams)
from accio.jobs import pipeline
from accio.jobs.pipeline import rerun, run_walk

N_PANOS = 4
BLURRED = 2                                     # the one smeared panorama
PARAMS = PipelineParams(
    faces=FaceParams(size=32, yaws=(45, 225)),
    gate=GateParams(dead=0.0),                  # veto nothing
    # fixed, so a test moving tau tests tau: the calibration stub would overwrite it
    dedup=DedupParams(rule="fixed", solo_floor=0.0),
)
CALIB = {"tau": 0.94, "falseMergePct": 1.0, "farSeconds": 20.0, "samples": 10,
         "pairs": [], "healthy": True, "referenceError": "",
         "far": {"n": 200, "median": 0.6, "p95": 0.9, "p99": 0.94, "max": 0.95,
                 "cosines": []},
         "reference": {"median": 0.98, "p05": 0.96, "min": 0.95, "n": 8},
         "gapSeconds": 0.033}


def fake_stitch(video, pano_dir, params):
    rng = np.random.default_rng(0)
    pano_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(N_PANOS):
        path = pano_dir / pano_name(i)
        img = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
        if i == BLURRED:
            img = cv2.GaussianBlur(img, (9, 9), 3)
        cv2.imwrite(str(path), img)
        frames.append(PanoFrame(index=i, t_sec=i * 15 / 29.97, path=path))
    extract.save_panos(pano_dir, frames)
    return frames


class StubEmbedder:
    """Unit rows far enough apart that nothing absorbs, whatever the face count."""

    def __init__(self):
        self.rng = np.random.default_rng(1)

    def embed(self, images):
        v = self.rng.normal(size=(len(images), 16)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def walk(tmp_path, monkeypatch):
    """An ingested walk on disk, with Docker and the SDK stubbed out."""
    monkeypatch.setattr(extract, "stitch", fake_stitch)
    monkeypatch.setattr(extract, "probe", lambda v: (29.97, 300, 3840, 1920))
    monkeypatch.setattr(extract, "lens_files", lambda v: [v])
    monkeypatch.setattr(pipeline, "calibrate", lambda *a, **k: CALIB)
    video = tmp_path / "walk.insv"
    run_walk(video, tmp_path / "out", PARAMS, StubEmbedder())
    return video, tmp_path / "out" / "walk"


def stamps(d):
    return {p.name: p.stat().st_mtime_ns for p in sorted(d.glob("*.jpg"))}


def names(d):
    return sorted(p.name for p in d.glob("*.jpg"))


def test_rerun_from_faces_reuses_stitch(walk):
    video, out = walk
    before = stamps(out / "pano")

    seen = []
    wider = replace(PARAMS, faces=replace(PARAMS.faces, yaws=(30, 150, 270)))
    rerun(video, out, wider, StubEmbedder(), "faces",
          progress=lambda s, st, **c: seen.append((s, st)))

    assert stamps(out / "pano") == before
    assert seen[:3] == [("video", "done"), ("stitch", "done"), ("gate", "done")]
    assert [s for s, st in seen if st == "running"] == [
        "faces", "embed", "calibrate", "select"]


def test_rerun_from_faces_clears_stale_frames(walk):
    video, out = walk
    assert len(names(out / "faces")) == N_PANOS * 2

    wider = replace(PARAMS, faces=replace(PARAMS.faces, yaws=(30, 150, 270)))
    rerun(video, out, wider, StubEmbedder(), "faces")

    on_disk = names(out / "faces")
    assert len(on_disk) == N_PANOS * 3
    assert all(n.startswith(("y030", "y150", "y270")) for n in on_disk)
    rows = pipeline.manifest_rows(out)
    with np.load(out / "embeddings.npz") as data:
        assert len(rows) == len(data["embeddings"]) == len(on_disk)
    assert sorted(r["path"] for r in rows) == on_disk


def test_rerun_from_gate_rescores(walk):
    """A stricter dead threshold vetoes the smeared panorama and only that one."""
    video, out = walk
    strict = replace(PARAMS, gate=GateParams(dead=0.40))
    rerun(video, out, strict, StubEmbedder(), "gate")

    rows = pipeline.manifest_rows(out)
    survivors = {int(r["pano_idx"]) for r in rows}
    assert survivors == {0, 1, 3}
    assert len(rows) == 3 * 2
    assert names(out / "faces") == sorted(r["path"] for r in rows)


def test_rerun_from_select_keeps_faces(walk):
    video, out = walk
    before = stamps(out / "faces")
    kept = sum(r["kept"] == "1" for r in pipeline.manifest_rows(out))
    assert kept == N_PANOS * 2      # nothing absorbs at the default threshold

    loose = replace(PARAMS, dedup=replace(PARAMS.dedup, tau=0.2))
    # select does not read the original, so the route passes none
    rerun(None, out, loose, StubEmbedder(), "select")

    assert stamps(out / "faces") == before
    assert sum(r["kept"] == "1" for r in pipeline.manifest_rows(out)) < kept


def test_rerun_refuses_stitch_stage(walk):
    video, out = walk
    with pytest.raises(ValueError, match="re-ingest"):
        rerun(video, out, PARAMS, StubEmbedder(), "stitch")
