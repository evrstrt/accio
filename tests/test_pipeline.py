"""run_walk glue: manifest.csv and embeddings.npz stay row-aligned. The stitch
is faked and the stub embedder repeats the first face as the last.
"""

import csv
import json

import cv2
import numpy as np
import pytest

from accio.core import extract
from accio.core.extract import PanoFrame, pano_name
from accio.core.params import (DedupParams, FaceParams, GateParams,
                               PipelineParams)
from accio.jobs.pipeline import MANIFEST_COLUMNS, gate, read_json, rerun, run_walk

N_PANOS = 3
PARAMS = PipelineParams(
    faces=FaceParams(size=32, yaws=(0, 180)),   # 2 faces per pano -> 6 rows
    gate=GateParams(dead=0.0),                  # keep every pano
    # three panoramas in one second have no far pairs, so the calibrated rule refuses
    dedup=DedupParams(rule="fixed"),
)


def fake_stitch(video, pano_dir, params):
    rng = np.random.default_rng(0)
    pano_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(N_PANOS):
        path = pano_dir / pano_name(i)
        cv2.imwrite(str(path), rng.integers(0, 255, (32, 64, 3), dtype=np.uint8))
        frames.append(PanoFrame(index=i, t_sec=i / 2.0, path=path))
    return frames


class StubEmbedder:
    """Orthogonal unit rows, except the last row repeats the first."""

    def __init__(self):
        self.rows = np.eye(8, dtype=np.float32)[: N_PANOS * 2]
        self.rows[-1] = self.rows[0]

    def embed(self, images):
        n = len(images)
        out, self.rows = self.rows[:n], self.rows[n:]
        return out


def test_run_walk_manifest_and_embeddings_align(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "stitch", fake_stitch)
    monkeypatch.setattr(extract, "probe", lambda v: (30.0, 300, 3840, 1920))
    monkeypatch.setattr(extract, "lens_files", lambda v: [v])
    embedder = StubEmbedder()
    expected = embedder.rows.copy()

    seen: list[tuple[str, str]] = []
    run_walk(tmp_path / "walk.insv", tmp_path / "out", PARAMS, embedder,
             progress=lambda stage, status, **c: seen.append((stage, status)))
    walk_dir = tmp_path / "out" / "walk"

    assert seen == [("video", "done")] + [
        (s, st) for s in ("stitch", "gate", "faces", "embed", "calibrate", "select")
        for st in ("running", "done")]

    with open(walk_dir / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    with np.load(walk_dir / "embeddings.npz") as saved:
        embeddings, model = saved["embeddings"], str(saved["model"])

    assert len(rows) == N_PANOS * 2
    assert embeddings.shape[0] == len(rows)
    assert np.allclose(embeddings, expected)
    assert model == PARAMS.embed.model_name

    # sharpness decides which of the identical pair anchors
    absorbed = [i for i, r in enumerate(rows) if r["kept"] == "0"]
    assert len(absorbed) == 1 and absorbed[0] in (0, 5)
    twin = 5 if absorbed[0] == 0 else 0
    assert rows[absorbed[0]]["anchor"] == str(twin)
    assert float(rows[absorbed[0]]["cosine"]) == 1.0
    assert float(rows[twin]["sharpness"]) >= float(rows[absorbed[0]]["sharpness"])

    assert [(r["pano_idx"], r["yaw"]) for r in rows] == [
        (str(p), str(y)) for p in range(N_PANOS) for y in (0, 180)]


def test_empty_stitch_named(tmp_path):
    """np.concatenate([]) on an empty walk names nothing."""
    with pytest.raises(RuntimeError, match="no panoramas"):
        gate([], PARAMS)


def empty_walk(tmp_path):
    out = tmp_path / "out" / "walk"
    out.mkdir(parents=True)
    (out / "source.json").write_text(json.dumps({"fps": 30.0}))
    (out / "manifest.csv").write_text(",".join(MANIFEST_COLUMNS) + "\n")
    return out


def test_rerun_empty_manifest_named(tmp_path):
    out = empty_walk(tmp_path)
    with pytest.raises(RuntimeError, match="no faces"):
        rerun(None, out, PARAMS, StubEmbedder(), "select")


def test_rerun_to_calibrate_needs_video(tmp_path):
    out = empty_walk(tmp_path)
    with pytest.raises(ValueError, match="needs the original video"):
        rerun(None, out, PARAMS, StubEmbedder(), "embed")


def test_rerun_corrupt_source_named(tmp_path):
    out = empty_walk(tmp_path)
    (out / "source.json").write_text("{torn")
    with pytest.raises(RuntimeError, match="source.json"):
        rerun(None, out, PARAMS, StubEmbedder(), "select")


def test_read_json_none_on_unreadable(tmp_path):
    assert read_json(tmp_path / "missing.json") is None
    (tmp_path / "torn.json").write_text("{")
    assert read_json(tmp_path / "torn.json") is None
    (tmp_path / "ok.json").write_text("[1]")
    assert read_json(tmp_path / "ok.json") == [1]
    assert read_json(tmp_path) is None
