"""run_walk glue invariants: manifest.csv and embeddings.npz stay row-aligned.

The stitch is faked (no Docker in tests) and the embedder is a stub returning
orthogonal unit vectors, with the last face a copy of the first so exactly one
absorption happens at a known place.
"""

import csv

import cv2
import numpy as np

from accio.core import extract
from accio.core.extract import PanoFrame, pano_name
from accio.core.params import (DedupParams, FaceParams, GateParams,
                               PipelineParams)
from accio.jobs.pipeline import run_walk

N_PANOS = 3
PARAMS = PipelineParams(
    faces=FaceParams(size=32, yaws=(0, 180)),   # 2 faces per pano -> 6 rows
    gate=GateParams(dead=0.0),                  # keep every pano
    # three panoramas spanning one second have no pairs 20s apart, so there is
    # no far distribution to calibrate against and the default rule refuses.
    # Fixed is what that refusal names, and this test is about the glue anyway.
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

    # the source is reported once, then every stage runs and completes in order
    assert seen == [("video", "done")] + [
        (s, st) for s in ("stitch", "gate", "faces", "embed", "calibrate", "select")
        for st in ("running", "done")]

    with open(walk_dir / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    saved = np.load(walk_dir / "embeddings.npz")

    # one embedding row per manifest row, in the same order, model recorded
    assert len(rows) == N_PANOS * 2
    assert saved["embeddings"].shape[0] == len(rows)
    assert np.allclose(saved["embeddings"], expected)
    assert str(saved["model"]) == PARAMS.embed.model_name

    # the duplicate embedding was absorbed, everything else kept. Which of the
    # identical pair anchors is decided by sharpness, not by arrival, so the
    # test asks that one absorbed the other rather than naming which.
    absorbed = [i for i, r in enumerate(rows) if r["kept"] == "0"]
    assert len(absorbed) == 1 and absorbed[0] in (0, 5)
    twin = 5 if absorbed[0] == 0 else 0
    assert rows[absorbed[0]]["anchor"] == str(twin)
    assert float(rows[absorbed[0]]["cosine"]) == 1.0
    assert float(rows[twin]["sharpness"]) >= float(rows[absorbed[0]]["sharpness"])

    # manifest rows follow capture order: pano major, yaw minor
    assert [(r["pano_idx"], r["yaw"]) for r in rows] == [
        (str(p), str(y)) for p in range(N_PANOS) for y in (0, 180)]
