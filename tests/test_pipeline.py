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
from accio.core.params import FaceParams, GateParams, PipelineParams
from accio.jobs.pipeline import run_walk

N_PANOS = 3
PARAMS = PipelineParams(
    faces=FaceParams(size=32, yaws=(0, 180)),   # 2 faces per pano -> 6 rows
    gate=GateParams(window=1),                  # keep every pano
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
    embedder = StubEmbedder()
    expected = embedder.rows.copy()

    run_walk(tmp_path / "walk.insv", tmp_path / "out", PARAMS, embedder,
             progress=lambda *_: None)
    walk_dir = tmp_path / "out" / "walk"

    with open(walk_dir / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    saved = np.load(walk_dir / "embeddings.npz")

    # one embedding row per manifest row, in the same order, model recorded
    assert len(rows) == N_PANOS * 2
    assert saved["embeddings"].shape[0] == len(rows)
    assert np.allclose(saved["embeddings"], expected)
    assert str(saved["model"]) == PARAMS.embed.model_name

    # the duplicate embedding was absorbed by its anchor, everything else kept
    assert [r["kept"] for r in rows] == ["1"] * 5 + ["0"]
    assert rows[-1]["anchor"] == "0"
    assert float(rows[-1]["cosine"]) == 1.0

    # manifest rows follow capture order: pano major, yaw minor
    assert [(r["pano_idx"], r["yaw"]) for r in rows] == [
        (str(p), str(y)) for p in range(N_PANOS) for y in (0, 180)]
