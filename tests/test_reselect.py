"""Re-select: a new threshold over the embeddings already on disk.

The properties that matter are that the face rows survive untouched (their
names are what review decisions are keyed by) and that only the grouping
columns change.
"""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from accio.core.params import DedupParams, PipelineParams, from_dict
from accio.jobs.pipeline import reselect

# four faces on a line: neighbours are close, the ends are far apart
ANGLES = [0.0, 0.15, 0.30, 1.2]
NAMES = ["y045_00000.jpg", "y135_00000.jpg", "y045_00001.jpg", "y135_00001.jpg"]


def make_walk(tmp_path):
    vecs = np.array([[np.cos(a), np.sin(a)] for a in ANGLES], dtype=np.float32)
    np.savez(tmp_path / "embeddings.npz", embeddings=vecs, model="stub")
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pano_idx", "t_sec", "yaw", "path", "kept", "anchor", "cosine"])
        for i, name in enumerate(NAMES):
            w.writerow([i, f"{i * 0.5:.1f}", name[1:4], name, 1, "", ""])
    return tmp_path


def read(tmp_path):
    with open(tmp_path / "manifest.csv") as f:
        return list(csv.DictReader(f))


def test_threshold_changes_grouping_not_rows(tmp_path):
    out = make_walk(tmp_path)
    loose = replace(PipelineParams(), dedup=DedupParams(tau=0.90))
    counts = reselect(out, loose)
    rows = read(out)

    # every face is still there, in order, under the same name
    assert [r["path"] for r in rows] == NAMES
    assert counts["faces"] == 4
    # cos(0.30) = 0.955 > 0.90, so the first three collapse into one group
    assert counts["anchors"] == 2 and counts["absorbed"] == 2
    assert [r["kept"] for r in rows] == ["1", "0", "0", "1"]
    assert rows[1]["anchor"] == "0" and float(rows[1]["cosine"]) > 0.9


def test_tighter_threshold_keeps_more(tmp_path):
    out = make_walk(tmp_path)
    tight = replace(PipelineParams(), dedup=DedupParams(tau=0.98))
    assert reselect(out, tight)["anchors"] == 3   # only cos(0.15) = 0.989 absorbs
    # and the walk's own settings land next to its frames, so the manifest and
    # the tau that produced it can never drift apart
    saved = from_dict(json.loads((out / "params.json").read_text()))
    assert saved.dedup.tau == 0.98


def test_reselect_rejects_a_mismatched_manifest(tmp_path):
    out = make_walk(tmp_path)
    with open(out / "manifest.csv", "a", newline="") as f:
        csv.writer(f).writerow([9, "9.0", 45, "y045_00009.jpg", 1, "", ""])
    with pytest.raises(ValueError, match="5 rows"):
        reselect(out, PipelineParams())
