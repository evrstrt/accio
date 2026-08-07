"""Every artefact is either the old file or the new one.

Request threads read these while the worker writes them, and a kill lands
wherever it lands. Both cases have the same fix, so both are pinned here.
"""

import csv
import json

import numpy as np
import pytest

from accio.core import atomic
from accio.core.dedup import greedy_dedup
from accio.core.params import DedupParams
from accio.jobs.pipeline import write_manifest


def test_a_failed_write_keeps_the_old_file_and_leaves_no_litter(tmp_path):
    """The truncate-then-stream version loses the file for good here."""
    target = tmp_path / "manifest.csv"
    target.write_text("the previous run\n")
    with pytest.raises(RuntimeError):
        atomic.atomically(target, lambda _t: (_ for _ in ()).throw(
            RuntimeError("disk full")))
    assert target.read_text() == "the previous run\n"
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.csv"]


def test_a_half_written_file_is_never_visible_under_the_real_name(tmp_path):
    """What a poll landing mid-rewrite used to see."""
    target = tmp_path / "calibration.json"
    seen = []

    def write(tmp):
        tmp.write_text('{"tau": ')          # a torn write, mid-value
        seen.append(target.exists())        # nothing at the real name yet
        tmp.write_text('{"tau": 0.93}')

    atomic.atomically(target, write)
    assert seen == [False]
    assert json.loads(target.read_text()) == {"tau": 0.93}


def test_the_temp_file_sits_beside_its_target(tmp_path):
    """os.replace is only atomic within a filesystem, so the temp file cannot
    live in /tmp while the target lives on the data volume."""
    where = []

    def write(tmp):
        where.append(tmp.parent)
        tmp.write_text("{}")

    atomic.atomically(tmp_path / "params.json", write)
    assert where == [tmp_path]


def test_the_manifest_is_written_atomically(tmp_path):
    """It is the only record of what the gate and faces stages produced, so a
    torn one does not lose the selection, it loses the walk."""
    target = tmp_path / "manifest.csv"
    target.write_text("previous\n")
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = greedy_dedup(emb, DedupParams(tau=0.94), np.array([10.0, 20.0]))
    faces = [(0, 0.0, 45, "y045_00000.jpg"), (1, 0.5, 45, "y045_00001.jpg")]

    write_manifest(target, faces, result, [10.0, 20.0])
    rows = list(csv.DictReader(target.open()))
    assert [r["path"] for r in rows] == [f[3] for f in faces]
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_json_round_trips(tmp_path):
    atomic.write_json(tmp_path / "x.json", {"a": [1, 2]}, indent=1)
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": [1, 2]}


def test_the_temp_file_keeps_the_suffix_its_writer_may_read(tmp_path):
    """np.savez appends .npz to any name that lacks one, so a temp file called
    .embeddings.npz.tmp gets written to .embeddings.npz.tmp.npz and the rename
    finds nothing."""
    target = tmp_path / "embeddings.npz"
    atomic.atomically(target, lambda tmp: np.savez(tmp, embeddings=np.zeros(3)))
    with np.load(target) as z:
        assert z["embeddings"].shape == (3,)
    assert not list(tmp_path.glob(".*"))
