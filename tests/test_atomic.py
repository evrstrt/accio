"""Every artefact is either the old file or the new one."""

import csv
import json
import os

import numpy as np
import pytest

from accio.core import atomic
from accio.core.dedup import greedy_dedup
from accio.core.params import DedupParams
from accio.jobs.pipeline import write_manifest


def test_failed_write_keeps_target(tmp_path):
    target = tmp_path / "manifest.csv"
    target.write_text("the previous run\n")
    with pytest.raises(RuntimeError):
        atomic.atomically(target, lambda _t: (_ for _ in ()).throw(
            RuntimeError("disk full")))
    assert target.read_text() == "the previous run\n"
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.csv"]


def test_partial_write_invisible(tmp_path):
    target = tmp_path / "calibration.json"
    seen = []

    def write(tmp):
        tmp.write_text('{"tau": ')
        seen.append(target.exists())
        tmp.write_text('{"tau": 0.93}')

    atomic.atomically(target, write)
    assert seen == [False]
    assert json.loads(target.read_text()) == {"tau": 0.93}


def test_temp_file_beside_target(tmp_path):
    """os.replace is only atomic within one filesystem."""
    where = []

    def write(tmp):
        where.append(tmp.parent)
        tmp.write_text("{}")

    atomic.atomically(tmp_path / "params.json", write)
    assert where == [tmp_path]


def test_manifest_written_atomically(tmp_path):
    target = tmp_path / "manifest.csv"
    target.write_text("previous\n")
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = greedy_dedup(emb, DedupParams(tau=0.94), np.array([10.0, 20.0]))
    faces = [(0, 0.0, 45, "y045_00000.jpg"), (1, 0.5, 45, "y045_00001.jpg")]

    write_manifest(target, faces, result, [10.0, 20.0])
    rows = list(csv.DictReader(target.open()))
    assert [r["path"] for r in rows] == [f[3] for f in faces]
    assert not list(tmp_path.glob(".*.tmp"))


def test_temp_file_synced_before_rename(tmp_path, monkeypatch):
    """os.replace alone leaves the new bytes in the page cache; a crash keeps
    the name and loses the content."""
    target = tmp_path / "state.json"
    synced = []
    monkeypatch.setattr(atomic.os, "fsync", lambda fd: synced.append(
        (os.fstat(fd).st_ino, target.exists())))
    atomic.atomically(target, lambda tmp: tmp.write_text("{}"))
    assert synced == [(target.stat().st_ino, False)]


def test_write_json_round_trips(tmp_path):
    atomic.write_json(tmp_path / "x.json", {"a": [1, 2]}, indent=1)
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": [1, 2]}


def test_temp_file_keeps_suffix(tmp_path):
    """np.savez appends .npz to a name without it, so the rename would find nothing."""
    target = tmp_path / "embeddings.npz"
    atomic.atomically(target, lambda tmp: np.savez(tmp, embeddings=np.zeros(3)))
    with np.load(target) as z:
        assert z["embeddings"].shape == (3,)
    assert not list(tmp_path.glob(".*"))
