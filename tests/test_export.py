"""Export invariants: naming, review-state resolution, EXIF payload,
byte-identical re-export."""

import json
import zipfile
from io import BytesIO

import cv2
import numpy as np
import piexif
import piexif.helper

from accio.core.export import (build_zip, effective_picks, export_name,
                               name_prefix, slug, stamp_exif)
from accio.core.params import PipelineParams

ROWS = [  # two kept anchors; row 1 is absorbed into 0
    {"pano_idx": "0", "t_sec": "0.5", "yaw": "45", "path": "y045_00000.jpg",
     "kept": "1", "anchor": "", "cosine": ""},
    {"pano_idx": "1", "t_sec": "1.0", "yaw": "45", "path": "y045_00001.jpg",
     "kept": "0", "anchor": "0", "cosine": "0.9700"},
    {"pano_idx": "2", "t_sec": "1.5", "yaw": "135", "path": "y135_00002.jpg",
     "kept": "1", "anchor": "", "cosine": ""},
]


def test_slug_and_names():
    assert slug("Tower 2 (East)!") == "tower-2-east"
    assert name_prefix("walk9", {"site": "GCMR", "building": "tower 2"}) == \
        "gcmr_tower-2_walk9"
    assert name_prefix("walk9", {}) == "walk9"
    assert export_name("gcmr_t2_w9", 12.5, 90) == "gcmr_t2_w9_t0012.5_y090.jpg"


def test_effective_picks_defaults_swaps_and_drops():
    assert [p for p, _ in effective_picks(ROWS, {})] == [0, 2]
    swapped = effective_picks(ROWS, {0: {"pick": 1, "dropped": False}})
    assert [p for p, _ in swapped] == [1, 2]
    dropped = effective_picks(ROWS, {2: {"pick": None, "dropped": True}})
    assert [p for p, _ in dropped] == [0]


def jpg_bytes() -> bytes:
    ok, arr = cv2.imencode(".jpg", np.full((16, 16, 3), 128, dtype=np.uint8))
    assert ok
    return arr.tobytes()


def test_stamp_exif_roundtrip_no_recompression():
    payload = {"walk": "w", "t_sec": 0.5}
    stamped = stamp_exif(jpg_bytes(), payload)
    comment = piexif.load(stamped)["Exif"][piexif.ExifIFD.UserComment]
    assert json.loads(piexif.helper.UserComment.load(comment)) == payload
    # pixels untouched: decoded images are identical
    orig = cv2.imdecode(np.frombuffer(jpg_bytes(), np.uint8), cv2.IMREAD_COLOR)
    after = cv2.imdecode(np.frombuffer(stamped, np.uint8), cv2.IMREAD_COLOR)
    assert (orig == after).all()


def test_build_zip_contents_and_determinism(tmp_path):
    faces = tmp_path / "faces"
    faces.mkdir()
    for r in ROWS:
        (faces / r["path"]).write_bytes(jpg_bytes())
    meta = {"site": "GCMR", "building": "t2", "shotDate": "2026-08-01"}
    args = ("walk9", tmp_path, ROWS, {}, meta, "dinov3", PipelineParams())

    data = build_zip(*args)
    z = zipfile.ZipFile(BytesIO(data))
    names = z.namelist()
    assert "frames/gcmr_t2_walk9_t0000.5_y045.jpg" in names
    assert "frames/gcmr_t2_walk9_t0001.5_y135.jpg" in names
    assert "manifest.csv" in names and "walk.json" in names

    walk = json.loads(z.read("walk.json"))
    assert walk["frames"] == 2
    assert walk["pipeline"]["dedup"]["tau"] == 0.94
    assert walk["pipeline"]["embed_model_used"] == "dinov3"

    assert build_zip(*args) == data  # byte-identical re-export
