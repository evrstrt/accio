"""Export invariants: naming, review-state resolution, EXIF payload,
byte-identical re-export."""

import csv
import io
import json
import zipfile
from io import BytesIO

import cv2
import numpy as np
import piexif
import piexif.helper

from accio.core.export import (MANIFEST_FIELDS, effective_picks, export_name,
                               name_prefix, review_counts, slug, stamp_exif,
                               write_zip)
from accio.core.params import PipelineParams

ROWS = [  # two kept anchors; row 1 is absorbed into 0, and 0 is the sharper
    {"pano_idx": "0", "t_sec": "0.5", "yaw": "45", "path": "y045_00000.jpg",
     "kept": "1", "anchor": "", "cosine": "", "sharpness": "90.0"},
    {"pano_idx": "1", "t_sec": "1.0", "yaw": "45", "path": "y045_00001.jpg",
     "kept": "0", "anchor": "0", "cosine": "0.9700", "sharpness": "40.0"},
    {"pano_idx": "2", "t_sec": "1.5", "yaw": "135", "path": "y135_00002.jpg",
     "kept": "1", "anchor": "", "cosine": "", "sharpness": "70.0"},
]


def test_slug_and_names():
    assert slug("Tower 2 (East)!") == "tower-2-east"
    assert name_prefix("walk9", {"site": "GCMR", "building": "tower 2"}) == \
        "gcmr_tower-2_walk9"
    assert name_prefix("walk9", {}) == "walk9"
    assert export_name("gcmr_t2_w9", 12.5, 90) == "gcmr_t2_w9_t0012.5_y090.jpg"


SWAP = {"y045_00000.jpg": {"pick": "y045_00001.jpg", "dropped": False}}
DROP = {"y135_00002.jpg": {"pick": None, "dropped": True}}


def test_effective_picks_defaults_swaps_and_drops():
    assert [p for _, p, _ in effective_picks(ROWS, {})] == [0, 2]
    assert [p for _, p, _ in effective_picks(ROWS, SWAP)] == [1, 2]
    assert [p for _, p, _ in effective_picks(ROWS, DROP)] == [0]


def test_default_pick_is_anchor():
    assert [(a, p) for a, p, _ in effective_picks(ROWS, {})] == [(0, 0), (2, 2)]


def test_human_pick_overrides_anchor():
    assert [p for _, p, _ in effective_picks(ROWS, SWAP)] == [1, 2]
    back = {"y045_00000.jpg": {"pick": "y045_00000.jpg", "dropped": False}}
    assert [p for _, p, _ in effective_picks(ROWS, back)] == [0, 2]


def test_picking_anchor_not_override():
    same = {"y045_00000.jpg": {"pick": "y045_00000.jpg", "dropped": False}}
    assert review_counts(ROWS, same) == (0, 0)
    assert review_counts(ROWS, SWAP) == (0, 1)


def test_effective_picks_report_anchor():
    assert [(a, p) for a, p, _ in effective_picks(ROWS, SWAP)] == [(0, 1), (2, 2)]


def test_effective_picks_survive_renumbering():
    """write_manifest renumbers the anchor column, so the reversed fixture points
    y045_00001 at its anchor's new index."""
    shifted = [dict(r, pano_idx=str(int(r["pano_idx"]) + 1)) for r in ROWS[::-1]]
    shifted[1]["anchor"] = "2"
    picks = effective_picks(shifted, SWAP)
    assert [r["path"] for _, _, r in picks] == ["y135_00002.jpg", "y045_00001.jpg"]


def test_pick_left_group_falls_back():
    """Honouring a pick that left its group would export one frame under two names."""
    regrouped = [dict(r) for r in ROWS]
    regrouped[1].update(kept="1", anchor="", cosine="")   # y045_00001 promoted
    picks = effective_picks(regrouped, SWAP)
    assert [(a, p) for a, p, _ in picks] == [(0, 0), (1, 1), (2, 2)]
    assert review_counts(regrouped, SWAP) == (0, 0)


def test_pick_moved_group_falls_back():
    regrouped = [dict(r) for r in ROWS]
    regrouped[1]["anchor"] = "2"                          # absorbed elsewhere
    assert [(a, p) for a, p, _ in effective_picks(regrouped, SWAP)] == \
        [(0, 0), (2, 2)]
    assert review_counts(regrouped, SWAP) == (0, 0)


def test_review_counts_separate_drops_from_swaps():
    assert review_counts(ROWS, {}) == (0, 0)
    assert review_counts(ROWS, SWAP) == (0, 1)
    assert review_counts(ROWS, DROP) == (1, 0)


def test_decision_on_demoted_anchor_ignored():
    stale = {"y045_00001.jpg": {"pick": None, "dropped": True}}   # now absorbed
    assert review_counts(ROWS, stale) == (0, 0)


def jpg_bytes() -> bytes:
    ok, arr = cv2.imencode(".jpg", np.full((16, 16, 3), 128, dtype=np.uint8))
    assert ok
    return arr.tobytes()


def zipped(*args, **kw) -> bytes:
    buf = BytesIO()
    write_zip(buf, *args, **kw)
    return buf.getvalue()


def test_stamp_exif_roundtrip_no_recompression():
    payload = {"walk": "w", "t_sec": 0.5}
    stamped = stamp_exif(jpg_bytes(), payload)
    comment = piexif.load(stamped)["Exif"][piexif.ExifIFD.UserComment]
    assert json.loads(piexif.helper.UserComment.load(comment)) == payload
    orig = cv2.imdecode(np.frombuffer(jpg_bytes(), np.uint8), cv2.IMREAD_COLOR)
    after = cv2.imdecode(np.frombuffer(stamped, np.uint8), cv2.IMREAD_COLOR)
    assert (orig == after).all()


def test_zipped_contents_and_determinism(tmp_path):
    faces = tmp_path / "faces"
    faces.mkdir()
    for r in ROWS:
        (faces / r["path"]).write_bytes(jpg_bytes())
    meta = {"site": "GCMR", "building": "t2", "shotDate": "2026-08-01"}
    args = ("walk9", tmp_path, ROWS, {}, meta, "dinov3", PipelineParams())

    data = zipped(*args)
    z = zipfile.ZipFile(BytesIO(data))
    names = z.namelist()
    assert "frames/gcmr_t2_walk9_t0000.5_y045.jpg" in names
    assert "frames/gcmr_t2_walk9_t0001.5_y135.jpg" in names
    assert "manifest.csv" in names and "walk.json" in names

    walk = json.loads(z.read("walk.json"))
    assert walk["frames"] == 2
    assert walk["pipeline"]["dedup"]["tau"] == 0.94
    assert walk["pipeline"]["embed_model_used"] == "dinov3"

    assert zipped(*args) == data


def written(tmp_path, state, decisions=None):
    faces = tmp_path / "faces"
    faces.mkdir(exist_ok=True)
    for r in ROWS:
        (faces / r["path"]).write_bytes(jpg_bytes())
    data = zipped("walk9", tmp_path, ROWS, state, {}, "dinov3",
                     PipelineParams(), decisions)
    return zipfile.ZipFile(BytesIO(data))


LOG = [{"walkId": "walk9", "anchor": "y045_00000.jpg", "action": "pick",
        "pick": "y045_00001.jpg", "createdAt": "2026-08-05 09:00:00"}]


def test_decision_log_exported(tmp_path):
    z = written(tmp_path, SWAP, LOG)
    d = json.loads(z.read("decisions.json"))
    assert d["overridden"] == 1 and d["dropped"] == 0
    # walkId is not repeated on every row
    assert d["log"] == [{"anchor": "y045_00000.jpg", "action": "pick",
                         "pick": "y045_00001.jpg",
                         "createdAt": "2026-08-05 09:00:00"}]


def test_empty_export_manifest_header(tmp_path):
    def header(z):
        return z.read("manifest.csv").decode().splitlines()[0]

    everything = written(tmp_path, {})
    nothing = written(tmp_path, {r["path"]: {"pick": None, "dropped": True}
                                 for r in ROWS})
    assert header(nothing) == header(everything) == ",".join(MANIFEST_FIELDS)
    assert json.loads(nothing.read("walk.json"))["frames"] == 0


def test_untouched_walk_logged(tmp_path):
    d = json.loads(written(tmp_path, {}).read("decisions.json"))
    assert d == {"walk": "walk9", "dropped": 0, "overridden": 0, "log": []}


def test_every_frame_logs_human_flag(tmp_path):
    z = written(tmp_path, SWAP, LOG)
    rows = list(csv.DictReader(io.StringIO(z.read("manifest.csv").decode())))
    assert [r["overridden"] for r in rows] == ["1", "0"]
    swapped = next(n for n in z.namelist() if n.startswith("frames/"))
    comment = piexif.load(z.read(swapped))["Exif"][piexif.ExifIFD.UserComment]
    assert json.loads(piexif.helper.UserComment.load(comment))["overridden"] is True


def test_log_deterministic(tmp_path):
    args = ("walk9", tmp_path, ROWS, SWAP, {}, "dinov3", PipelineParams(), LOG)
    faces = tmp_path / "faces"
    faces.mkdir(exist_ok=True)
    for r in ROWS:
        (faces / r["path"]).write_bytes(jpg_bytes())
    assert zipped(*args) == zipped(*args)
