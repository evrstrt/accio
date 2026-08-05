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

from accio.core.export import (build_zip, effective_picks, export_name,
                               name_prefix, review_counts, slug, stamp_exif)
from accio.core.params import PipelineParams

ROWS = [  # two kept anchors; row 1 is absorbed into 0, and 0 is the sharper
    {"pano_idx": "0", "t_sec": "0.5", "yaw": "45", "path": "y045_00000.jpg",
     "kept": "1", "anchor": "", "cosine": "", "sharpness": "90.0",
     "pick": "y045_00000.jpg"},
    {"pano_idx": "1", "t_sec": "1.0", "yaw": "45", "path": "y045_00001.jpg",
     "kept": "0", "anchor": "0", "cosine": "0.9700", "sharpness": "40.0",
     "pick": ""},
    {"pano_idx": "2", "t_sec": "1.5", "yaw": "135", "path": "y135_00002.jpg",
     "kept": "1", "anchor": "", "cosine": "", "sharpness": "70.0",
     "pick": "y135_00002.jpg"},
]

# the same walk where the frame that arrived first was the smeared one, which
# is the ordinary case: the operator enters a bay mid-turn, then settles
BLURRY_ANCHOR = [dict(r) for r in ROWS]
BLURRY_ANCHOR[0] |= {"sharpness": "12.0", "pick": "y045_00001.jpg"}
BLURRY_ANCHOR[1] |= {"sharpness": "88.0"}


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


def test_the_default_pick_is_the_sharpest_member_not_the_anchor():
    """The frame the group exports is the one worth labelling, and the anchor
    is only the frame that happened to arrive first."""
    assert [p for _, p, _ in effective_picks(BLURRY_ANCHOR, {})] == [1, 2]
    # the group is still anchored where it was, so review decisions still land
    assert [a for a, _, _ in effective_picks(BLURRY_ANCHOR, {})] == [0, 2]


def test_a_human_still_overrides_the_sharpest():
    assert [p for _, p, _ in effective_picks(BLURRY_ANCHOR, SWAP)] == [1, 2]
    back = {"y045_00000.jpg": {"pick": "y045_00000.jpg", "dropped": False}}
    assert [p for _, p, _ in effective_picks(BLURRY_ANCHOR, back)] == [0, 2]


def test_an_override_is_disagreeing_with_the_machine_not_with_the_anchor():
    """The auto-pick moved, so agreeing with it is not an override and putting
    it back to the anchor is."""
    assert review_counts(BLURRY_ANCHOR, SWAP) == (0, 0)      # same as auto
    back = {"y045_00000.jpg": {"pick": "y045_00000.jpg", "dropped": False}}
    assert review_counts(BLURRY_ANCHOR, back) == (0, 1)


def test_effective_picks_report_the_anchor_they_came_from():
    """What lets the export say which frames a human moved."""
    assert [(a, p) for a, p, _ in effective_picks(ROWS, SWAP)] == [(0, 1), (2, 2)]


def test_effective_picks_survive_a_renumbered_manifest():
    """The point of keying on names: shift every row and the swap holds."""
    shifted = [dict(r, pano_idx=str(int(r["pano_idx"]) + 1)) for r in ROWS[::-1]]
    picks = effective_picks(shifted, SWAP)
    assert [r["path"] for _, _, r in picks] == ["y135_00002.jpg", "y045_00001.jpg"]


def test_review_counts_separate_drops_from_swaps():
    assert review_counts(ROWS, {}) == (0, 0)
    assert review_counts(ROWS, SWAP) == (0, 1)
    assert review_counts(ROWS, DROP) == (1, 0)


def test_a_decision_on_a_face_that_is_no_longer_an_anchor_does_not_count():
    """A re-run can move the anchors. The log keeps the click; the funnel
    must not, or it reports a drop the export cannot show."""
    stale = {"y045_00001.jpg": {"pick": None, "dropped": True}}   # now absorbed
    assert review_counts(ROWS, stale) == (0, 0)


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


def written(tmp_path, state, decisions=None):
    faces = tmp_path / "faces"
    faces.mkdir(exist_ok=True)
    for r in ROWS:
        (faces / r["path"]).write_bytes(jpg_bytes())
    data = build_zip("walk9", tmp_path, ROWS, state, {}, "dinov3",
                     PipelineParams(), decisions)
    return zipfile.ZipFile(BytesIO(data))


LOG = [{"walkId": "walk9", "anchor": "y045_00000.jpg", "action": "pick",
        "pick": "y045_00001.jpg", "createdAt": "2026-08-05 09:00:00"}]


def test_the_decision_log_leaves_with_the_frames(tmp_path):
    z = written(tmp_path, SWAP, LOG)
    d = json.loads(z.read("decisions.json"))
    assert d["overridden"] == 1 and d["dropped"] == 0
    # the walk is named once at the top, not repeated on every row
    assert d["log"] == [{"anchor": "y045_00000.jpg", "action": "pick",
                         "pick": "y045_00001.jpg",
                         "createdAt": "2026-08-05 09:00:00"}]


def test_an_untouched_walk_still_says_so(tmp_path):
    d = json.loads(written(tmp_path, {}).read("decisions.json"))
    assert d == {"walk": "walk9", "dropped": 0, "overridden": 0, "log": []}


def test_every_frame_says_whether_a_human_moved_it(tmp_path):
    z = written(tmp_path, SWAP, LOG)
    rows = list(csv.DictReader(io.StringIO(z.read("manifest.csv").decode())))
    assert [r["overridden"] for r in rows] == ["1", "0"]
    swapped = next(n for n in z.namelist() if n.startswith("frames/"))
    comment = piexif.load(z.read(swapped))["Exif"][piexif.ExifIFD.UserComment]
    assert json.loads(piexif.helper.UserComment.load(comment))["overridden"] is True


def test_the_log_does_not_break_determinism(tmp_path):
    args = ("walk9", tmp_path, ROWS, SWAP, {}, "dinov3", PipelineParams(), LOG)
    faces = tmp_path / "faces"
    faces.mkdir(exist_ok=True)
    for r in ROWS:
        (faces / r["path"]).write_bytes(jpg_bytes())
    assert build_zip(*args) == build_zip(*args)
