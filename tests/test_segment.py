"""The segmentation stage: what it records and what it must never do.

It annotates the frames Select kept. It does not choose them, so a wrong mask
costs an annotator a correction rather than costing the dataset a candidate.
"""

import cv2
import numpy as np
import pytest

from accio.core.params import SEGMENTERS, PipelineParams, SegmentParams
from accio.core.segment import MASK_DIR, shares, write_mask
from accio.server.app import Rerun, check, merge

LABELS = {0: "wall", 3: "floor", 5: "ceiling", 14: "door"}


def frame(counts: dict[int, int]) -> np.ndarray:
    """A 100x100 mask made of the given class -> pixel counts, rest class 0."""
    flat = np.zeros(10_000, dtype=np.int32)
    at = 0
    for cls, n in counts.items():
        flat[at:at + n] = cls
        at += n
    return flat.reshape(100, 100)


def test_shares_are_fractions_of_the_frame_largest_first():
    s = shares(frame({5: 3000, 3: 1000}), LABELS)   # rest stays class 0
    assert list(s) == ["wall", "ceiling", "floor"]
    assert s == {"wall": 0.6, "ceiling": 0.3, "floor": 0.1}


def test_a_sliver_of_a_class_is_not_worth_naming():
    """Without a floor every frame lists a tail of single-pixel noise."""
    s = shares(frame({5: 5000, 14: 20}), LABELS)   # 4980 left as class 0
    assert "door" not in s                          # 0.2% of the frame
    assert list(s) == ["ceiling", "wall"]           # 5000 then 4980


def test_an_unnamed_class_still_gets_a_row():
    s = shares(frame({5: 5000, 99: 2000}), LABELS)
    assert s["99"] == 0.2


def test_a_mask_is_class_indices_and_survives_the_round_trip(tmp_path):
    """PNG because it is lossless: the pixel value is the label, so a lossy
    codec would invent classes that were never predicted."""
    seg = frame({5: 3000, 14: 1000})
    write_mask(tmp_path / "m.png", seg)
    back = cv2.imread(str(tmp_path / "m.png"), cv2.IMREAD_UNCHANGED)
    assert back.shape == seg.shape
    assert set(np.unique(back).tolist()) == {0, 5, 14}


def test_segmentation_is_the_last_stage_so_it_re_runs_alone():
    _p, first, changed = merge(PipelineParams(), Rerun(segment={"enabled": True}))
    assert first == "segment"
    assert changed == {"segment.enabled"}


def test_a_faces_change_still_re_runs_everything_below_it():
    _p, first, _c = merge(PipelineParams(),
                          Rerun(faces={"fov_deg": 120}, segment={"enabled": True}))
    assert first == "faces"


def test_an_unlisted_segmenter_is_refused():
    with pytest.raises(Exception) as e:
        check(Rerun(segment={"model_name": "some/other-model"}))
    assert e.value.status_code == 422


def test_the_stage_is_off_until_asked_for():
    """It pulls a few hundred megabytes of weights; nobody should pay for that
    on an ingest that never wanted it."""
    assert PipelineParams().segment.enabled is False
    assert SegmentParams().model_name in SEGMENTERS
    assert MASK_DIR == "masks"
