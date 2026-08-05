"""The segmentation stage: what it records and what it must never do.

It annotates the frames Select kept. It does not choose them, so a wrong mask
costs an annotator a correction rather than costing the dataset a candidate.
"""

import cv2
import numpy as np
import pytest

from accio.core.params import SEGMENTERS, PipelineParams, SegmentParams
from accio.core.segment import (MASK_DIR, MIN_MARGIN, OpenVocabSegmenter, paint,
                                prompt_spans, shares, write_mask)
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


# --- the open-vocabulary path ----------------------------------------------

def box(n=4):
    return [[0.5, 0.5, 0.2, 0.2]] * n


def stub(classes, tokens):
    """An OpenVocabSegmenter with its spans set by hand, so the scoring can be
    tested without downloading a detector."""
    import torch
    s = OpenVocabSegmenter(SegmentParams(classes=tuple(classes), threshold=0.2))
    s.torch = torch
    s._tokens = tokens
    return s


def detection(rows, tokens_wide=8):
    """Fake Grounding DINO output: one row of per-token probabilities per
    query, as logits so the model's own sigmoid recovers them."""
    import torch
    p = torch.tensor(rows, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    logits = torch.log(p / (1 - p)).unsqueeze(0)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]] * len(rows)).unsqueeze(0)
    return type("Det", (), {"logits": logits, "pred_boxes": boxes})()


def test_a_class_is_scored_on_its_own_tokens_not_on_the_longest_name():
    """The old rule took whichever class name the decoded string contained
    most of, so "concrete column" (15 chars) beat "concrete beam" (13) every
    time the model lit up tokens from both."""
    s = stub(["concrete column", "concrete beam"], [[0, 1], [0, 2]])
    #                 token 0 shared "concrete", 1 = column, 2 = beam
    det = detection([[0.3, 0.1, 0.9, 0.0]])       # the evidence is on "beam"
    got = s._named(det, (100, 100))
    assert [c for _s, _b, c in got] == [2]        # concrete beam, not column


def test_a_word_two_classes_share_cannot_decide_between_them():
    """Scoring a span by its max ties them at the shared token's value and
    throws away the word that tells them apart."""
    s = stub(["concrete column", "concrete beam"], [[0, 1], [0, 2]])
    det = detection([[0.95, 0.5, 0.9, 0.0]])      # "concrete" loudest of all
    assert [c for _s, _b, c in s._named(det, (100, 100))] == [2]


def test_a_detection_the_vocabulary_cannot_name_is_dropped():
    """Two classes tied is a different answer from either of them, and writing
    one down teaches an annotator to distrust the whole class."""
    s = stub(["window opening", "door opening"], [[0, 1], [0, 2]])
    det = detection([[0.9, 0.88, 0.88, 0.0]])     # both, equally
    assert s._named(det, (100, 100)) == []

    clear = detection([[0.9, 0.9, 0.9 - 4 * MIN_MARGIN, 0.0]])
    assert [c for _s, _b, c in s._named(clear, (100, 100))] == [1]


def test_a_weak_detection_is_dropped_whatever_it_names():
    s = stub(["pipe"], [[0]])
    assert s._named(detection([[0.05, 0, 0, 0]]), (100, 100)) == []


def test_boxes_come_back_in_pixels_of_the_frame():
    s = stub(["pipe"], [[0]])
    (_score, b, _c), = s._named(detection([[0.9, 0, 0, 0]]), (200, 400))
    assert b == pytest.approx([160.0, 80.0, 240.0, 120.0])   # of a 400x200 frame


def test_the_prompt_records_where_each_class_sits_in_it():
    prompt, spans = prompt_spans(("rebar", "concrete column"))
    assert prompt == "rebar. concrete column."
    assert [prompt[a:b] for a, b in spans] == ["rebar", "concrete column"]


# --- painting order ---------------------------------------------------------

def masks_of(*shapes):
    out = []
    for y0, y1, x0, x1 in shapes:
        m = np.zeros((100, 100), dtype=bool)
        m[y0:y1, x0:x1] = True
        out.append(m)
    return np.array(out)


def test_a_small_object_inside_a_large_one_survives():
    """A confident slab covering half the frame used to paint over a conduit
    inside it, and the conduit ceased to exist rather than being mislabelled."""
    m = masks_of((0, 100, 0, 100), (40, 60, 40, 60))   # wall, then a pipe on it
    seg = paint(np.zeros((100, 100), np.int32), m, [3, 7], [0.9, 0.3])
    assert seg[50, 50] == 7          # the pipe, despite scoring a third as high
    assert seg[10, 10] == 3          # the wall everywhere else


def test_confidence_does_not_buy_precedence():
    m = masks_of((40, 60, 40, 60), (0, 100, 0, 100))   # small first this time
    seg = paint(np.zeros((100, 100), np.int32), m, [7, 3], [0.1, 0.99])
    assert seg[50, 50] == 7          # order of arrival changes nothing


def test_equal_areas_fall_back_to_confidence():
    m = masks_of((0, 50, 0, 100), (0, 50, 0, 100))
    seg = paint(np.zeros((100, 100), np.int32), m, [3, 7], [0.4, 0.8])
    assert seg[10, 10] == 7
