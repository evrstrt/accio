"""The segmentation stage annotates the frames Select kept; it never chooses them."""

import cv2
import numpy as np
import pytest

from accio.core.params import SEGMENTERS, PipelineParams, SegmentParams
from accio.core.segment import (MASK_DIR, MIN_MARGIN, OpenVocabSegmenter, paint,
                                prompt_spans, shares, write_mask)

LABELS = {0: "wall", 3: "floor", 5: "ceiling", 14: "door"}


def frame(counts: dict[int, int]) -> np.ndarray:
    """A 100x100 mask made of the given class -> pixel counts, rest class 0."""
    flat = np.zeros(10_000, dtype=np.int32)
    at = 0
    for cls, n in counts.items():
        flat[at:at + n] = cls
        at += n
    return flat.reshape(100, 100)


def test_shares_largest_first():
    s = shares(frame({5: 3000, 3: 1000}), LABELS)   # rest stays class 0
    assert list(s) == ["wall", "ceiling", "floor"]
    assert s == {"wall": 0.6, "ceiling": 0.3, "floor": 0.1}


def test_sliver_class_omitted():
    s = shares(frame({5: 5000, 14: 20}), LABELS)   # 4980 left as class 0
    assert "door" not in s                          # 0.2% of the frame
    assert list(s) == ["ceiling", "wall"]           # 5000 then 4980


def test_unnamed_class_gets_row():
    s = shares(frame({5: 5000, 99: 2000}), LABELS)
    assert s["99"] == 0.2


def test_mask_round_trip(tmp_path):
    """PNG: a lossy codec would invent classes."""
    seg = frame({5: 3000, 14: 1000})
    write_mask(tmp_path / "m.png", seg)
    back = cv2.imread(str(tmp_path / "m.png"), cv2.IMREAD_UNCHANGED)
    assert back.shape == seg.shape
    assert set(np.unique(back).tolist()) == {0, 5, 14}


def test_stage_off_by_default():
    assert PipelineParams().segment.enabled is False
    assert SegmentParams().model_name in SEGMENTERS
    assert MASK_DIR == "masks"


def box(n=4):
    return [[0.5, 0.5, 0.2, 0.2]] * n


def stub(classes, tokens):
    """An OpenVocabSegmenter with its spans set by hand."""
    import torch
    s = OpenVocabSegmenter(SegmentParams(classes=tuple(classes), threshold=0.2))
    s.torch = torch
    s._tokens = tokens
    return s


def detection(rows, tokens_wide=8):
    """Fake Grounding DINO output: per-token probabilities as logits."""
    import torch
    p = torch.tensor(rows, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    logits = torch.log(p / (1 - p)).unsqueeze(0)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]] * len(rows)).unsqueeze(0)
    return type("Det", (), {"logits": logits, "pred_boxes": boxes})()


def test_class_scored_on_own_tokens():
    """Tokens lit for both classes must not hand the win to the longer class name."""
    s = stub(["concrete column", "concrete beam"], [[0, 1], [0, 2]])
    #                 token 0 shared "concrete", 1 = column, 2 = beam
    det = detection([[0.3, 0.1, 0.9, 0.0]])       # the evidence is on "beam"
    got = s._named(det, (100, 100))
    assert [c for _s, _b, c in got] == [2]


def test_shared_token_does_not_decide():
    s = stub(["concrete column", "concrete beam"], [[0, 1], [0, 2]])
    det = detection([[0.95, 0.5, 0.9, 0.0]])      # "concrete" loudest of all
    assert [c for _s, _b, c in s._named(det, (100, 100))] == [2]


def test_unnamed_detection_dropped():
    s = stub(["window opening", "door opening"], [[0, 1], [0, 2]])
    det = detection([[0.9, 0.88, 0.88, 0.0]])     # both, equally
    assert s._named(det, (100, 100)) == []

    clear = detection([[0.9, 0.9, 0.9 - 4 * MIN_MARGIN, 0.0]])
    assert [c for _s, _b, c in s._named(clear, (100, 100))] == [1]


def test_weak_detection_dropped():
    s = stub(["pipe"], [[0]])
    assert s._named(detection([[0.05, 0, 0, 0]]), (100, 100)) == []


def test_boxes_in_frame_pixels():
    s = stub(["pipe"], [[0]])
    (_score, b, _c), = s._named(detection([[0.9, 0, 0, 0]]), (200, 400))
    assert b == pytest.approx([160.0, 80.0, 240.0, 120.0])   # of a 400x200 frame


def test_prompt_records_class_spans():
    prompt, spans = prompt_spans(("rebar", "concrete column"))
    assert prompt == "rebar. concrete column."
    assert [prompt[a:b] for a, b in spans] == ["rebar", "concrete column"]


def masks_of(*shapes):
    out = []
    for y0, y1, x0, x1 in shapes:
        m = np.zeros((100, 100), dtype=bool)
        m[y0:y1, x0:x1] = True
        out.append(m)
    return np.array(out)


def test_small_object_inside_large_survives():
    """A confident slab must not paint over the conduit inside it."""
    m = masks_of((0, 100, 0, 100), (40, 60, 40, 60))   # wall, then a pipe on it
    seg = paint(np.zeros((100, 100), np.int32), m, [3, 7], [0.9, 0.3])
    assert seg[50, 50] == 7
    assert seg[10, 10] == 3


def test_confidence_no_precedence():
    m = masks_of((40, 60, 40, 60), (0, 100, 0, 100))   # small first this time
    seg = paint(np.zeros((100, 100), np.int32), m, [7, 3], [0.1, 0.99])
    assert seg[50, 50] == 7


def test_equal_areas_fall_back_to_confidence():
    m = masks_of((0, 50, 0, 100), (0, 50, 0, 100))
    seg = paint(np.zeros((100, 100), np.int32), m, [3, 7], [0.4, 0.8])
    assert seg[10, 10] == 7


def test_class_index_over_byte_refused(tmp_path):
    with pytest.raises(ValueError, match="300"):
        write_mask(tmp_path / "m.png", frame({300: 10}))
