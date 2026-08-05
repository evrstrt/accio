"""Stage 6: what is in each kept frame, before anyone labels it.

A frozen semantic segmenter run over the frames Select kept. ADE20K is the
only public label set with the vocabulary an interior needs (wall, floor,
ceiling, door, windowpane, column, stairs, railing), and it is trained on
finished rooms, so the question was whether bare RCC carries it. Measured on
GCMR and ASHV footage (Aug 2026): the structure holds up on both, and the
weak spot is a debris-covered floor, which reads partly as wall because there
is no visual floor left.

This annotates, it never selects. A mask that is wrong should cost an
annotator a correction, not cost the dataset a frame.

Masks are written as single-channel PNGs of ADE20K class indices, not colour:
the index is the label, and a palette is a rendering choice the viewer makes.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .params import SegmentParams

MASK_DIR = "masks"
# the mask model the open-vocabulary path pairs with its detector
SAM_MODEL = "facebook/sam-vit-huge"
# a class has to cover this much of a frame before it is worth naming in the
# manifest; below it the row becomes a list of single-pixel noise
MIN_SHARE = 0.01
# how far a detection's best class has to beat its second before the name is
# worth writing down. Under it the box found something and the vocabulary
# cannot say what, which is a different answer from "nothing here".
MIN_MARGIN = 0.05
# Grounding DINO's text side is a fixed 256 positions; a longer prompt runs off
# the end and those classes would silently never score
MAX_TEXT = 256
# frames per forward pass. Small because these are 1024 square and the heads
# are large; the point is that it is more than one, which is what it was.
SEG_BATCH = 4


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


@dataclass(frozen=True)
class Segmented:
    name: str                      # the face this belongs to
    shares: dict[str, float]       # class -> fraction of the frame, descending


class Segmenter(Protocol):
    def segment(self, images: list[np.ndarray]) -> tuple[np.ndarray, dict[int, str]]:
        """RGB uint8 images -> (N, H, W) int class indices, and their names."""
        ...


def shares(seg: np.ndarray, labels: dict[int, str],
           floor: float = MIN_SHARE) -> dict[str, float]:
    """What each class covers, largest first, noise dropped."""
    total = seg.size
    counts = Counter(seg.flatten().tolist())
    out = {labels.get(i, str(i)): round(n / total, 4)
           for i, n in counts.most_common() if n / total >= floor}
    return out


def write_mask(path: Path, seg: np.ndarray) -> None:
    """Class indices, not colours: PNG is lossless so the index survives."""
    cv2.imwrite(str(path), seg.astype(np.uint8))


def pick_device(want: str | None) -> str:
    import torch
    return want or ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")


def to_device(batch, device: str):
    """MPS has no float64, and several processors emit it for boxes and target
    sizes. Casting beats dropping the whole model back to the CPU."""
    import torch
    for k, v in list(batch.items()):
        if torch.is_tensor(v) and v.dtype == torch.float64:
            batch[k] = v.float()
    return batch.to(device)


class SemanticSegmenter:
    """A fixed label set over every pixel: SegFormer, Mask2Former, OneFormer.

    They are three different heads with three different classes in
    transformers, but one output shape, so the stage above does not care which
    is loaded. Weights load lazily; importing this module stays cheap.
    """

    def __init__(self, params: SegmentParams, batch: int = SEG_BATCH):
        self.params = params
        self.batch = batch
        self._model = None

    def _load(self):
        import torch
        from transformers import AutoImageProcessor

        name = self.params.model_name
        self.torch = torch
        self._proc = AutoImageProcessor.from_pretrained(name)
        if "mask2former" in name:
            from transformers import Mask2FormerForUniversalSegmentation as M
            model = M.from_pretrained(name)
        elif "oneformer" in name:
            from transformers import OneFormerForUniversalSegmentation as M
            from transformers import OneFormerProcessor
            self._proc = OneFormerProcessor.from_pretrained(name)
            model = M.from_pretrained(name)
        else:
            from transformers import AutoModelForSemanticSegmentation as M
            model = M.from_pretrained(name)
        self._device = pick_device(self.params.device)
        self._model = model.eval().to(self._device)
        self._labels = {int(k): v for k, v in model.config.id2label.items()}
        self._task = {"task_inputs": ["semantic"]} if "oneformer" in name else {}

    def segment(self, images: list[np.ndarray]) -> tuple[np.ndarray, dict[int, str]]:
        if self._model is None:
            self._load()
        out = []
        # One forward per batch, not per image. The caller already gathers 64
        # frames and used to hand them to a loop, so the batching was cosmetic
        # and every frame paid the full launch and post-processing overhead.
        for chunk in _chunks(images, self.batch):
            task = {"task_inputs": ["semantic"] * len(chunk)} if self._task else {}
            batch = self._proc(images=list(chunk), return_tensors="pt", **task)
            with self.torch.no_grad():
                pred = self._model(**to_device(batch, self._device))
            # each mask has to line up with the frame it annotates, so the
            # processor scales them back to their own sizes
            segs = self._proc.post_process_semantic_segmentation(
                pred, target_sizes=[im.shape[:2] for im in chunk])
            out += [s.cpu().numpy().astype(np.int32) for s in segs]
        return np.stack(out), self._labels


def prompt_spans(classes: tuple[str, ...]) -> tuple[str, list[tuple[int, int]]]:
    """The prompt Grounding DINO is given, and where each class sits in it.

    Character spans, because the tokenizer reports offsets in characters and
    that is the only handle on which token belongs to which class.
    """
    prompt, spans, at = "", [], 0
    for c in classes:
        prompt += c + ". "
        spans.append((at, at + len(c)))
        at += len(c) + 2
    return prompt.strip(), spans


class OpenVocabSegmenter:
    """Classes as text: Grounding DINO scores what a phrase names, SAM turns
    each box into a mask.

    This finds things rather than covering the frame, so the mask has a
    background where nothing was named.

    Two things it deliberately does not do. It does not read the decoded label
    string: post_process_grounded_object_detection runs every token over its
    threshold through get_phrases_from_posmap and decodes them as one string,
    so a detection comes back as "window opening door opening" and the score is
    a max over all 256 text positions rather than over the class that was
    found. Instead each class is scored against its own token span, which is
    unambiguous and is the number the model actually computed.

    And it does not paint by confidence. On bare RCC the large surfaces are the
    confident detections, so painting the strongest last buried a conduit
    inside a wall and the object ceased to exist. Smallest mask wins instead,
    which is the only order that survives one thing being inside another.
    """

    def __init__(self, params: SegmentParams):
        self.params = params
        self._model = None

    def _load(self):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor, SamModel, SamProcessor)

        self.torch = torch
        self._device = pick_device(self.params.device)
        name = self.params.model_name
        self._gp = AutoProcessor.from_pretrained(name)
        self._gd = AutoModelForZeroShotObjectDetection.from_pretrained(
            name).eval().to(self._device)
        self._sp = SamProcessor.from_pretrained(SAM_MODEL)
        self._sam = SamModel.from_pretrained(SAM_MODEL).eval().to(self._device)
        self._model = True
        # class 0 is "nothing named here", so the phrases start at 1
        self._labels = {0: "unlabelled"}
        self._labels.update(dict(enumerate(self.params.classes, start=1)))
        self._prompt, self._tokens = self._spans()

    def _spans(self) -> tuple[str, list[list[int]]]:
        """Which text positions carry each class, in prompt order."""
        prompt, spans = prompt_spans(self.params.classes)
        offsets = self._gp.tokenizer(prompt, return_offsets_mapping=True,
                                     truncation=True, max_length=MAX_TEXT
                                     )["offset_mapping"]
        tokens = []
        for cls, (lo, hi) in zip(self.params.classes, spans):
            idx = [t for t, (a, b) in enumerate(offsets)
                   if b > a and a >= lo and b <= hi]
            if not idx:
                raise ValueError(
                    f"'{cls}' fell outside the {MAX_TEXT} text positions "
                    f"Grounding DINO has; use fewer or shorter classes")
            tokens.append(idx)
        return prompt, tokens

    def _named(self, det, size: tuple[int, int]):
        """Every detection as (score, box, class), scored per class.

        A class scores the mean alignment over its own tokens, not the max.
        Max is the usual convention and it is wrong the moment two classes
        share a word: "concrete column" and "concrete beam" both contain
        "concrete", so if that token lights up they tie at its value and the
        word that tells them apart never enters the number. The mean spends the
        shared evidence on both and lets the distinctive token decide.
        """
        from transformers.image_transforms import center_to_corners_format

        prob = det.logits[0].sigmoid()                       # (queries, 256)
        per_class = self.torch.stack(
            [prob[:, idx].mean(dim=1) for idx in self._tokens], dim=1)
        top2 = per_class.topk(min(2, per_class.shape[1]), dim=1)
        score = top2.values[:, 0]
        runner_up = top2.values[:, 1] if per_class.shape[1] > 1 \
            else self.torch.zeros_like(score)
        cls = top2.indices[:, 0] + 1                         # 0 is unlabelled

        h, w = size
        boxes = center_to_corners_format(det.pred_boxes[0]) * \
            self.torch.tensor([w, h, w, h], device=det.pred_boxes.device)

        keep = (score > self.params.threshold) & \
               (score - runner_up >= MIN_MARGIN)
        return [(float(s), [float(v) for v in b], int(c))
                for s, b, c in zip(score[keep], boxes[keep].cpu(), cls[keep])]

    def segment(self, images: list[np.ndarray]) -> tuple[np.ndarray, dict[int, str]]:
        from PIL import Image

        if self._model is None:
            self._load()
        out = []
        for image in images:
            pil = Image.fromarray(image)
            batch = self._gp(images=pil, text=self._prompt, return_tensors="pt")
            with self.torch.no_grad():
                det = self._gd(**to_device(batch, self._device))
            found = self._named(det, image.shape[:2])
            seg = np.zeros(image.shape[:2], dtype=np.int32)
            if found:
                si = self._sp(pil, input_boxes=[[f[1] for f in found]],
                              return_tensors="pt")
                with self.torch.no_grad():
                    so = self._sam(**to_device(si, self._device),
                                   multimask_output=False)
                masks = self._sp.image_processor.post_process_masks(
                    so.pred_masks.cpu(), si["original_sizes"].cpu(),
                    si["reshaped_input_sizes"].cpu())[0][:, 0].numpy()
                seg = paint(seg, masks, [f[2] for f in found],
                            [f[0] for f in found])
            out.append(seg)
        return np.stack(out), self._labels


def paint(seg: np.ndarray, masks: np.ndarray, classes: list[int],
          scores: list[float]) -> np.ndarray:
    """Largest mask first, so the smallest ends up on top.

    Nesting is the normal case here: a pipe runs along a slab, a conduit
    crosses a wall. Area is the only ordering that keeps the inner thing
    visible, and it has to be the mask's area rather than the box's, because a
    long diagonal pipe has a huge box and almost no pixels. Score breaks ties
    only, never precedence.
    """
    order = sorted(range(len(classes)),
                   key=lambda i: (-int(masks[i].sum()), scores[i]))
    for i in order:
        seg[masks[i]] = classes[i]
    return seg
