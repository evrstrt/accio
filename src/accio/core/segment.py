"""Semantic segmentation of the kept frames, as pre-annotation.

ADE20K is the only public label set with an interior vocabulary (wall, floor,
ceiling, door, windowpane, column, stairs, railing). Measured on GCMR and ASHV
(Aug 2026) the structure holds on bare RCC; a debris-covered floor reads
partly as wall.

This annotates, it never selects. Masks are single-channel PNGs of class
indices.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .params import SegmentParams

MASK_DIR = "masks"
SAM_MODEL = "facebook/sam-vit-huge"
MIN_SHARE = 0.01     # a class covering less of a frame is not named in the manifest
MIN_MARGIN = 0.05    # the best class must beat the runner-up by this to be named
MAX_TEXT = 256       # Grounding DINO's text positions; classes past it never score
SEG_BATCH = 4        # frames per forward pass; they are 1024 square


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
    """Fraction of the frame per class, largest first, under `floor` dropped."""
    total = seg.size
    counts = Counter(seg.flatten().tolist())
    out = {labels.get(i, str(i)): round(n / total, 4)
           for i, n in counts.most_common() if n / total >= floor}
    return out


def write_mask(path: Path, seg: np.ndarray) -> None:
    if seg.max() >= 256:
        raise ValueError(f"class index {int(seg.max())} does not fit an 8-bit mask")
    cv2.imwrite(str(path), seg.astype(np.uint8))


def pick_device(want: str | None) -> str:
    import torch
    return want or ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")


def to_device(batch, device: str):
    """MPS has no float64; several processors emit it for boxes and target sizes."""
    import torch
    for k, v in list(batch.items()):
        if torch.is_tensor(v) and v.dtype == torch.float64:
            batch[k] = v.float()
    return batch.to(device)


class SemanticSegmenter:
    """SegFormer, Mask2Former or OneFormer; one output shape. Weights load lazily."""

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
        for chunk in batched(images, self.batch):
            task = {"task_inputs": ["semantic"] * len(chunk)} if self._task else {}
            batch = self._proc(images=list(chunk), return_tensors="pt", **task)
            with self.torch.no_grad():
                pred = self._model(**to_device(batch, self._device))
            segs = self._proc.post_process_semantic_segmentation(
                pred, target_sizes=[im.shape[:2] for im in chunk])
            out += [s.cpu().numpy().astype(np.int32) for s in segs]
        return np.stack(out), self._labels


def prompt_spans(classes: tuple[str, ...]) -> tuple[str, list[tuple[int, int]]]:
    """The Grounding DINO prompt and each class's character span in it.

    Character spans because the tokenizer reports offsets in characters.
    """
    prompt, spans, at = "", [], 0
    for c in classes:
        prompt += c + ". "
        spans.append((at, at + len(c)))
        at += len(c) + 2
    return prompt.strip(), spans


class OpenVocabSegmenter:
    """Grounding DINO scores boxes against text classes; SAM turns each box
    into a mask. Pixels nothing named stay class 0.

    Classes are scored against their own token spans rather than through
    post_process_grounded_object_detection, which decodes every token over
    threshold into one label string ("window opening door opening") and takes
    the score as a max over all 256 text positions.
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
        self._labels = {0: "unlabelled"}
        self._labels.update(dict(enumerate(self.params.classes, start=1)))
        self._prompt, self._tokens = self._spans()

    def _spans(self) -> tuple[str, list[list[int]]]:
        """Token positions of each class, in prompt order."""
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
        """(score, box, class) per detection over threshold.

        A class scores the mean over its tokens, not the max: under max,
        "concrete column" and "concrete beam" tie on "concrete".
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
    """Largest mask first, so a nested object (a conduit on a wall) stays visible.

    Mask area, not box area: a diagonal pipe has a huge box and few pixels.
    Score breaks ties only.
    """
    order = sorted(range(len(classes)),
                   key=lambda i: (-int(masks[i].sum()), scores[i]))
    for i in order:
        seg[masks[i]] = classes[i]
    return seg
