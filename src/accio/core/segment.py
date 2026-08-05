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
# a class has to cover this much of a frame before it is worth naming in the
# manifest; below it the row becomes a list of single-pixel noise
MIN_SHARE = 0.01


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


class SegformerSegmenter:
    """transformers SegFormer, loaded lazily so importing this stays cheap."""

    def __init__(self, params: SegmentParams):
        self.params = params
        self._model = None

    def _load(self):
        import torch
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)

        self.torch = torch
        self._proc = SegformerImageProcessor.from_pretrained(self.params.model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(self.params.model_name)
        self._device = self.params.device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
        self._model = model.eval().to(self._device)
        self._labels = {int(k): v for k, v in model.config.id2label.items()}

    def segment(self, images: list[np.ndarray]) -> tuple[np.ndarray, dict[int, str]]:
        if self._model is None:
            self._load()
        out = []
        for image in images:
            batch = self._proc(images=image, return_tensors="pt").to(self._device)
            with self.torch.no_grad():
                logits = self._model(**batch).logits
            # the model works at its own resolution; the mask has to line up
            # with the frame it annotates, so it is scaled back before argmax
            up = self.torch.nn.functional.interpolate(
                logits, size=image.shape[:2], mode="bilinear", align_corners=False)
            out.append(up.argmax(1)[0].cpu().numpy().astype(np.int32))
        return np.stack(out), self._labels
