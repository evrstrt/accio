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

    def __init__(self, params: SegmentParams):
        self.params = params
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
        for image in images:
            batch = self._proc(images=image, return_tensors="pt", **self._task)
            with self.torch.no_grad():
                pred = self._model(**to_device(batch, self._device))
            # the mask has to line up with the frame it annotates, so the
            # processor scales it back to the frame's own size
            seg = self._proc.post_process_semantic_segmentation(
                pred, target_sizes=[image.shape[:2]])[0]
            out.append(seg.cpu().numpy().astype(np.int32))
        return np.stack(out), self._labels


class OpenVocabSegmenter:
    """Classes as text: Grounding DINO finds what a phrase names, SAM turns
    each box into a mask.

    This finds things rather than covering the frame, so the mask has a
    background where nothing was named. Overlapping detections are painted
    weakest first, leaving the most confident one on top.
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
        self._index = {c: i for i, c in self._labels.items()}

    def _phrase(self, said: str) -> int:
        """Grounding DINO answers in token spans, so it merges neighbouring
        phrases ("window opening door opening"). The class is whichever of
        ours the span contains the most of."""
        best, score = 0, 0
        for cls, i in self._index.items():
            if i and cls in said and len(cls) > score:
                best, score = i, len(cls)
        return best

    def segment(self, images: list[np.ndarray]) -> tuple[np.ndarray, dict[int, str]]:
        from PIL import Image

        if self._model is None:
            self._load()
        prompt = ". ".join(self.params.classes) + "."
        out = []
        for image in images:
            pil = Image.fromarray(image)
            batch = self._gp(images=pil, text=prompt, return_tensors="pt")
            with self.torch.no_grad():
                det = self._gd(**to_device(batch, self._device))
            found = self._gp.post_process_grounded_object_detection(
                det, threshold=self.params.threshold, text_threshold=0.2,
                target_sizes=[image.shape[:2]])[0]
            seg = np.zeros(image.shape[:2], dtype=np.int32)
            said = found.get("text_labels", found.get("labels", []))
            boxes = found["boxes"].cpu().tolist()
            keep = [(float(s), b, self._phrase(str(t)))
                    for s, b, t in zip(found["scores"], boxes, said)]
            keep = [k for k in keep if k[2]]
            if keep:
                keep.sort(key=lambda k: k[0])          # weakest painted first
                si = self._sp(pil, input_boxes=[[k[1] for k in keep]],
                              return_tensors="pt")
                with self.torch.no_grad():
                    so = self._sam(**to_device(si, self._device),
                                   multimask_output=False)
                masks = self._sp.image_processor.post_process_masks(
                    so.pred_masks.cpu(), si["original_sizes"].cpu(),
                    si["reshaped_input_sizes"].cpu())[0][:, 0].numpy()
                for m, (_s, _b, cls) in zip(masks, keep):
                    seg[m] = cls
            out.append(seg)
        return np.stack(out), self._labels
