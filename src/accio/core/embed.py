"""Stage 4: frozen-backbone image embeddings.

The embedder sits behind a small protocol so the backbone stays swappable
(DINOv2, AnyLoc-style descriptors) without touching dedup or the store. The
default is DINOv3 ViT-B/16, CLS token: the PoC showed mean-pooled patch
tokens collapse on bare concrete (median random-pair cosine 0.95).

Preprocessing (INTER_AREA resize, mean/std from the timm config) matches the
PoC that calibrated tau = 0.94; changing it invalidates the threshold.

Embeddings are L2-normalised float32, so dot product == cosine similarity
everywhere downstream.
"""

from itertools import batched
from typing import Protocol

import cv2
import numpy as np

from .params import EmbedParams


class Embedder(Protocol):
    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """RGB uint8 images -> (N, D) L2-normalised float32."""
        ...


def l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero-norm embedding; the backbone returned garbage")
    return (x / norms).astype(np.float32)


class Dinov3Embedder:
    """timm DINOv3, loaded lazily so importing this module stays cheap."""

    def __init__(self, params: EmbedParams = EmbedParams()):
        self.params = params
        self._model = None

    def _load(self):
        import timm
        import torch

        device = self.params.device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
        model = timm.create_model(self.params.model_name, pretrained=True,
                                  num_classes=0, img_size=self.params.img_size)
        model.eval().to(device)
        cfg = timm.data.resolve_model_data_config(model)
        self._model = model
        self._device = device
        self._mean = np.array(cfg["mean"], dtype=np.float32)
        self._std = np.array(cfg["std"], dtype=np.float32)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        import torch

        if self._model is None:
            self._load()
        size = self.params.img_size
        out = []
        for batch in batched(images, self.params.batch_size):
            arr = np.stack([
                (cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                 .astype(np.float32) / 255.0 - self._mean) / self._std
                for img in batch
            ])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self._device)
            with torch.no_grad():
                tokens = self._model.forward_features(x)
            out.append(tokens[:, 0].cpu().numpy())  # CLS token
        return l2_normalise(np.concatenate(out))
