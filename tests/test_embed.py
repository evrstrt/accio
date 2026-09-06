import numpy as np
import pytest

from accio.core.embed import Embedder, TimmEmbedder, l2_normalise
from accio.core.params import EmbedParams


class FakeEmbedder:
    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        return l2_normalise(np.array([[img.mean(), 1.0] for img in images],
                                     dtype=np.float32))


def test_l2_normalise_unit_rows():
    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
    out = l2_normalise(x)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def test_l2_normalise_rejects_zero_rows():
    with pytest.raises(ValueError):
        l2_normalise(np.zeros((2, 4)))


def test_fake_embedder_satisfies_protocol():
    embedder: Embedder = FakeEmbedder()
    out = embedder.embed([np.full((4, 4, 3), 10, np.uint8),
                          np.full((4, 4, 3), 200, np.uint8)])
    assert out.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def test_failed_load_leaves_unloaded(monkeypatch):
    """The `_model is None` guard must not pass with mean and std unset."""
    import timm

    class Model:
        def eval(self):
            return self

        def to(self, _device):
            return self

    monkeypatch.setattr(timm, "create_model", lambda *a, **k: Model())
    monkeypatch.setattr(timm.data, "resolve_model_data_config",
                        lambda _m: (_ for _ in ()).throw(RuntimeError("no config")))
    e = TimmEmbedder(EmbedParams(device="cpu"))
    with pytest.raises(RuntimeError, match="no config"):
        e.embed([np.zeros((4, 4, 3), np.uint8)])
    assert e._model is None
