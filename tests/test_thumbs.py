"""Thumbnails. The grids ask for 256 px; sending the 1024 px original costs a
132-group walk about 38 MB.
"""

import cv2
import numpy as np
import pytest

from accio.jobs.thumbs import THUMB_WIDTHS, sized, thumbnail


def face(path, size=1024):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def test_thumbnail_smaller(tmp_path):
    src = face(tmp_path / "faces" / "y045_00000.jpg")
    out = thumbnail(src, tmp_path / "thumbs", 256)
    assert cv2.imread(str(out)).shape[:2] == (256, 256)
    assert out.stat().st_size < src.stat().st_size / 4


def test_thumbnail_cached(tmp_path):
    src = face(tmp_path / "faces" / "y045_00000.jpg")
    first = thumbnail(src, tmp_path / "thumbs", 256)
    stamp = first.stat().st_mtime_ns
    again = thumbnail(src, tmp_path / "thumbs", 256)
    assert again == first
    assert again.stat().st_mtime_ns == stamp


def test_rerendered_face_invalidates_thumbnail(tmp_path):
    """The comparison is in nanoseconds: a re-render within the same second
    as the thumbnail must still replace it."""
    src = face(tmp_path / "faces" / "y045_00000.jpg")
    out = thumbnail(src, tmp_path / "thumbs", 256)
    before = out.read_bytes()

    face(src, size=512)                           # a different face, same name
    assert thumbnail(src, tmp_path / "thumbs", 256).read_bytes() != before


def test_aspect_ratio_kept(tmp_path):
    src = tmp_path / "faces" / "wide.jpg"
    src.parent.mkdir(parents=True)
    cv2.imwrite(str(src), np.zeros((256, 512, 3), dtype=np.uint8))
    out = thumbnail(src, tmp_path / "thumbs", 256)
    assert cv2.imread(str(out)).shape[:2] == (128, 256)


def test_unreadable_face_named(tmp_path):
    bad = tmp_path / "faces" / "torn.jpg"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(FileNotFoundError, match="torn.jpg"):
        thumbnail(bad, tmp_path / "thumbs", 256)


def test_mask_keeps_class_indices(tmp_path):
    """A mask holds class indices; averaging or JPEG would invent classes."""
    src = tmp_path / "masks" / "y045_00000.png"
    src.parent.mkdir(parents=True)
    seg = np.zeros((1024, 1024), dtype=np.uint8)
    seg[:, 512:] = 7
    seg[::2, ::2] = 3
    cv2.imwrite(str(src), seg)
    out = thumbnail(src, tmp_path / "thumbs", 256)
    small = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert out.suffix == ".png"
    assert small.shape == (256, 256)
    assert set(np.unique(small).tolist()) <= {0, 3, 7}


def test_widths_fixed_set(tmp_path):
    """Otherwise one cache entry per width anybody types."""
    assert THUMB_WIDTHS == {256}
    src = face(tmp_path / "faces" / "y045_00000.jpg")
    assert sized(src, tmp_path / "thumbs", None) == src
    with pytest.raises(ValueError, match="256"):
        sized(src, tmp_path / "thumbs", 300)
