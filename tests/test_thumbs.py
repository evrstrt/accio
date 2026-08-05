"""Thumbnails: the grids ask for 256 px and the file is 1024.

Sending the original meant a 132-group walk pulled about 38 MB to fill squares
that need a couple of hundred kilobytes between them.
"""

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from accio.server.app import THUMB_WIDTHS, thumbnail


def face(path, size=1024):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def test_a_thumbnail_is_smaller_in_both_pixels_and_bytes(tmp_path):
    src = face(tmp_path / "faces" / "y045_00000.jpg")
    out = thumbnail(src, tmp_path / "thumbs", 256)
    assert cv2.imread(str(out)).shape[:2] == (256, 256)
    assert out.stat().st_size < src.stat().st_size / 4


def test_it_is_built_once_and_then_read(tmp_path):
    src = face(tmp_path / "faces" / "y045_00000.jpg")
    first = thumbnail(src, tmp_path / "thumbs", 256)
    stamp = first.stat().st_mtime_ns
    again = thumbnail(src, tmp_path / "thumbs", 256)
    assert again == first
    assert again.stat().st_mtime_ns == stamp      # not regenerated


def test_a_re_rendered_face_invalidates_its_thumbnail(tmp_path):
    """render_faces clears and rewrites faces/, so a stale thumbnail would
    show the previous run's frame under this run's manifest."""
    src = face(tmp_path / "faces" / "y045_00000.jpg")
    out = thumbnail(src, tmp_path / "thumbs", 256)
    before = out.read_bytes()

    face(src, size=512)                           # a different face, same name
    import os
    os.utime(src, (out.stat().st_atime + 10, out.stat().st_mtime + 10))
    assert thumbnail(src, tmp_path / "thumbs", 256).read_bytes() != before


def test_an_aspect_ratio_is_kept(tmp_path):
    src = tmp_path / "faces" / "wide.jpg"
    src.parent.mkdir(parents=True)
    cv2.imwrite(str(src), np.zeros((256, 512, 3), dtype=np.uint8))
    out = thumbnail(src, tmp_path / "thumbs", 256)
    assert cv2.imread(str(out)).shape[:2] == (128, 256)


def test_an_unreadable_face_is_a_404_not_a_crash(tmp_path):
    bad = tmp_path / "faces" / "torn.jpg"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(HTTPException) as e:
        thumbnail(bad, tmp_path / "thumbs", 256)
    assert e.value.status_code == 404


def test_the_widths_are_a_fixed_set():
    """Otherwise one cache entry per width anybody types, on a disk already
    measured in terabytes."""
    assert THUMB_WIDTHS == {256}
