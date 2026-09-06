"""Cached downscaled copies of faces and masks.

The grids draw ~150 px tiles; the originals are 1024 square at ~280 KB,
which was ~38 MB per page for a 132-group walk.
"""

import threading
from pathlib import Path

import cv2

from ..core import atomic

# fixed, or a caller could fill the disk with one cache entry per width
THUMB_WIDTHS = {256}

# cv2 work runs in the request threadpool; cap it so a grid load cannot pin every core
_workers = threading.Semaphore(4)


def sized(path: Path, cache: Path, w: int | None) -> Path:
    if w is None:
        return path
    if w not in THUMB_WIDTHS:
        raise ValueError(f"thumbnail width must be one of {sorted(THUMB_WIDTHS)}")
    return thumbnail(path, cache, w)


def thumbnail(src: Path, cache: Path, width: int) -> Path:
    """A .png source is a mask: nearest-neighbour and lossless, so the class
    indices survive; anything else is a photo, area-averaged to JPEG."""
    out = cache / f"{width}" / src.name
    if out.exists() and out.stat().st_mtime_ns >= src.stat().st_mtime_ns:
        return out
    with _workers:
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(src.name)
        h = round(img.shape[0] * width / img.shape[1])
        if src.suffix.lower() == ".png":
            small = cv2.resize(img, (width, h), interpolation=cv2.INTER_NEAREST)
            encode = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            small = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            encode = [cv2.IMWRITE_JPEG_QUALITY, 82]
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic.atomically(out, lambda tmp: cv2.imwrite(str(tmp), small, encode))
    return out
