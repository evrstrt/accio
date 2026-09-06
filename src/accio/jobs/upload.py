"""Staging an uploaded recording before it becomes a walk."""

import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from ..core import extract

COPY_CHUNK = 8 * 2**20


class UploadTooLarge(Exception):
    pass


class VolumeFull(Exception):
    pass


class NotARecording(ValueError):
    pass


def stage_upload(files: list[tuple[str, BinaryIO]], into: Path, cap: int,
                 headroom: int) -> list[Path]:
    """Copy (name, stream) pairs into a fresh directory under `into`.

    The caller removes that directory. Staged outside the videos directory
    until probed, so a rejected upload cannot overwrite an original.
    """
    into.mkdir(parents=True, exist_ok=True)
    hold = Path(tempfile.mkdtemp(dir=into))
    try:
        saved = []
        copied = 0
        for name, stream in files:
            dst = hold / name
            with open(dst, "wb") as f:
                while chunk := stream.read(COPY_CHUNK):
                    copied += len(chunk)
                    # a chunked upload carries no Content-Length to check up front
                    if copied > cap:
                        raise UploadTooLarge(copied)
                    if shutil.disk_usage(hold).free < headroom:
                        raise VolumeFull(hold)
                    f.write(chunk)
            saved.append(dst)
        return saved
    except BaseException:
        shutil.rmtree(hold, ignore_errors=True)
        raise


def check_recording(saved: list[Path]) -> Path:
    """The front lens file, once ffprobe reads it and the lens count adds up."""
    video = extract.front_lens(saved)
    try:
        _fps, _n, width, height = extract.probe(video)
    except extract.FfmpegNotFound:
        raise
    except RuntimeError as e:
        raise NotARecording(str(e)) from e
    # a square frame is one fisheye circle: a lone _00_ stitches to a smeared half sphere
    if extract.lenses_in_frame(width, height) == 1 and len(saved) < 2:
        raise NotARecording(f"{video.name} is {width}x{height}: one lens of a "
                            "two-file recording. Upload its _10_ file "
                            "alongside it.")
    return video
