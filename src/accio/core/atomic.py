"""Writes that are either the old file or the new one, never half of either.

Every artefact a walk produces is read by request threads while the worker is
writing it, and a truncate-then-stream write is visible in between. A poll that
lands mid-rewrite of manifest.csv parses a short but syntactically valid CSV,
so the walk reports wrong counts, and a decision validated against it lands on
a different frame. A kill in the same window leaves that file on disk for good,
and manifest.csv is the only record of what the gate and faces stages produced,
so losing it does not lose the selection, it loses the walk.

os.replace is atomic within a filesystem, so writing beside the target and
renaming over it closes both the crash and the concurrent-read case at once.
The temp file sits in the same directory precisely so the rename cannot cross
a device boundary.
"""

import json
import os
from pathlib import Path
from typing import Callable


def atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Run `write` against a temp path, then move it into place.

    The temp file is cleaned up if `write` raises, so a failed run does not
    leave litter next to the artefact it failed to produce. It keeps the
    target's suffix because some writers read it: np.savez appends .npz to any
    name that does not already end in one, and would silently write beside the
    file we are about to rename.
    """
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text(path: Path, text: str) -> None:
    atomically(path, lambda tmp: tmp.write_text(text))


def write_json(path: Path, payload, **dumps) -> None:
    write_text(path, json.dumps(payload, **dumps))
