"""Write-then-rename, so readers see the old file or the new one.

Request threads read manifest.csv while the worker rewrites it, and a kill
mid-write would lose the only record of what the gate and faces stages
produced.
"""

import json
import os
from pathlib import Path
from typing import Callable


def atomically(path: Path, write: Callable[[Path], None]) -> None:
    # keeps the suffix: np.savez appends .npz to a name without one
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        write(tmp)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text(path: Path, text: str) -> None:
    atomically(path, lambda tmp: tmp.write_text(text))


def write_json(path: Path, payload, **dumps) -> None:
    write_text(path, json.dumps(payload, **dumps))
