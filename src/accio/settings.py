"""Paths, and how to name them to the host's Docker daemon.

The stitcher runs as a sibling container: accio shells out to `docker run`,
which talks to the host's daemon, so every -v path in that command is
resolved on the host. Docker does not fail on a bind mount whose source is
missing; it mounts an empty directory. A containerised accio that passes its
own /app/data path therefore gets a stitch that exports nothing.

Either mount the data root at the same path inside and out (what compose
does; host_path is then the identity), or set ACCIO_HOST_DATA to the host
side.
"""

import os
import subprocess
from pathlib import Path

DATA_ROOT = Path(os.environ.get("ACCIO_DATA", "data")).resolve()
HOST_DATA_ROOT = Path(os.environ.get("ACCIO_HOST_DATA", str(DATA_ROOT)))

PROBE_MOUNT = "/probe"

WEB_DIST = Path(os.environ.get("ACCIO_WEB_DIST",
                               Path(__file__).resolve().parents[2] / "web" / "dist"))

# must be a volume in a container, or every restart re-downloads ~1.2 GB
MODEL_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

# a half-hour dual-lens 5.7K walk is ~20 GB across both files
MAX_UPLOAD_BYTES = int(float(os.environ.get("ACCIO_MAX_UPLOAD_GB", "40")) * 2**30)


def host_path(path: Path) -> Path:
    """The path a sibling container's bind mount has to use for `path`."""
    if HOST_DATA_ROOT == DATA_ROOT:
        return path
    return HOST_DATA_ROOT / path.resolve().relative_to(DATA_ROOT)


def mount_check(image: str, timeout: float = 30.0) -> str:
    """Ask a container what it sees at our data root. "" if it agrees with us.

    Run when a stitch exports nothing: an unreadable video and an empty mount
    look identical from here.
    """
    try:
        out = subprocess.run(
            ["docker", "run", "--rm", "--platform=linux/amd64",
             "-v", f"{HOST_DATA_ROOT}:{PROBE_MOUNT}:ro",
             "--entrypoint", "ls", image, PROBE_MOUNT],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not probe the data mount: {e}"
    if out.returncode:
        return (f"the data root is not mountable as {HOST_DATA_ROOT}: "
                f"{out.stderr.strip()[:300]}")
    seen = set(out.stdout.split())
    ours = {p.name for p in DATA_ROOT.iterdir()} if DATA_ROOT.exists() else set()
    if seen & ours:
        return ""
    return (f"a container mounting {HOST_DATA_ROOT} sees {sorted(seen) or 'nothing'}, "
            f"but {DATA_ROOT} holds {sorted(ours) or 'nothing'}. The stitcher runs "
            f"on the host's daemon, so it needs the host's path: set "
            f"ACCIO_HOST_DATA, or mount the data root at the same path inside "
            f"and out.")
