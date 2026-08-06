"""Where the data lives, and how to name it to a daemon outside this process.

One knob today (ACCIO_DATA) and one that only matters once accio is itself
containerised (ACCIO_HOST_DATA), but they belong together because the second
exists entirely to correct the first.

The stitcher is a sibling container, not a child: accio shells out to `docker
run`, which talks to the host's daemon over its socket, so every -v path in
that command is resolved on the *host*, not in accio's filesystem. Bind mounts
are the one thing Docker will not fail on when it cannot find the source. It
creates an empty directory instead. So a containerised accio asking for
/app/data/videos gets a valid mount of nothing, the SDK starts, finds no
input, and exports zero frames with no hint that the path was the problem.

Two ways to be right, and this module supports both. Mount the data root at
the same path inside and out (ACCIO_HOST_DATA unset, host_path is identity),
which is what the compose file does and what makes every path in the process
literally true. Or mount it anywhere and set ACCIO_HOST_DATA to the host side.
"""

import os
import subprocess
from pathlib import Path

DATA_ROOT = Path(os.environ.get("ACCIO_DATA", "data")).resolve()

# What the host calls DATA_ROOT. Equal to it on bare metal, and equal to it
# under compose too, because mounting to the same path is cheaper than
# translating between two.
HOST_DATA_ROOT = Path(os.environ.get("ACCIO_HOST_DATA", str(DATA_ROOT)))

PROBE_MOUNT = "/probe"


def host_path(path: Path) -> Path:
    """The name a sibling container's bind mount has to use for `path`.

    Refuses to guess: outside the data root there is no mapping to apply, and
    returning the path unchanged would be exactly the silent empty mount this
    module exists to prevent.
    """
    if HOST_DATA_ROOT == DATA_ROOT:
        return path
    return HOST_DATA_ROOT / path.resolve().relative_to(DATA_ROOT)


def mount_check(image: str, timeout: float = 30.0) -> str:
    """Ask a container what it sees at our data root. "" if it agrees with us.

    Run when a stitch exports nothing, because that failure looks identical
    whether the video was unreadable or the mount was empty, and one of those
    is a five-second fix that could otherwise cost an afternoon.
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
                f"{out.stderr.strip()[-200:]}")
    seen = set(out.stdout.split())
    ours = {p.name for p in DATA_ROOT.iterdir()} if DATA_ROOT.exists() else set()
    if seen & ours:
        return ""
    return (f"a container mounting {HOST_DATA_ROOT} sees {sorted(seen) or 'nothing'}, "
            f"but {DATA_ROOT} holds {sorted(ours) or 'nothing'}. The stitcher runs "
            f"on the host's daemon, so it needs the host's path: set "
            f"ACCIO_HOST_DATA, or mount the data root at the same path inside "
            f"and out.")
