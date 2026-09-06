"""Dual-fisheye .insv -> equirect panorama JPEGs, via the Insta360 MediaSDK
container.

The SDK reads the factory calibration in the .insv and optical-flow blends the
lens seams; -enable_flowstate gyro-levels the horizon. Only the decimated
frames are exported, via -export_frame_index.

Filenames are a projection of the frame index (pano_name/pano_index); other
layers import these rather than parsing names.
"""

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .. import settings
from . import atomic
from .params import ExtractParams

PANO_PREFIX = "pano_"
PANO_EXT = ".jpg"
PANO_INDEX = "panos.json"

JPEG_EOI = b"\xff\xd9"
SDK_POLL = 2.0
SDK_STALL = 300.0            # no new frame for this long means stuck
# backstop under the stall detector; measured 0.41-0.78 s per panorama
SDK_CAP_PER_FRAME = 3.0
SDK_CAP_FLOOR = 600.0
SDK_KILL_GRACE = 10.0


def pano_name(index: int) -> str:
    return f"{PANO_PREFIX}{index:05d}{PANO_EXT}"


def pano_index(path: Path) -> int:
    return int(path.stem.removeprefix(PANO_PREFIX))


@dataclass(frozen=True)
class PanoFrame:
    index: int
    t_sec: float
    path: Path


class FfmpegNotFound(RuntimeError):
    pass


class DockerNotFound(RuntimeError):
    pass


def probe(video: Path) -> tuple[float, int, int, int]:
    """fps, frames, width, height of the raw recording."""
    if shutil.which("ffprobe") is None:
        raise FfmpegNotFound("ffprobe not on PATH; install it (brew install ffmpeg)")
    ran = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,width,height:format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True)
    if ran.returncode:
        raise RuntimeError(f"{video.name} is not a video ffprobe can read: "
                           f"{ran.stderr.strip()[:300] or 'no video stream'}")
    # ffprobe exits 0 on a container with no video stream and answers "0/0" or "N/A"
    try:
        info = json.loads(ran.stdout)
        stream = info["streams"][0]
        num, den = stream["avg_frame_rate"].split("/")
        fps = float(num) / float(den)
        return (fps, int(float(info["format"]["duration"]) * fps),
                int(stream["width"]), int(stream["height"]))
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
        raise RuntimeError(f"{video.name} has no video stream ffprobe can "
                           "measure; is it a video?") from e


def probe_fps_nframes(video: Path) -> tuple[float, int]:
    fps, frames, _w, _h = probe(video)
    return fps, frames


# Insta360 appends a trailer after the MP4 ending in this marker. Its records
# are protobuf-shaped: field 1 serial, field 2 model, field 3 firmware.
TRAILER_MAGIC = b"8db42d694ccc418790edff439fe026bf"
TRAILER_SCAN = 2_000_000     # the record sits ~2 KB from the end


def camera(video: Path) -> dict:
    """Model, serial and firmware from the .insv trailer, or {}."""
    size = video.stat().st_size
    with open(video, "rb") as f:
        f.seek(-min(size, TRAILER_SCAN), 2)
        tail = f.read()
    if not tail.endswith(TRAILER_MAGIC):
        return {}
    # the model anchors the record: the tags alone match gyro data all over the trailer
    at = re.search(rb"\x12([\x01-\x40])(Insta360 [ -~]{0,32})", tail)
    if at is None:
        return {}
    model = tail[at.end(1):at.end(1) + at.group(1)[0]]
    if not re.fullmatch(rb"[ -~]+", model):
        return {}
    out = {"model": model.decode()}

    def field(tag: bytes, start: int) -> str:
        """Length-delimited string at `start` if its tag matches, else ""."""
        if start < 0 or tail[start:start + 1] != tag:
            return ""
        n = tail[start + 1]
        value = tail[start + 2:start + 2 + n]
        return value.decode() if re.fullmatch(rb"[ -~]+", value or b"") else ""

    # the serial ends where the model's tag begins; the firmware starts after the model
    for n in range(1, 33):
        start = at.start() - n - 2
        if tail[start:start + 1] == b"\x0a" and tail[start + 1] == n:
            out["serial"] = field(b"\x0a", start)
            break
    out["firmware"] = field(b"\x1a", at.end(1) + at.group(1)[0])
    return {k: v for k, v in out.items() if v}


# VID_20260728_114811_...: the camera's local clock. The container's
# creation_time is UTC, so the name wins when it parses.
NAME_STAMP = re.compile(r"VID_(\d{8})_(\d{6})_")


def recorded_at(video: Path) -> str:
    """When the walk was shot, "YYYY-MM-DDTHH:MM", local if the name says so."""
    m = NAME_STAMP.search(video.name)
    if m:
        d, t = m.group(1), m.group(2)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True).stdout.strip()
    return out[:16] if out else ""


def lenses_in_frame(width: int, height: int) -> int:
    """2 for circles side by side (3840x1920), 1 for one circle per file (2880x2880).

    A square frame is half a recording; stitched alone, the back hemisphere is a smear.
    """
    return 2 if width >= height * 1.5 else 1


def frame_numbers(native_fps: float, nframes: int, target_fps: float) -> list[int]:
    """Frame numbers to export at target_fps; t_sec of each is frame_no/native_fps."""
    step = max(1, round(native_fps / target_fps))
    return list(range(0, max(nframes - 1, 1), step))


def lens_files(video: Path) -> list[Path]:
    """Newer X-series cameras write the back lens to a _10_ sibling of the _00_ file."""
    if "_00_" in video.name:
        sib = video.with_name(video.name.replace("_00_", "_10_"))
        if sib.exists():
            return [video, sib]
    return [video]


def front_lens(paths: list[Path]) -> Path:
    """The _00_ file of a dual-file recording, else the single file."""
    return next((p for p in paths if "_00_" in p.name), paths[0])


def sdk_cmd(video: Path, pano_dir: Path, frame_nos: list[int], cname: str,
            params: ExtractParams) -> list[str]:
    # the example binary's -inputs parser reads args until the next dash, so it
    # cannot be last
    inputs = [f"/in/{p.name}" for p in lens_files(video)]
    # host paths: the host's daemon cannot see inside our container (see accio.settings)
    return ["docker", "run", "--rm", "--platform=linux/amd64",
            "--name", cname,
            "-v", f"{settings.host_path(video.parent)}:/in:ro",
            "-v", f"{settings.host_path(pano_dir)}:/outdir",
            "--entrypoint", "MediaSDKTest", params.sdk_image,
            "-model_root_dir", "/opt/models",
            "-inputs", *inputs,
            "-stitch_type", "optflow",
            "-enable_flowstate",
            "-output_size", f"{params.pano_width}x{params.pano_height}",
            "-enable_soft_decode", "-disable_cuda",
            "-image_sequence_dir", "/outdir", "-image_type", "jpg",
            "-export_frame_index", "-".join(map(str, frame_nos))]


def whole_jpeg(path: Path) -> bool:
    """Ends with the end-of-image marker; a killed container leaves truncated files.

    A truncated frame that passed on existence alone surfaced as cv2.imread
    returning None three stages later.
    """
    try:
        with open(path, "rb") as f:
            if f.seek(0, 2) < 128:
                return False
            f.seek(-2, 2)
            return f.read(2) == JPEG_EOI
    except OSError:
        return False


def run_sdk(cmd: list[str], cname: str, done, stall: float, cap: float) -> str:
    """Run the container until the frames land, it exits, or it stalls.
    Returns "" on success, else why it failed.

    Waits on the frames, not the process: the SDK often writes every frame and
    then hangs in emulation teardown.
    """
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        started = time.monotonic()
        progress, seen, gave_up = started, -1, False
        while True:
            try:
                proc.wait(timeout=SDK_POLL)
                break                                  # exited on its own
            except subprocess.TimeoutExpired:
                pass
            landed = done()
            if landed is True:
                break                                  # every frame is in
            if landed != seen:
                progress, seen = time.monotonic(), landed
            now = time.monotonic()
            if now - progress > stall or now - started > cap:
                gave_up = True
                break
        if proc.poll() is None:
            subprocess.run(["docker", "kill", cname], capture_output=True)
            try:
                proc.wait(timeout=SDK_KILL_GRACE)
            except subprocess.TimeoutExpired:
                # the kill did not take; waiting longer would undo the deadline
                proc.kill()
                proc.wait()
        err.seek(0)
        tail = err.read().decode("utf-8", "replace").strip()[-600:]

    if done() is True:
        return ""
    if gave_up:
        # returncode carries our own kill signal by now
        return "MediaSDK stopped making progress" + (f": {tail}" if tail else "")
    if proc.returncode:
        # Docker Desktop not running exits 125
        return f"MediaSDK exited {proc.returncode}" + (f": {tail}" if tail else "")
    return "MediaSDK exited cleanly without exporting every frame" + (
        f": {tail}" if tail else "")


def stitch(video: Path, pano_dir: Path, params: ExtractParams) -> list[PanoFrame]:
    """Stitch the .insv in the MediaSDK container and return panoramas in order."""
    if shutil.which("docker") is None:
        raise DockerNotFound("docker not on PATH; the MediaSDK stitcher runs in "
                             f"a container ({params.sdk_image})")
    pano_dir.mkdir(parents=True, exist_ok=True)
    native_fps, nframes = probe_fps_nframes(video)
    frame_nos = frame_numbers(native_fps, nframes, params.fps)

    def digit_jpgs() -> dict[int, Path]:
        return {int(p.stem): p for p in pano_dir.glob("*.jpg")
                if p.stem.isdigit() and whole_jpeg(p)}

    def landed():
        """True once every requested frame is in, else the count so far."""
        have = set(digit_jpgs())
        return True if set(frame_nos) <= have else len(have)

    # a prior killed run may have left a complete export
    if landed() is not True:
        cname = "sdk-" + re.sub(r"[^a-zA-Z0-9_.-]", "", video.stem)
        cap = SDK_CAP_PER_FRAME * len(frame_nos) + SDK_CAP_FLOOR
        for attempt in (1, 2):
            # a container still dying holds the name
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
            why = run_sdk(sdk_cmd(video, pano_dir, frame_nos, cname, params),
                          cname, landed, SDK_STALL, cap)
            if not why:
                break
            if attempt == 2:
                # no output at all looks the same for an unreadable video and
                # an empty mount
                mount = (settings.mount_check(params.sdk_image)
                         if not digit_jpgs() else "")
                raise RuntimeError(
                    f"{why}\nExported {len(digit_jpgs())}/{len(frame_nos)} "
                    f"frames for {video.name}."
                    + (f"\n{mount}" if mount else ""))

    # an earlier run at another fps leaves digit-named frames this one did not ask for
    wanted = set(frame_nos)
    for p in pano_dir.glob("*.jpg"):
        if p.stem.isdigit() and int(p.stem) not in wanted:
            p.unlink()
    exported = digit_jpgs()
    frames = []
    for index, frame_no in enumerate(sorted(exported)):
        path = pano_dir / pano_name(index)
        exported[frame_no].rename(path)
        frames.append(PanoFrame(index=index, t_sec=frame_no / native_fps, path=path))
    save_panos(pano_dir, frames)
    return frames


def save_panos(pano_dir: Path, frames: list[PanoFrame]) -> None:
    """Record each panorama's source timestamp; calibration times its
    reference frames off it."""
    atomic.write_json(pano_dir / PANO_INDEX,
                      [{"index": f.index, "tSec": f.t_sec} for f in frames])


def load_panos(pano_dir: Path) -> list[PanoFrame]:
    """The panoramas a previous stitch left behind, in walk order."""
    recorded = json.loads((pano_dir / PANO_INDEX).read_text())
    frames = [PanoFrame(index=r["index"], t_sec=r["tSec"],
                        path=pano_dir / pano_name(r["index"])) for r in recorded]
    missing = [f.path.name for f in frames if not f.path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} stitched panoramas are gone "
                                f"(first: {missing[0]})")
    return frames
