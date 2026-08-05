"""Stage 1: dual-fisheye .insv -> decimated equirect panorama JPEGs.

The Insta360 MediaSDK container does the stitch (replaces ffmpeg v360, which
seamed from a nominal lens FOV and left parallax at yaw +-90). The SDK reads
the factory calibration in the .insv and optical-flow blends the lens seams;
-enable_flowstate additionally gyro-levels the horizon (the helmet cam tilts
with the wearer's head). Only the frames we keep are exported, via
-export_frame_index, which replaces the old fps-filter decimation and skips
stitching the ~93% of frames we would throw away.

Frame identity lives in pano_name()/pano_index(): filenames are a projection
of the frame index, never the other way around. The store and export layers
import these instead of re-parsing filenames.
"""

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .params import ExtractParams

PANO_PREFIX = "pano_"
PANO_EXT = ".jpg"
PANO_INDEX = "panos.json"


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
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,width,height:format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.split()
    width, height, rate = out[0].split(",")
    num, den = rate.split("/")
    fps = float(num) / float(den)
    return fps, int(float(out[1]) * fps), int(width), int(height)


def probe_fps_nframes(video: Path) -> tuple[float, int]:
    fps, frames, _w, _h = probe(video)
    return fps, frames


# Insta360 appends its own trailer after the MP4, ending in this marker. The
# records inside are protobuf-shaped: field 1 the serial, field 2 the model,
# field 3 the firmware. Reading them beats asking the operator to type
# "Insta360 X3" correctly, and it is the only place the camera is recorded.
TRAILER_MAGIC = b"8db42d694ccc418790edff439fe026bf"
TRAILER_SCAN = 2_000_000     # the record sits ~2 KB from the end; read wide


def camera(video: Path) -> dict:
    """Model, serial and firmware of the camera that shot this, or {}."""
    size = video.stat().st_size
    with open(video, "rb") as f:
        f.seek(-min(size, TRAILER_SCAN), 2)
        tail = f.read()
    if not tail.endswith(TRAILER_MAGIC):
        return {}
    # The three fields sit in one record, so the model anchors it: the tags
    # alone match binary gyro data all over the trailer.
    at = re.search(rb"\x12([\x01-\x40])(Insta360 [ -~]{0,32})", tail)
    if at is None:
        return {}
    model = tail[at.end(1):at.end(1) + at.group(1)[0]]
    if not re.fullmatch(rb"[ -~]+", model):
        return {}
    out = {"model": model.decode()}

    def field(tag: bytes, start: int) -> str:
        """One length-delimited string at `start`, if that is where it is."""
        if start < 0 or tail[start:start + 1] != tag:
            return ""
        n = tail[start + 1]
        value = tail[start + 2:start + 2 + n]
        return value.decode() if re.fullmatch(rb"[ -~]+", value or b"") else ""

    # the serial is the field that ends where the model's tag begins, and the
    # firmware is the one that starts after the model
    for n in range(1, 33):
        start = at.start() - n - 2
        if tail[start:start + 1] == b"\x0a" and tail[start + 1] == n:
            out["serial"] = field(b"\x0a", start)
            break
    out["firmware"] = field(b"\x1a", at.end(1) + at.group(1)[0])
    return {k: v for k, v in out.items() if v}


# VID_20260728_114811_..., the camera's own local clock. The container's
# creation_time is UTC (06:18 for this 11:48 walk), and a site cares which
# hour of its own day the walk happened, so the name wins when it parses.
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
    """How many fisheye circles a frame holds.

    Both packings are dual-fisheye; they differ in how they are stored. Two
    circles side by side make a 2:1 frame (3840x1920), one circle per file
    makes a square one (2880x2880). A square frame is therefore half a
    recording, and stitching it alone stretches the front hemisphere over the
    whole sphere: the back of every panorama comes out a smear.
    """
    return 2 if width >= height * 1.5 else 1


def frame_numbers(native_fps: float, nframes: int, target_fps: float) -> list[int]:
    """Frame numbers to export at target_fps; t_sec of each is frame_no/native_fps."""
    step = max(1, round(native_fps / target_fps))
    return list(range(0, max(nframes - 1, 1), step))


def lens_files(video: Path) -> list[Path]:
    """Newer X-series cameras write each lens to its own file: _00_ (front) has a
    _10_ (back) sibling. Older cameras (X3) pack both lenses in the one file."""
    if "_00_" in video.name:
        sib = video.with_name(video.name.replace("_00_", "_10_"))
        if sib.exists():
            return [video, sib]
    return [video]


def front_lens(paths: list[Path]) -> Path:
    """The file whose stem names the walk: the _00_ (front) file of a dual-file
    recording, or the single file of an older camera."""
    return next((p for p in paths if "_00_" in p.name), paths[0])


def sdk_cmd(video: Path, pano_dir: Path, frame_nos: list[int], cname: str,
            params: ExtractParams) -> list[str]:
    # The example binary's -inputs parser reads args until the next dash, so
    # -inputs must not be last.
    inputs = [f"/in/{p.name}" for p in lens_files(video)]
    return ["docker", "run", "--rm", "--platform=linux/amd64",
            "--name", cname,
            "-v", f"{video.parent}:/in:ro", "-v", f"{pano_dir}:/outdir",
            "--entrypoint", "MediaSDKTest", params.sdk_image,
            "-model_root_dir", "/opt/models",
            "-inputs", *inputs,
            "-stitch_type", "optflow",
            "-enable_flowstate",
            "-output_size", f"{params.pano_width}x{params.pano_height}",
            "-enable_soft_decode", "-disable_cuda",
            "-image_sequence_dir", "/outdir", "-image_type", "jpg",
            "-export_frame_index", "-".join(map(str, frame_nos))]


def stitch(video: Path, pano_dir: Path, params: ExtractParams) -> list[PanoFrame]:
    """Stitch the .insv in the MediaSDK container and return panoramas in order."""
    if shutil.which("docker") is None:
        raise DockerNotFound("docker not on PATH; the MediaSDK stitcher runs in "
                             f"a container ({params.sdk_image})")
    pano_dir.mkdir(parents=True, exist_ok=True)
    native_fps, nframes = probe_fps_nframes(video)
    frame_nos = frame_numbers(native_fps, nframes, params.fps)

    def digit_jpgs() -> dict[int, Path]:
        return {int(p.stem): p for p in pano_dir.glob("*.jpg") if p.stem.isdigit()}

    # The SDK occasionally finishes writing frames but never exits (emulation
    # teardown hang), so: hard timeout, kill the container, and count the export
    # as success if every requested frame landed. A prior killed run may also
    # have left a complete export behind -> skip the stitch entirely.
    if not set(frame_nos) <= set(digit_jpgs()):
        cname = "sdk-" + re.sub(r"[^a-zA-Z0-9_.-]", "", video.stem)
        timeout = 120 + 4 * len(frame_nos)  # roomy: 5.7K dual-file stitches are slower
        for attempt in (1, 2):
            try:
                subprocess.run(sdk_cmd(video, pano_dir, frame_nos, cname, params),
                               check=True, capture_output=True, timeout=timeout)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                subprocess.run(["docker", "kill", cname], capture_output=True)
            if set(frame_nos) <= set(digit_jpgs()):
                break
            if attempt == 2:
                raise RuntimeError(
                    f"MediaSDK exported {len(digit_jpgs())}/{len(frame_nos)} "
                    f"frames for {video.name}")

    exported = digit_jpgs()
    frames = []
    for index, frame_no in enumerate(sorted(exported)):
        path = pano_dir / pano_name(index)
        exported[frame_no].rename(path)
        frames.append(PanoFrame(index=index, t_sec=frame_no / native_fps, path=path))
    save_panos(pano_dir, frames)
    return frames


def save_panos(pano_dir: Path, frames: list[PanoFrame]) -> None:
    """Record which source frame each panorama came from.

    A timestamp cannot be re-derived from the file name without re-deriving the
    decimation, and it is what calibration times its neighbour frames off. So
    the stitch writes it down and every later stage reads it instead of guessing.
    """
    (pano_dir / PANO_INDEX).write_text(json.dumps(
        [{"index": f.index, "tSec": f.t_sec} for f in frames]))


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
