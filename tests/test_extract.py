import json
from pathlib import Path

import pytest

from accio.core import extract
from accio.core.extract import (JPEG_EOI, PanoFrame, frame_numbers, front_lens,
                                lens_files, load_panos, pano_name, probe,
                                save_panos, sdk_cmd, stitch)
from accio.core.params import ExtractParams


def test_frame_numbers_decimates_to_target_fps():
    nos = frame_numbers(native_fps=29.97, nframes=300, target_fps=2.0)
    assert nos[:3] == [0, 15, 30]          # step = round(29.97 / 2) = 15
    assert all(b - a == 15 for a, b in zip(nos, nos[1:]))


def test_frame_numbers_never_upsamples():
    assert frame_numbers(native_fps=1.0, nframes=5, target_fps=2.0) == [0, 1, 2, 3]


def test_lens_files_pairs_dual_file_recordings(tmp_path):
    front = tmp_path / "VID_20260722_00_001.insv"
    back = tmp_path / "VID_20260722_10_001.insv"
    front.touch(), back.touch()
    assert lens_files(front) == [front, back]


def test_lens_files_single_file_recording(tmp_path):
    solo = tmp_path / "VID_20260722_00_001.insv"
    solo.touch()
    assert lens_files(solo) == [solo]


def test_front_lens_names_the_walk():
    front = Path("VID_20260722_00_001.insv")
    back = Path("VID_20260722_10_001.insv")
    assert front_lens([back, front]) == front
    assert front_lens([Path("X3_single.insv")]) == Path("X3_single.insv")


def test_sdk_cmd_flowstate_on_and_inputs_not_last(tmp_path):
    cmd = sdk_cmd(tmp_path / "walk.insv", tmp_path, [0, 15], "sdk-walk",
                  ExtractParams())
    assert "-enable_flowstate" in cmd
    assert cmd[cmd.index("-stitch_type") + 1] == "optflow"
    # the SDK's -inputs parser eats args until the next dash
    i = cmd.index("-inputs") + 1
    while cmd[i].startswith("/in/"):
        i += 1
    assert cmd[i].startswith("-")


def written(tmp_path, t_secs):
    frames = []
    for i, t in enumerate(t_secs):
        path = tmp_path / f"pano_{i:05d}.jpg"
        path.touch()
        frames.append(PanoFrame(index=i, t_sec=t, path=path))
    save_panos(tmp_path, frames)
    return frames


def test_panos_round_trip_true_timestamps(tmp_path):
    # 15-frame decimation of 29.97 fps, not the nominal 0.5 s
    frames = written(tmp_path, [0.0, 0.5005, 1.001])
    assert load_panos(tmp_path) == frames


def test_load_panos_names_missing(tmp_path):
    written(tmp_path, [0.0, 0.5005])
    (tmp_path / "pano_00001.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="pano_00001"):
        load_panos(tmp_path)


def test_panos_empty_walk(tmp_path):
    save_panos(tmp_path, [])
    assert load_panos(tmp_path) == []


def fake_ffprobe(monkeypatch, stdout, returncode=0, stderr=""):
    import subprocess as sp
    monkeypatch.setattr(extract.shutil, "which", lambda _: "/usr/bin/ffprobe")
    monkeypatch.setattr(extract.subprocess, "run", lambda *a, **k: sp.CompletedProcess(
        a[0], returncode, stdout=stdout, stderr=stderr))


def ffprobe_json(streams, duration=None) -> str:
    fmt = {} if duration is None else {"duration": duration}
    return json.dumps({"programs": [], "streams": streams, "format": fmt})


STREAM = {"width": 3840, "height": 1920, "avg_frame_rate": "30000/1001"}


def test_probe_reads_stream_by_key(monkeypatch):
    fake_ffprobe(monkeypatch, ffprobe_json([STREAM], "280.947333"))
    fps, frames, w, h = probe(Path("walk.insv"))
    assert (w, h) == (3840, 1920)
    assert fps == pytest.approx(29.97, abs=1e-2)
    assert frames == pytest.approx(8420, abs=2)


def test_probe_field_order_independent(monkeypatch):
    fake_ffprobe(monkeypatch, ffprobe_json(
        [{"avg_frame_rate": "30/1", "height": 1024, "width": 2048,
          "codec_name": "h264"}], "330.496000"))
    fps, frames, w, h = probe(Path("pano.mp4"))
    assert (w, h, fps) == (2048, 1024, 30.0)
    assert frames == 9914


def test_probe_names_rejected_file(monkeypatch):
    fake_ffprobe(monkeypatch, "", returncode=1, stderr="Invalid data found")
    with pytest.raises(RuntimeError, match=r"holiday\.pdf.*Invalid data"):
        probe(Path("holiday.pdf"))


@pytest.mark.parametrize("stdout", [
    ffprobe_json([], "280.9"),                              # audio only
    ffprobe_json([]),                                       # not a media file
    ffprobe_json([dict(STREAM, avg_frame_rate="0/0")], "280.9"),
    ffprobe_json([STREAM], "N/A"),
    ffprobe_json([{"width": 3840, "height": 1920}], "280.9"),
    "",
])
def test_probe_names_unmeasurable_file(monkeypatch, stdout):
    fake_ffprobe(monkeypatch, stdout)
    with pytest.raises(RuntimeError, match=r"odd\.bin"):
        probe(Path("odd.bin"))


def test_last_frame_has_reference_successor():
    """calibrate() reads frame round(t*fps)+1 of every exported panorama."""
    for n in (2, 5, 31, 8420):
        assert frame_numbers(29.97, n, 2.0)[-1] + 1 <= n - 1


def whole_jpeg_bytes() -> bytes:
    return b"\xff\xd8" + b"\x00" * 400 + JPEG_EOI


def test_stitch_drops_stale_rate_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(extract.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(extract, "probe_fps_nframes", lambda _v: (2.0, 5))
    for no in (0, 1, 2, 3, 7):
        (tmp_path / f"{no}.jpg").write_bytes(whole_jpeg_bytes())

    frames = stitch(tmp_path / "walk.insv", tmp_path, ExtractParams(fps=2.0))
    assert [f.t_sec for f in frames] == [0.0, 0.5, 1.0, 1.5]
    assert sorted(p.name for p in tmp_path.glob("*.jpg")) == \
        [pano_name(i) for i in range(4)]
