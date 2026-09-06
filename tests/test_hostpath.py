"""Host paths for the Docker daemon. Docker creates an empty directory for a
bind mount whose source does not exist, so a container path mounts nothing.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from accio import settings
from accio.core.extract import sdk_cmd
from accio.core.params import ExtractParams


@pytest.fixture
def containerised(monkeypatch, tmp_path):
    """accio at /app/data, the same volume at /srv/accio on the host."""
    monkeypatch.setattr(settings, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(settings, "HOST_DATA_ROOT", Path("/srv/accio"))
    return tmp_path


def test_bare_metal_leaves_paths_alone():
    assert settings.HOST_DATA_ROOT == settings.DATA_ROOT
    p = settings.DATA_ROOT / "videos" / "a.insv"
    assert settings.host_path(p) == p


def test_container_path_to_host_path(containerised):
    got = settings.host_path(containerised / "walks" / "w1" / "pano")
    assert got == Path("/srv/accio/walks/w1/pano")


def test_path_outside_data_root_unmapped(containerised):
    with pytest.raises(ValueError):
        settings.host_path(Path("/somewhere/else/video.insv"))


def test_stitch_command_mounts_host_paths(containerised):
    video = containerised / "videos" / "VID_1_00_9.insv"
    video.parent.mkdir(parents=True)
    video.touch()
    pano = containerised / "walks" / "w1" / "pano"

    cmd = sdk_cmd(video, pano, [0, 15], "sdk-test", ExtractParams())
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert mounts == ["/srv/accio/videos:/in:ro", "/srv/accio/walks/w1/pano:/outdir"]
    assert str(containerised) not in " ".join(cmd)


IMAGE = ExtractParams().sdk_image
have_image = shutil.which("docker") and subprocess.run(
    ["docker", "image", "inspect", IMAGE],
    capture_output=True).returncode == 0


@pytest.mark.skipif(not have_image, reason=f"{IMAGE} not pulled")
def test_working_mount_reports_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(settings, "HOST_DATA_ROOT", tmp_path)
    (tmp_path / "walks").mkdir()
    assert settings.mount_check(IMAGE) == ""


@pytest.mark.skipif(not have_image, reason=f"{IMAGE} not pulled")
def test_wrong_mount_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(settings, "HOST_DATA_ROOT", tmp_path / "nothing-here")
    (tmp_path / "walks").mkdir()
    why = settings.mount_check(IMAGE)
    assert "ACCIO_HOST_DATA" in why
    assert "sees nothing" in why
