import numpy as np
import pytest

from accio.core.faces import gnomonic_maps, render_face, render_faces
from accio.core.params import FaceParams

PARAMS = FaceParams(size=256)


def synthetic_pano(w=720, h=360):
    """Equirect image whose red channel encodes longitude, green latitude."""
    pano = np.zeros((h, w, 3), np.uint8)
    pano[:, :, 2] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]  # BGR: red
    pano[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    return pano


def test_face_centre_samples_its_heading():
    pano = synthetic_pano()
    for yaw in (45, 135, 225, 315):
        face = render_face(pano, yaw, PARAMS)
        c = PARAMS.size // 2
        lon_frac = face[c, c, 2] / 255.0          # 0..1 over -180..180
        expected = ((yaw + 180) % 360) / 360.0    # lon 0 sits at pano centre
        assert abs(lon_frac - expected) < 0.02, f"yaw {yaw}"


def test_face_centre_is_on_horizon():
    face = render_face(synthetic_pano(), 45, PARAMS)
    c = PARAMS.size // 2
    assert abs(face[c, c, 1] / 255.0 - 0.5) < 0.02


def test_maps_are_cached():
    a = gnomonic_maps(360, 720, 256, 110.0, 45.0)
    b = gnomonic_maps(360, 720, 256, 110.0, 45.0)
    assert a[0] is b[0]


def test_render_faces_returns_all_yaws():
    faces = render_faces(synthetic_pano(), PARAMS)
    assert set(faces) == {45, 135, 225, 315}
    assert all(f.shape == (256, 256, 3) for f in faces.values())


@pytest.mark.parametrize("fov", [100.0, 110.0])
def test_face_edges_half_fov_from_heading(fov):
    """fov 90 is left out: yaw 135's right edge lands on the pano seam."""
    params = FaceParams(size=256, fov_deg=fov)
    pano = synthetic_pano()
    c = params.size // 2
    for yaw in params.yaws:
        face = render_face(pano, yaw, params)
        for col, side in ((0, -1), (params.size - 1, 1)):
            expected = ((yaw + side * fov / 2 + 180) % 360) / 360.0
            off = abs(face[c, col, 2] / 255.0 - expected) % 1.0
            assert min(off, 1.0 - off) < 0.02, f"yaw {yaw} col {col}"
