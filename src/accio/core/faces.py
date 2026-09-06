"""Equirect panorama -> gnomonic (pinhole) side faces.

For a face of S x S pixels at heading phi with field of view theta, output
pixel (u, v) samples the panorama along the ray

    x = (2u/S - 1) tan(theta/2)
    y = (2v/S - 1) tan(theta/2)
    d = (x cos(phi) + sin(phi),  y,  -x sin(phi) + cos(phi))
    lon = atan2(d_x, d_z),  lat = atan2(d_y, sqrt(d_x^2 + d_z^2))
"""

from functools import lru_cache

import cv2
import numpy as np

from .params import FaceParams


@lru_cache(maxsize=32)
def gnomonic_maps(pano_h: int, pano_w: int, size: int, fov_deg: float,
                  yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    half = np.tan(np.radians(fov_deg) / 2.0)
    u = (2.0 * (np.arange(size) + 0.5) / size - 1.0) * half
    x, y = np.meshgrid(u, u)

    phi = np.radians(yaw_deg)
    dx = x * np.cos(phi) + np.sin(phi)
    dy = y
    dz = -x * np.sin(phi) + np.cos(phi)

    lon = np.arctan2(dx, dz)
    lat = np.arctan2(dy, np.hypot(dx, dz))

    map_x = ((lon / (2.0 * np.pi)) + 0.5) * pano_w - 0.5
    map_y = ((lat / np.pi) + 0.5) * pano_h - 0.5
    return map_x.astype(np.float32), map_y.astype(np.float32)


def render_face(pano: np.ndarray, yaw_deg: float, params: FaceParams) -> np.ndarray:
    map_x, map_y = gnomonic_maps(pano.shape[0], pano.shape[1],
                                 params.size, params.fov_deg, yaw_deg)
    return cv2.remap(pano, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def render_faces(pano: np.ndarray, params: FaceParams) -> dict[int, np.ndarray]:
    return {yaw: render_face(pano, yaw, params) for yaw in params.yaws}
