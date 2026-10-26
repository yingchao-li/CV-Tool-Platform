"""Dataset access for the point-cloud module.

Each ``.mat`` capture contains three arrays captured by a depth camera:

``amplitudes*``
    Intensity image, ``(H, W)``.
``distances*``
    Depth image in metres, ``(H, W)``.
``cloud*``
    Per-pixel 3D coordinates, ``(H, W, 3)``.

A point is considered invalid when its ``z`` coordinate is zero, which is
how the sensor encodes "no return" for that pixel.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io import loadmat

from ..paths import POINTCLOUD_DIR


def list_mat_files(data_dir: str | os.PathLike | None = None) -> list[str]:
    """Return every ``.mat`` capture in ``data_dir``, sorted by name."""
    directory = os.fspath(data_dir) if data_dir is not None else os.fspath(POINTCLOUD_DIR)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Point-cloud directory does not exist: {directory}")

    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".mat")
    )


def load_mat_file(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one capture and return ``(amplitude, distance, point_cloud)``."""
    data = loadmat(os.fspath(path))

    keys = {"amplitudes": None, "distances": None, "cloud": None}
    for key in data:
        if key.startswith("__"):
            continue
        for prefix in keys:
            if key.startswith(prefix):
                keys[prefix] = key

    missing = [prefix for prefix, key in keys.items() if key is None]
    if missing:
        raise ValueError(
            f"Could not find {missing} in {path}. Available keys: {list(data.keys())}"
        )

    amplitude = np.squeeze(data[keys["amplitudes"]])
    distance = np.squeeze(data[keys["distances"]])
    cloud = data[keys["cloud"]]

    check_data_shapes(amplitude, distance, cloud)
    return amplitude, distance, cloud


def load_example(
    example_id: int, data_dir: str | os.PathLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``example{example_id}kinect.mat`` from the point-cloud directory."""
    directory = os.fspath(data_dir) if data_dir is not None else os.fspath(POINTCLOUD_DIR)
    path = os.path.join(directory, f"example{example_id}kinect.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return load_mat_file(path)


def check_data_shapes(
    amplitude: np.ndarray, distance: np.ndarray, cloud: np.ndarray
) -> None:
    """Validate the amplitude/distance/point-cloud shape contract."""
    if amplitude.shape != distance.shape:
        raise ValueError(
            f"Amplitude and distance shapes differ: {amplitude.shape} vs {distance.shape}"
        )
    if cloud.ndim != 3 or cloud.shape[2] != 3:
        raise ValueError(f"Point cloud should have shape H x W x 3, got {cloud.shape}")
    if amplitude.shape != cloud.shape[:2]:
        raise ValueError(
            f"Images and point cloud differ: {amplitude.shape} vs {cloud.shape[:2]}"
        )


def describe_data(
    amplitude: np.ndarray, distance: np.ndarray, cloud: np.ndarray
) -> dict[str, object]:
    """Summarise a capture without printing, for logs and machine-readable output."""
    x, y, z = cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2]
    return {
        "shape": amplitude.shape,
        "amplitude_range": (float(np.nanmin(amplitude)), float(np.nanmax(amplitude))),
        "distance_range": (float(np.nanmin(distance)), float(np.nanmax(distance))),
        "invalid_ratio": float(np.mean(z == 0)),
        "bounds": {
            axis: (float(np.nanmin(values)), float(np.nanmax(values)))
            for axis, values in (("x", x), ("y", y), ("z", z))
        },
    }


def valid_point_mask(cloud: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that produced a valid 3D point."""
    return cloud[:, :, 2] != 0


def valid_points(cloud: np.ndarray) -> np.ndarray:
    """All valid 3D points as an ``(N, 3)`` array."""
    return cloud[valid_point_mask(cloud)]


def valid_points_with_indices(
    cloud: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid points plus their pixel coordinates.

    The row/column arrays let a 1D inlier mask be projected back onto the
    original image grid, which is required by every mask-based stage.
    """
    mask = valid_point_mask(cloud)
    rows, cols = np.where(mask)
    return cloud[mask], rows, cols
