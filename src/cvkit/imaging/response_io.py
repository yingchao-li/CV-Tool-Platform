"""Input handling for JPEG exposure brackets.

Finds the bracketed frames, resolves their exposure times and samples the
pixel positions shared by every frame.  Keeping this layer separate means
all three response estimators see exactly the same inputs in the same
order, which is what makes their gamma estimates comparable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .config import EXPOSURE_DIR


# Default bracket and its nominal exposure times.
DEFAULT_DATA_DIR = EXPOSURE_DIR
DEFAULT_EXPOSURE_TEXT = "0.1,0.05,0.025,0.0125,0.00625,0.003125"
EXPECTED_STEMS = tuple(f"IMG_{number}" for number in range(3044, 3050))
EXPECTED_EXPOSURES = {
    f"IMG_{number}": 0.1 / (2 ** (number - 3044))
    for number in range(3044, 3050)
}
JPG_SUFFIXES = {".jpg", ".jpeg"}


# ============================================================
# File discovery and exposure times
# ============================================================

def list_jpg_paths(data_dir: str | Path) -> list[Path]:
    """Find IMG_3044 to IMG_3049 and sort them by file name."""
    folder = Path(data_dir).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Data directory not found: {folder}")

    candidates = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in JPG_SUFFIXES
        and path.stem.upper() in EXPECTED_STEMS
    ]
    paths = sorted(candidates, key=lambda path: path.name.lower())
    if not paths:
        raise FileNotFoundError(
            f"No IMG_3044..IMG_3049 JPG/JPEG files found in {folder}"
        )
    return paths


def default_exposure_times(paths: Sequence[str | Path]) -> np.ndarray:
    """Return the nominal exposure times; unknown stems fall back to halving per frame."""
    stems = [Path(path).stem.upper() for path in paths]
    if all(stem in EXPECTED_EXPOSURES for stem in stems):
        return np.asarray(
            [EXPECTED_EXPOSURES[stem] for stem in stems],
            dtype=np.float64,
        )
    return np.asarray(
        [1.0 / (2**index) for index in range(len(paths))],
        dtype=np.float64,
    )


def parse_exposure_times(
    command_line_value: str | Sequence[float] | None,
    paths: Sequence[str | Path],
) -> np.ndarray:
    """Parse exposure seconds separated by commas, spaces or semicolons and check the count."""
    if command_line_value is None or (
        isinstance(command_line_value, str) and not command_line_value.strip()
    ):
        values = default_exposure_times(paths)
    elif isinstance(command_line_value, str):
        parts = [
            part
            for part in re.split(r"[\s,;]+", command_line_value.strip())
            if part
        ]
        try:
            values = np.asarray([float(part) for part in parts], dtype=np.float64)
        except ValueError as exc:
            raise ValueError("Exposure times must be numeric seconds") from exc
    else:
        values = np.asarray(command_line_value, dtype=np.float64)

    if values.ndim != 1 or len(values) != len(paths):
        raise ValueError(
            f"Expected {len(paths)} exposure times for {len(paths)} images, "
            f"got {len(values)}"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Exposure times must be finite and greater than zero")
    return values


# ============================================================
# Image loading and luminance
# ============================================================

def load_jpg_float(path: str | Path) -> np.ndarray:
    """Read a JPG, convert it to RGB float32 and normalize to [0, 1]."""
    with Image.open(path) as image:
        rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb_u8.astype(np.float32) / 255.0


def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute luminance Y using the Rec.709 coefficients."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim < 2 or rgb.shape[-1] != 3:
        raise ValueError("RGB input must have a final dimension of size 3")
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def image_shape(paths: Sequence[str | Path]) -> tuple[int, int]:
    """Verify that all JPGs share one size and return (height, width)."""
    if not paths:
        raise ValueError("At least one JPG path is required")
    expected: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            size = image.size
        if expected is None:
            expected = size
        elif size != expected:
            raise ValueError(
                f"Image size mismatch: {path} is {size}, expected {expected}"
            )
    assert expected is not None
    width, height = expected
    return height, width


# ============================================================
# Shared pixel sampling
# ============================================================

def sample_shared_pixels(
    height: int,
    width: int,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample unique coordinates reproducibly so every exposure uses the same pixels."""
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    total = height * width
    count = min(int(sample_count), total)
    rng = np.random.default_rng(seed)
    flat = rng.choice(total, size=count, replace=False)
    rows, cols = np.divmod(flat, width)
    return rows.astype(np.intp), cols.astype(np.intp)


def collect_luminance_samples(
    paths: Sequence[str | Path],
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Extract luminance samples at the same coordinates from every JPG.

    Images stay uint8 until the selected pixels are read, then become float32.
    This avoids keeping a full float image of a 32MP JPG in memory during
    response estimation.
    """
    rows = np.asarray(rows, dtype=np.intp)
    cols = np.asarray(cols, dtype=np.intp)
    if rows.shape != cols.shape:
        raise ValueError("rows and cols must have the same shape")

    height, width = image_shape(paths)
    outside = (
        np.any(rows < 0)
        or np.any(rows >= height)
        or np.any(cols < 0)
        or np.any(cols >= width)
    )
    if outside:
        raise ValueError("Sample coordinates are outside the image")

    samples = np.empty((len(paths), rows.size), dtype=np.float32)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
        selected = rgb_u8[rows, cols].astype(np.float32) / 255.0
        samples[index] = compute_luminance(selected)
    return samples


# ============================================================
# Command-line output helpers
# ============================================================

def format_exposure_times(exposure_times: Sequence[float]) -> str:
    """Format exposure times as a single line for the summary."""
    return ", ".join(f"{value:.10g}" for value in exposure_times)


def print_input_report(
    paths: Sequence[Path],
    exposure_times: Sequence[float],
    sample_count: int,
) -> None:
    """Print input files, exposure times and the sample count to the command line."""
    print("Found JPG files:")
    for path, exposure in zip(paths, exposure_times):
        print(f"  {path}  exposure={exposure:.10g} s")
    print(f"Requested shared sampled pixels: {sample_count}")
