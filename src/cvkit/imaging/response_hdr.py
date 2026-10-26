"""HDR merging from an exposure bracket of JPEG frames.

After each frame is linearised with the estimated inverse response, the
radiance merge reuses :func:`cvkit.imaging.pipeline.replace_saturated_values`
so JPEG and RAW brackets share one merge implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .response_io import compute_luminance, image_shape

# Reuse the shared merge core instead of duplicating it.
from .pipeline import replace_saturated_values


HDR_DESCRIPTION = (
    "Threshold-replacement merge: start from the longest exposure, replace "
    "near-saturated pixels with exposure-scaled shorter frames, and keep "
    "replacing while the shorter measurement stays above the threshold."
)


# ============================================================
# Radiance merge
# ============================================================

def merge_hdr_from_jpg(
    paths: Sequence[str | Path],
    exposure_times: Sequence[float],
    gamma: float,
    min_valid: float = 0.02,
    max_valid: float = 0.95,
    chunk_rows: int = 256,
    threshold_ratio: float = 0.8,
) -> np.ndarray:
    """Linearise a JPEG bracket and merge it into a radiance map.

    The merge starts from the longest exposure and walks towards the
    shortest one:

    1. the longest exposure initialises the radiance estimate ``h``;
    2. a near-saturation threshold is set at ``threshold_ratio`` of its
       maximum luminance;
    3. each shorter frame is scaled by ``t_longest / t_current``;
    4. :func:`replace_saturated_values` swaps saturated pixels;
    5. pixels that are still saturated are retried on the next frame.

    Frames are processed in row chunks to keep 32 MP images within memory.
    ``min_valid``/``max_valid`` are accepted for a uniform estimator
    signature; the merge itself is driven by ``threshold_ratio``.
    """
    times = np.asarray(exposure_times, dtype=np.float64)
    if len(paths) != len(times):
        raise ValueError("One exposure time is required for each JPG")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if not 0.0 <= min_valid < max_valid <= 1.0:
        raise ValueError("Require 0 <= min_valid < max_valid <= 1")
    if not 0.0 < threshold_ratio < 1.0:
        raise ValueError("threshold_ratio must lie between 0 and 1")

    height, width = image_shape(paths)
    hdr = np.zeros((height, width, 3), dtype=np.float32)
    current_sensor_luminance = np.zeros((height, width), dtype=np.float32)

    # Walk from the longest exposure to the shortest.
    ordered = sorted(zip(paths, times), key=lambda item: item[1], reverse=True)
    longest_path, longest_exposure = ordered[0]

    # Initialise h with the longest exposure (not yet divided by time).
    with Image.open(longest_path) as source:
        image = source.convert("RGB")
        for top in range(0, height, chunk_rows):
            bottom = min(top + chunk_rows, height)
            rgb = np.asarray(
                image.crop((0, top, width, bottom)),
                dtype=np.float32,
            )
            rgb /= 255.0
            np.clip(rgb, 1e-8, 1.0, out=rgb)
            np.power(rgb, gamma, out=rgb)
            hdr[top:bottom] = rgb
            current_sensor_luminance[top:bottom] = compute_luminance(rgb)

    threshold = np.float32(threshold_ratio * current_sensor_luminance.max())
    print(
        f"Radiance merge: initial={Path(longest_path).name}, "
        f"threshold={float(threshold):.8g}"
    )

    # Replace pixels that are still on the saturation plateau.
    for path, exposure in ordered[1:]:
        scale_factor = np.float32(longest_exposure / exposure)
        replaced = 0
        with Image.open(path) as source:
            image = source.convert("RGB")
            for top in range(0, height, chunk_rows):
                bottom = min(top + chunk_rows, height)
                rgb = np.asarray(
                    image.crop((0, top, width, bottom)),
                    dtype=np.float32,
                )
                rgb /= 255.0
                np.clip(rgb, 1e-8, 1.0, out=rgb)
                np.power(rgb, gamma, out=rgb)
                sensor_luminance = compute_luminance(rgb)

                # Shared merge primitive.
                replaced += replace_saturated_values(
                    hdr[top:bottom],
                    current_sensor_luminance[top:bottom],
                    rgb,
                    sensor_luminance,
                    scale_factor=scale_factor,
                    threshold=threshold,
                )

        print(
            f"  {Path(path).name}: scale={float(scale_factor):.8g}, "
            f"replaced={replaced}"
        )

    # Values are relative to the longest exposure; convert to radiance.
    hdr /= np.float32(longest_exposure)
    return hdr


# ============================================================
# Display mapping
# ============================================================

def tone_map_log(
    hdr_rgb: np.ndarray,
    low_percentile: float = 0.1,
    high_percentile: float = 99.9,
    in_place: bool = False,
) -> np.ndarray:
    """Map HDR radiance to [0, 1] with log compression and percentiles."""
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("Invalid tone-map percentile range")
    mapped = np.asarray(hdr_rgb, dtype=np.float32)
    if not in_place:
        mapped = mapped.copy()
    np.maximum(mapped, 0.0, out=mapped)
    np.log1p(mapped, out=mapped)
    finite = mapped[np.isfinite(mapped)]
    if finite.size == 0:
        return np.zeros_like(mapped, dtype=np.float32)
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if high <= low + 1e-12:
        mapped.fill(0.0)
        return mapped
    mapped -= np.float32(low)
    mapped /= np.float32(high - low)
    np.clip(mapped, 0.0, 1.0, out=mapped)
    mapped[~np.isfinite(mapped)] = 0.0
    return mapped


def save_jpg_float(
    image: np.ndarray,
    path: str | Path,
    quality: int = 98,
) -> Path:
    """Write a [0, 1] float RGB image as a high-quality JPEG."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(rgb_u8, mode="RGB").save(
        output,
        format="JPEG",
        quality=int(quality),
    )
    return output


# ============================================================
# Output paths and summaries
# ============================================================

def output_paths(output_dir: str | Path, output_prefix: str) -> dict[str, Path]:
    """Return the five output paths produced by one estimator run."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return {
        "csv": folder / f"{output_prefix}_response_curve.csv",
        "curve_plot": folder / f"{output_prefix}_response_curve.png",
        "diagnostic": folder / f"{output_prefix}_diagnostic.png",
        "hdr": folder / f"{output_prefix}_hdr_result.jpg",
        "summary": folder / f"{output_prefix}_summary.txt",
    }


def save_summary_txt(
    path: str | Path,
    fields: Mapping[str, object] | Iterable[str],
) -> Path:
    """Write a human-readable summary of one run."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fields, Mapping):
        lines = [f"{key}: {value}" for key, value in fields.items()]
    else:
        lines = [str(line) for line in fields]
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output
