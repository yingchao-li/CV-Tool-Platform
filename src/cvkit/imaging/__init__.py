"""Computational photography: RAW development, calibration and HDR.

The module keeps the stages separate so they can be composed or studied
individually:

``bayer``
    Colour-filter-array masks and a bilinear demosaicing baseline.
``pipeline``
    The full RAW-to-JPEG path plus the RAW HDR merge and iCAM06 tone
    mapping used for the bracketed exposures.
``response_io`` / ``response``
    Loading JPEG brackets and the shared gamma response model.
``response_gamma``, ``response_pairwise``, ``response_quantile``
    Three independent estimators of the camera inverse response.
``response_hdr``
    Radiance merge and display mapping for linearised JPEG brackets.
"""

from __future__ import annotations

from .bayer import create_bayer_masks, demosaic_simple
from .pipeline import (
    apply_icam06,
    compare_bayer_patterns,
    demosaic_raw,
    hdr_merge_demo,
    icam06_demo,
    icam06_tone_mapping,
    merge_hdr_raw,
    process_bayer_raw_to_rgb,
    process_raw,
    process_raw_to_array,
    replace_saturated_values,
    run_hdr_merge,
)
from .response import gamma_inverse_response, valid_weights
from .response_hdr import merge_hdr_from_jpg, save_jpg_float, tone_map_log
from .response_io import list_jpg_paths, parse_exposure_times

__all__ = [
    "apply_icam06",
    "compare_bayer_patterns",
    "create_bayer_masks",
    "demosaic_raw",
    "demosaic_simple",
    "gamma_inverse_response",
    "hdr_merge_demo",
    "icam06_demo",
    "icam06_tone_mapping",
    "list_jpg_paths",
    "merge_hdr_from_jpg",
    "merge_hdr_raw",
    "parse_exposure_times",
    "process_bayer_raw_to_rgb",
    "process_raw",
    "process_raw_to_array",
    "replace_saturated_values",
    "run_hdr_merge",
    "save_jpg_float",
    "tone_map_log",
    "valid_weights",
]
