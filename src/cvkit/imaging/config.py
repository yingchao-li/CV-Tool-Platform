"""Default Bayer phases and dataset locations for the imaging module.

The sample capture in ``data/raw`` was shot with a Canon body; the phase of
the colour filter array is fixed per camera, so it is configured once here
instead of being hard-coded in every stage.
"""

from __future__ import annotations

from ..paths import RAW_BRACKET_DIR, RAW_EXPOSURE_DIR, RAW_SAMPLE_DIR, RAW_DIR

#: Bayer phase of the numpy sample capture (estimated from colour reference patches).
PATTERN_NPY = "GBRG"

#: Bayer phase of the CR3 captures.
PATTERN_CR3 = "RGGB"

#: Single numpy RAW plus the camera JPEG used as a colour reference.
SAMPLE_NPY = str(RAW_DIR / "01" / "IMG_9939.npy")
SAMPLE_JPG = str(RAW_DIR / "01" / "IMG_9939.JPG")

#: A single CR3 used by the demosaicing and tone-mapping stages.
SAMPLE_RAW = str(RAW_SAMPLE_DIR / "IMG_4782.CR3")

#: Exposure bracket used to verify sensor linearity.
EXPOSURE_DIR = str(RAW_EXPOSURE_DIR)

#: Exposure bracket used to merge an HDR radiance map.
BRACKET_DIR = str(RAW_BRACKET_DIR)

#: Nominal exposure times (seconds) for the six bracketed JPG frames.
DEFAULT_EXPOSURE_TIMES = (0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125)
