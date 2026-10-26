"""Filesystem layout shared by every CVKit module.

All paths are derived from the repository root so that the library works
from any working directory (scripts, notebooks, the Streamlit studio, or an
installed package).  Three environment variables can override the defaults:

``CVKIT_DATA_DIR``
    Location of the datasets.
``CVKIT_OUTPUT_DIR``
    Where generated results and caches are written.
``CVKIT_ROOT``
    Repository root, only needed when the package is relocated.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


PROJECT_ROOT = _env_path("CVKIT_ROOT", Path(__file__).resolve().parents[2])

DATA_DIR = _env_path("CVKIT_DATA_DIR", PROJECT_ROOT / "data")
OUTPUT_DIR = _env_path("CVKIT_OUTPUT_DIR", PROJECT_ROOT / "outputs")
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

POINTCLOUD_DIR = DATA_DIR / "pointcloud"
RAW_DIR = DATA_DIR / "raw"
BALLOON_DIR = DATA_DIR / "balloons"
PAINTING_DIR = DATA_DIR / "paintings"
FACE_DIR = DATA_DIR / "faces"

RAW_SAMPLE_DIR = RAW_DIR / "02"
RAW_EXPOSURE_DIR = RAW_DIR / "05"
RAW_BRACKET_DIR = RAW_DIR / "06"

OUTPUT_GEOMETRY = OUTPUT_DIR / "geometry"
OUTPUT_IMAGING = OUTPUT_DIR / "imaging"
OUTPUT_RETRIEVAL = OUTPUT_DIR / "retrieval"
OUTPUT_DETECTION = OUTPUT_DIR / "detection"
OUTPUT_FACES = OUTPUT_DIR / "faces"


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def relative_to_root(path: str | Path) -> str:
    """Render a path relative to the repository root when possible."""
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)
