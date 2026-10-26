"""Paths and output layout for the detection module."""

from __future__ import annotations

from pathlib import Path

from ..paths import BALLOON_DIR, OUTPUT_DETECTION, PROJECT_ROOT

DEFAULT_RESULTS_ROOT = OUTPUT_DETECTION


def resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    """Return the balloon dataset root.

    ``dataset_root`` may be absolute, relative to the repository root, or
    omitted, in which case the standard ``data/balloons`` location is used.
    """
    if dataset_root is None:
        if not BALLOON_DIR.is_dir():
            raise FileNotFoundError(f"Balloon dataset not found at {BALLOON_DIR}")
        return BALLOON_DIR

    candidate = Path(dataset_root)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {candidate}")
    return candidate


def resolve_results_root(results_root: str | Path | None = None) -> Path:
    """Resolve the output root, defaulting to ``outputs/detection``."""
    root = DEFAULT_RESULTS_ROOT if results_root is None else Path(results_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def ensure_output_tree(results_root: str | Path | None = None) -> dict[str, Path]:
    """Create and return every directory used by the detection pipeline."""
    root = resolve_results_root(results_root)
    directories = {
        "root": root,
        "proposals": root / "proposals",
        "models": root / "models",
        "detections": root / "detections",
        "evaluation": root / "evaluation",
    }
    for split in ("train", "valid", "test"):
        directories[f"proposals_{split}"] = directories["proposals"] / split
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories
