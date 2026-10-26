"""Configuration shared by the face-analysis components."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Final

from ..paths import FACE_DIR


class ReIdMode(Enum):
    """Operating mode of the re-identification pipeline."""

    IDENT = "ident"
    """Identify a face against a labelled gallery."""

    CLUSTER = "cluster"
    """Group faces into clusters without identity labels."""


def enum_choices(enum_type: type[Enum]) -> list:
    """Return enum members plus their string values, for argparse/Typer."""
    return list(enum_type) + [member.value for member in enum_type]


class Config:
    """Dataset locations for the face module.

    The face fixtures are small compared with the RAW data, so they ship
    with the repository under ``data/faces``.
    """

    DATA_DIR: Final[Path] = FACE_DIR

    TRAIN_DATA: Final[Path] = DATA_DIR / "train_data"
    """Per-identity training image sequences."""

    TEST_DATA: Final[Path] = DATA_DIR / "test_data"
    """Per-identity test sequences, including unknown identities."""

    RESNET50: Final[Path] = DATA_DIR / "resnet50_128.onnx"
    """Face embedding network (128-D output)."""

    CLUSTER_GALLERY: Final[Path] = DATA_DIR / "clustering_gallery.pkl"
    REC_GALLERY: Final[Path] = DATA_DIR / "recognition_gallery.pkl"

    EVAL_TRAIN_DATA: Final[Path] = DATA_DIR / "evaluation_train_data.pkl"
    EVAL_TEST_DATA: Final[Path] = DATA_DIR / "evaluation_test_data.pkl"

    CHAL_TRAIN_DATA: Final[Path] = DATA_DIR / "challenge_train_data.csv"
