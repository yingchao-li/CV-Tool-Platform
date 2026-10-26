"""Experiment 8: input preprocessing by binary image inversion.

The baseline codebook is kept fixed so that this experiment isolates the
effect of changing image polarity before RootSIFT and VLAD encoding.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .common import (
    add_common_arguments,
    cached_value,
    evaluate_row,
    experiment_result,
    feature_cache_tag,
    get_codebook,
    get_esvm_encodings,
    normalize_config,
    prepare_data,
    print_summary,
    public_parameters,
    save_result_rows,
)
from ..pipeline import (
    nearest_cluster_indices,
    normalize_vlad,
    toBinary,
    vlad,
)


EXPERIMENT = "exp8_image_inversion"


def computeDescs_inverted(
    fname: str,
    norm_hellinger: bool = True,
    to_binary: bool = True,
) -> np.ndarray:
    """Compute fixed-angle RootSIFT after reversing foreground polarity."""
    image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(fname)

    if to_binary:
        image = toBinary(image)

    # Input experiment: 0/255 ink and background values exchange roles.
    image = 255 - image

    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    for keypoint in keypoints:
        keypoint.angle = 0.0

    _, descriptors = sift.compute(image, keypoints)
    if descriptors is None:
        return np.empty((0, 128), dtype=np.float32)

    descriptors = descriptors.astype(np.float32, copy=False)
    if norm_hellinger:
        eps = np.finfo(np.float32).eps
        descriptors /= descriptors.sum(axis=1, keepdims=True) + eps
        descriptors = np.sqrt(descriptors).astype(np.float32, copy=False)
    return descriptors


def vlad_inverted(
    files: list[str], codebook: np.ndarray, powernorm: bool = True
) -> np.ndarray:
    """Encode inverted images as float32 RootSIFT VLAD vectors."""
    k, dimension = codebook.shape
    encodings = np.empty((len(files), k * dimension), dtype=np.float32)

    for index, filename in enumerate(tqdm(files, desc="VLAD image inversion")):
        descriptors = computeDescs_inverted(filename, True, True)
        residual_matrix = np.zeros((k, dimension), dtype=np.float32)
        if len(descriptors):
            nearest = nearest_cluster_indices(descriptors, codebook)
            np.add.at(residual_matrix, nearest, descriptors - codebook[nearest])
        encodings[index] = normalize_vlad(residual_matrix.ravel(), powernorm)
    return encodings


def _get_inversion_encodings(
    config: dict,
    files: list[str],
    codebook: np.ndarray,
    split: str,
    *,
    inverted: bool,
) -> tuple[np.ndarray, Path]:
    """Load or compute VLAD while keeping inv0 and inv1 caches separate."""
    inversion_tag = f"inv{int(inverted)}"
    tag = feature_cache_tag(config, intranorm=False)
    path = Path(config["cache_dir"]) / f"vlad_{split}_{tag}_{inversion_tag}.pkl.gz"
    expected_shape = (len(files), codebook.shape[0] * codebook.shape[1])

    def compute() -> np.ndarray:
        if inverted:
            result = vlad_inverted(files, codebook, config["powernorm"])
        else:
            result = vlad(files, codebook, config["powernorm"])
        return result.astype(np.float32, copy=False)

    features = cached_value(
        path,
        compute,
        overwrite=config["overwrite"],
        validator=lambda value: isinstance(value, np.ndarray)
        and value.shape == expected_shape,
    )
    return features.astype(np.float32, copy=False), path


def run_experiment(config: dict) -> dict:
    started_at = time.perf_counter()
    np.random.seed(42)
    config = normalize_config(config)
    data = prepare_data(config)
    codebook = get_codebook(config, data["files_train"])

    baseline, _ = _get_inversion_encodings(
        config,
        data["files_test"],
        codebook,
        "test",
        inverted=False,
    )
    inverted, _ = _get_inversion_encodings(
        config,
        data["files_test"],
        codebook,
        "test",
        inverted=True,
    )

    suffix = " + power normalization" if config["powernorm"] else ""
    common = public_parameters(config, {"image_inversion": [False, True]})
    rows = [
        evaluate_row(
            EXPERIMENT,
            f"VLAD{suffix}",
            baseline,
            data["labels_test"],
            {**common, "image_inversion": False},
            started_at,
        ),
        evaluate_row(
            EXPERIMENT,
            f"VLAD{suffix} + image inversion",
            inverted,
            data["labels_test"],
            {**common, "image_inversion": True},
            started_at,
        ),
    ]

    if config["esvm"]:
        train_baseline, _ = _get_inversion_encodings(
            config,
            data["files_train"],
            codebook,
            "train",
            inverted=False,
        )
        train_inverted, _ = _get_inversion_encodings(
            config,
            data["files_train"],
            codebook,
            "train",
            inverted=True,
        )
        base_tag = feature_cache_tag(config, intranorm=False)
        baseline_esvm, _ = get_esvm_encodings(
            config,
            baseline,
            train_baseline,
            f"{base_tag}_inv0",
        )
        inverted_esvm, _ = get_esvm_encodings(
            config,
            inverted,
            train_inverted,
            f"{base_tag}_inv1",
        )
        rows.extend(
            [
                evaluate_row(
                    EXPERIMENT,
                    f"VLAD{suffix} + E-SVM",
                    baseline_esvm,
                    data["labels_test"],
                    {**common, "image_inversion": False, "esvm": True},
                    started_at,
                ),
                evaluate_row(
                    EXPERIMENT,
                    f"VLAD{suffix} + image inversion + E-SVM",
                    inverted_esvm,
                    data["labels_test"],
                    {**common, "image_inversion": True, "esvm": True},
                    started_at,
                ),
            ]
        )

    result_file = save_result_rows(rows, config["results_dir"], f"{EXPERIMENT}.csv")
    return experiment_result(EXPERIMENT, rows, result_file, common, started_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VLAD input preprocessing experiment: image inversion"
    )
    return add_common_arguments(parser)


def main() -> None:
    result = run_experiment(vars(build_parser().parse_args()))
    print_summary(result)


if __name__ == "__main__":
    main()
