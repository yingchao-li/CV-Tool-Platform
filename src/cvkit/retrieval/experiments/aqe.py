"""Experiment 6: Average Query Expansion (AQE) re-ranking."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    get_vlad_encodings,
    l2_normalize_rows,
    normalize_config,
    prepare_data,
    print_summary,
    public_parameters,
    save_result_rows,
)


EXPERIMENT = "exp6_aqe_rerank"


def _requested_ks(config: dict) -> list[int]:
    if config.get("aqe_ks"):
        values = config["aqe_ks"]
    elif config.get("aqe_k") is not None:
        values = [config["aqe_k"]]
    else:
        values = [5, 10]
    result = []
    for value in values:
        value = int(value)
        if value <= 0:
            raise ValueError("AQE k values must be positive")
        if value not in result:
            result.append(value)
    return result


def average_query_expansion(features: np.ndarray, top_k: int) -> np.ndarray:
    """Average every query with its top-k non-self nearest neighbors."""
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("AQE needs a 2D feature matrix with at least two samples")
    if top_k <= 0 or top_k >= len(features):
        raise ValueError(f"AQE k must be in [1, {len(features) - 1}], got {top_k}")

    normalized = l2_normalize_rows(features)
    similarities = normalized.dot(normalized.T).astype(np.float32, copy=False)
    np.fill_diagonal(similarities, -np.inf)
    neighbors = np.argpartition(-similarities, kth=top_k - 1, axis=1)[:, :top_k]
    del similarities

    # Accumulate one neighbor rank at a time to avoid an N x k x D tensor.
    expanded = normalized.copy()
    for rank in tqdm(range(top_k), desc=f"AQE k={top_k}"):
        expanded += normalized[neighbors[:, rank]]
    expanded /= np.float32(top_k + 1)
    return l2_normalize_rows(expanded)


def get_aqe_features(
    config: dict,
    features: np.ndarray,
    top_k: int,
    source_tag: str,
) -> tuple[np.ndarray, Path]:
    path = Path(config["cache_dir"]) / f"aqe_{source_tag}_aqe{top_k}.pkl.gz"
    result = cached_value(
        path,
        lambda: average_query_expansion(features, top_k),
        overwrite=config["overwrite"],
        validator=lambda x: isinstance(x, np.ndarray) and x.shape == features.shape,
    )
    return result.astype(np.float32, copy=False), path


def run_experiment(config: dict) -> dict:
    started_at = time.perf_counter()
    np.random.seed(42)
    config = normalize_config(config)
    aqe_ks = _requested_ks(config)
    data = prepare_data(config)
    if max(aqe_ks) >= len(data["files_test"]):
        raise ValueError(
            f"AQE k must be smaller than test set size ({len(data['files_test'])})"
        )

    codebook = get_codebook(config, data["files_train"])
    test_features, _ = get_vlad_encodings(
        config, data["files_test"], codebook, "test", intranorm=False
    )

    suffix = " + powernorm" if config["powernorm"] else ""
    common = public_parameters(config, {"aqe_ks": aqe_ks})
    rows = [
        evaluate_row(
            EXPERIMENT,
            f"VLAD{suffix}",
            test_features,
            data["labels_test"],
            {**common, "aqe_k": None},
            started_at,
        )
    ]
    base_tag = feature_cache_tag(config, intranorm=False)
    for top_k in aqe_ks:
        expanded, _ = get_aqe_features(config, test_features, top_k, base_tag)
        rows.append(
            evaluate_row(
                EXPERIMENT,
                f"VLAD{suffix} + AQE k={top_k}",
                expanded,
                data["labels_test"],
                {**common, "aqe_k": top_k},
                started_at,
            )
        )

    if config["esvm"]:
        train_features, _ = get_vlad_encodings(
            config, data["files_train"], codebook, "train", intranorm=False
        )
        test_esvm, _ = get_esvm_encodings(
            config, test_features, train_features, base_tag
        )
        c_tag = format(config["C"], "g").replace(".", "p")
        esvm_tag = f"{base_tag}_esvm_c{c_tag}"
        for top_k in aqe_ks:
            expanded, _ = get_aqe_features(config, test_esvm, top_k, esvm_tag)
            rows.append(
                evaluate_row(
                    EXPERIMENT,
                    f"VLAD{suffix} + E-SVM + AQE k={top_k}",
                    expanded,
                    data["labels_test"],
                    {**common, "aqe_k": top_k, "esvm": True},
                    started_at,
                )
            )

    result_file = save_result_rows(rows, config["results_dir"], f"{EXPERIMENT}.csv")
    return experiment_result(EXPERIMENT, rows, result_file, common, started_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLAD Average Query Expansion experiment")
    add_common_arguments(parser)
    parser.add_argument("--aqe_k", type=int)
    parser.add_argument("--aqe_ks", type=int, nargs="+")
    return parser


def main() -> None:
    result = run_experiment(vars(build_parser().parse_args()))
    print_summary(result)


if __name__ == "__main__":
    main()
