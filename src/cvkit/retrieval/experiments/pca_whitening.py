"""Experiment 5: PCA whitening fitted only on training VLAD encodings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

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


EXPERIMENT = "exp5_pca_whitening"


def _requested_dims(config: dict) -> list[int]:
    values = config.get("pca_dims") or [config.get("pca_dim", 2048)]
    dims = []
    for value in values:
        value = int(value)
        if value <= 0:
            raise ValueError("PCA dimensions must be positive")
        if value not in dims:
            dims.append(value)
    return dims


def _pca_features(
    config: dict,
    train_features: np.ndarray,
    test_features: np.ndarray,
    requested_dim: int,
) -> tuple[np.ndarray, np.ndarray, int, Path]:
    max_dim = min(train_features.shape[0] - 1, train_features.shape[1])
    if max_dim < 1:
        raise ValueError("PCA whitening needs at least two training encodings")
    effective_dim = min(requested_dim, max_dim)
    base_tag = feature_cache_tag(config, intranorm=False)
    path = Path(config["cache_dir"]) / (
        f"pca_{base_tag}_pca{requested_dim}_effective{effective_dim}.pkl.gz"
    )

    def compute() -> dict:
        print(
            f"> fit PCA whitening: requested={requested_dim}, "
            f"effective={effective_dim}"
        )
        model = PCA(
            n_components=effective_dim,
            whiten=True,
            svd_solver="randomized",
            random_state=42,
        )
        model.fit(train_features)
        train_projected = l2_normalize_rows(model.transform(train_features))
        test_projected = l2_normalize_rows(model.transform(test_features))
        return {
            "train": train_projected.astype(np.float32, copy=False),
            "test": test_projected.astype(np.float32, copy=False),
            "requested_dim": requested_dim,
            "effective_dim": effective_dim,
        }

    expected_train = (len(train_features), effective_dim)
    expected_test = (len(test_features), effective_dim)
    value = cached_value(
        path,
        compute,
        overwrite=config["overwrite"],
        validator=lambda x: isinstance(x, dict)
        and isinstance(x.get("train"), np.ndarray)
        and isinstance(x.get("test"), np.ndarray)
        and x["train"].shape == expected_train
        and x["test"].shape == expected_test,
    )
    return value["train"], value["test"], effective_dim, path


def run_experiment(config: dict) -> dict:
    started_at = time.perf_counter()
    np.random.seed(42)
    config = normalize_config(config)
    dims = _requested_dims(config)
    data = prepare_data(config)
    codebook = get_codebook(config, data["files_train"])
    test_features, _ = get_vlad_encodings(
        config, data["files_test"], codebook, "test", intranorm=False
    )
    train_features, _ = get_vlad_encodings(
        config, data["files_train"], codebook, "train", intranorm=False
    )

    suffix = " + powernorm" if config["powernorm"] else ""
    common = public_parameters(config, {"pca_dims": dims})
    rows = [
        evaluate_row(
            EXPERIMENT,
            f"VLAD{suffix}",
            test_features,
            data["labels_test"],
            {**common, "pca_dim": None},
            started_at,
        )
    ]

    for requested_dim in dims:
        train_pca, test_pca, effective_dim, _ = _pca_features(
            config, train_features, test_features, requested_dim
        )
        params = {
            **common,
            "pca_dim": requested_dim,
            "effective_pca_dim": effective_dim,
        }
        method = f"VLAD{suffix} + PCA whitening ({effective_dim}D)"
        rows.append(
            evaluate_row(
                EXPERIMENT,
                method,
                test_pca,
                data["labels_test"],
                params,
                started_at,
            )
        )
        if config["esvm"]:
            pca_tag = f"{feature_cache_tag(config)}_pca{requested_dim}_effective{effective_dim}"
            test_esvm, _ = get_esvm_encodings(
                config, test_pca, train_pca, pca_tag
            )
            rows.append(
                evaluate_row(
                    EXPERIMENT,
                    method + " + E-SVM",
                    test_esvm,
                    data["labels_test"],
                    {**params, "esvm": True},
                    started_at,
                )
            )

    result_file = save_result_rows(rows, config["results_dir"], f"{EXPERIMENT}.csv")
    return experiment_result(EXPERIMENT, rows, result_file, common, started_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLAD PCA whitening experiment")
    add_common_arguments(parser)
    parser.add_argument("--pca_dim", type=int, default=2048)
    parser.add_argument("--pca_dims", type=int, nargs="+")
    return parser


def main() -> None:
    result = run_experiment(vars(build_parser().parse_args()))
    print_summary(result)


if __name__ == "__main__":
    main()
