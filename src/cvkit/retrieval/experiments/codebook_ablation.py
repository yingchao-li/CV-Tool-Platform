"""Experiment 7: visual codebook size ablation."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .common import (
    add_common_arguments,
    evaluate_row,
    experiment_result,
    feature_cache_tag,
    get_codebook,
    get_esvm_encodings,
    get_vlad_encodings,
    normalize_config,
    prepare_data,
    print_summary,
    public_parameters,
    save_result_rows,
)


EXPERIMENT = "exp7_codebook_ablation"


def _k_values(config: dict) -> list[int]:
    values = config.get("k_values") or [64, 100, 128, 200]
    result = []
    for value in values:
        value = int(value)
        if value <= 1:
            raise ValueError("codebook K values must be greater than one")
        if value not in result:
            result.append(value)
    return result


def run_experiment(config: dict) -> dict:
    started_at = time.perf_counter()
    np.random.seed(42)
    config = normalize_config(config)
    k_values = _k_values(config)
    data = prepare_data(config)
    suffix = " + powernorm" if config["powernorm"] else ""
    common = public_parameters(config, {"k_values": k_values})
    rows = []

    for k in k_values:
        codebook = get_codebook(config, data["files_train"], n_clusters=k)
        test_features, _ = get_vlad_encodings(
            config,
            data["files_test"],
            codebook,
            "test",
            intranorm=False,
            n_clusters=k,
        )
        params = {**common, "n_clusters": k}
        rows.append(
            evaluate_row(
                EXPERIMENT,
                f"VLAD{suffix}, K={k}",
                test_features,
                data["labels_test"],
                params,
                started_at,
            )
        )

        if config["esvm"]:
            train_features, _ = get_vlad_encodings(
                config,
                data["files_train"],
                codebook,
                "train",
                intranorm=False,
                n_clusters=k,
            )
            tag = feature_cache_tag(config, n_clusters=k, intranorm=False)
            test_esvm, _ = get_esvm_encodings(
                config, test_features, train_features, tag
            )
            rows.append(
                evaluate_row(
                    EXPERIMENT,
                    f"VLAD{suffix} + E-SVM, K={k}",
                    test_esvm,
                    data["labels_test"],
                    {**params, "esvm": True},
                    started_at,
                )
            )

    result_file = save_result_rows(rows, config["results_dir"], f"{EXPERIMENT}.csv")
    return experiment_result(EXPERIMENT, rows, result_file, common, started_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLAD codebook size ablation")
    add_common_arguments(parser)
    parser.add_argument("--k_values", type=int, nargs="+", default=[64, 100, 128, 200])
    return parser


def main() -> None:
    result = run_experiment(vars(build_parser().parse_args()))
    print_summary(result)


if __name__ == "__main__":
    main()
