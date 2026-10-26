"""Experiment 4: VLAD intra-normalization."""

from __future__ import annotations

import argparse
import time
from typing import Any

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


EXPERIMENT = "exp4_intranorm"


def run_experiment(config: dict) -> dict:
    started_at = time.perf_counter()
    np.random.seed(42)
    config = normalize_config(config)
    data = prepare_data(config)
    codebook = get_codebook(config, data["files_train"])

    baseline, _ = get_vlad_encodings(
        config, data["files_test"], codebook, "test", intranorm=False
    )
    intra, _ = get_vlad_encodings(
        config, data["files_test"], codebook, "test", intranorm=True
    )

    suffix = " + powernorm" if config["powernorm"] else ""
    common = public_parameters(config)
    rows = [
        evaluate_row(
            EXPERIMENT,
            f"VLAD{suffix}",
            baseline,
            data["labels_test"],
            {**common, "intranorm": False},
            started_at,
        ),
        evaluate_row(
            EXPERIMENT,
            f"VLAD + intranorm{suffix}",
            intra,
            data["labels_test"],
            {**common, "intranorm": True},
            started_at,
        ),
    ]

    if config["esvm"]:
        train_baseline, _ = get_vlad_encodings(
            config, data["files_train"], codebook, "train", intranorm=False
        )
        train_intra, _ = get_vlad_encodings(
            config, data["files_train"], codebook, "train", intranorm=True
        )
        baseline_esvm, _ = get_esvm_encodings(
            config,
            baseline,
            train_baseline,
            feature_cache_tag(config, intranorm=False),
        )
        intra_esvm, _ = get_esvm_encodings(
            config,
            intra,
            train_intra,
            feature_cache_tag(config, intranorm=True),
        )
        rows.extend(
            [
                evaluate_row(
                    EXPERIMENT,
                    f"VLAD{suffix} + E-SVM",
                    baseline_esvm,
                    data["labels_test"],
                    {**common, "intranorm": False, "esvm": True},
                    started_at,
                ),
                evaluate_row(
                    EXPERIMENT,
                    f"VLAD + intranorm{suffix} + E-SVM",
                    intra_esvm,
                    data["labels_test"],
                    {**common, "intranorm": True, "esvm": True},
                    started_at,
                ),
            ]
        )

    result_file = save_result_rows(rows, config["results_dir"], f"{EXPERIMENT}.csv")
    return experiment_result(EXPERIMENT, rows, result_file, common, started_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLAD intra-normalization experiment")
    return add_common_arguments(parser)


def main() -> None:
    result = run_experiment(vars(build_parser().parse_args()))
    print_summary(result)


if __name__ == "__main__":
    main()
