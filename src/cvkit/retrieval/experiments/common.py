"""Shared infrastructure for the individual VLAD experiments.

The baseline implementation lives in ``cvkit.retrieval.pipeline``.  This module only
adds parameter handling, robust cache keys, result persistence, and reusable
experiment helpers around that implementation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import _pickle as cPickle
import numpy as np
from tqdm import tqdm

from ..pipeline import (
    computeDescs,
    dictionary,
    esvm,
    evaluate,
    getFiles,
    loadRandomDescriptors,
    nearest_cluster_indices,
    normalize_vlad,
    vlad,
)


COMMON_DEFAULTS = {
    "in_train": "icdar2017-training-binary",
    "in_test": "ScriptNet-HistoricalWI-2017-binarized",
    "labels_train": "icdar17_labels_train.txt",
    "labels_test": "icdar17_labels_test.txt",
    "suffix_train": ".png",
    "suffix_test": ".jpg",
    "n_clusters": 100,
    "max_descriptors": 500000,
    "max_dictionary_files": 0,
    "overwrite": False,
    "powernorm": True,
    "esvm": False,
    "C": 1000.0,
    "results_dir": "results",
    "cache_dir": "cache",
}

# Reuse one descriptor sample across several K values in a single ablation run.
# The sample is still persisted on disk and is recomputed on the first access
# when --overwrite is active.
_DESCRIPTOR_MEMORY_CACHE: dict[str, np.ndarray] = {}


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--in_train", default=COMMON_DEFAULTS["in_train"])
    parser.add_argument("--in_test", default=COMMON_DEFAULTS["in_test"])
    parser.add_argument("--labels_train", default=COMMON_DEFAULTS["labels_train"])
    parser.add_argument("--labels_test", default=COMMON_DEFAULTS["labels_test"])
    parser.add_argument("--suffix_train", default=".png")
    parser.add_argument("--suffix_test", default=".jpg")
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--max_descriptors", type=int, default=500000)
    parser.add_argument("--max_dictionary_files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--powernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use signed square-root power normalization (default: enabled)",
    )
    parser.add_argument("--esvm", action="store_true")
    parser.add_argument("--C", type=float, default=1000.0)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--cache_dir", default="cache")
    return parser


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(COMMON_DEFAULTS)
    result.update({key: value for key, value in config.items() if value is not None})

    for key in ("n_clusters", "max_descriptors"):
        result[key] = int(result[key])
        if result[key] <= 0:
            raise ValueError(f"{key} must be positive")
    result["max_dictionary_files"] = int(result["max_dictionary_files"])
    if result["max_dictionary_files"] < 0:
        raise ValueError("max_dictionary_files must be zero or positive")
    result["C"] = float(result["C"])
    if result["C"] <= 0:
        raise ValueError("C must be positive")

    result["powernorm"] = bool(result["powernorm"])
    result["overwrite"] = bool(result["overwrite"])
    result["esvm"] = bool(result["esvm"])
    Path(result["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(result["cache_dir"]).mkdir(parents=True, exist_ok=True)
    return result


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    files_train, labels_train = getFiles(
        config["in_train"], config["suffix_train"], config["labels_train"]
    )
    files_test, labels_test = getFiles(
        config["in_test"], config["suffix_test"], config["labels_test"]
    )
    if not files_train or not files_test:
        raise ValueError("training and test sets must both be non-empty")
    _validate_files(files_train, "training")
    _validate_files(files_test, "test")
    return {
        "files_train": files_train,
        "labels_train": labels_train,
        "files_test": files_test,
        "labels_test": labels_test,
    }


def _validate_files(files: Iterable[str], split: str) -> None:
    for filename in files:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"missing {split} file: {filename}")


def dataset_key(config: dict[str, Any]) -> str:
    """Short fingerprint preventing caches from crossing dataset definitions."""
    parts = []
    for key in ("in_train", "in_test", "labels_train", "labels_test"):
        path = os.path.abspath(str(config[key]))
        stat = os.stat(path)
        parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
    parts.extend((config["suffix_train"], config["suffix_test"]))
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


def base_cache_tag(config: dict[str, Any], n_clusters: int | None = None) -> str:
    k = int(config["n_clusters"] if n_clusters is None else n_clusters)
    max_files = int(config["max_dictionary_files"])
    return (
        f"k{k}_d{int(config['max_descriptors'])}_f{max_files}_"
        f"data{dataset_key(config)}"
    )


def feature_cache_tag(
    config: dict[str, Any],
    *,
    n_clusters: int | None = None,
    intranorm: bool = False,
) -> str:
    return (
        f"{base_cache_tag(config, n_clusters)}_"
        f"pn{int(config['powernorm'])}_intra{int(intranorm)}"
    )


def load_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return cPickle.load(handle)


def save_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wb") as handle:
        cPickle.dump(value, handle, -1)
    os.replace(temporary, path)


def cached_value(
    path: Path,
    compute: Callable[[], Any],
    *,
    overwrite: bool,
    validator: Callable[[Any], bool] | None = None,
) -> Any:
    if path.exists() and not overwrite:
        value = load_pickle(path)
        if validator is None or validator(value):
            print(f"> load cache: {path}")
            return value
        print(f"> ignore incompatible cache: {path}")
    value = compute()
    save_pickle(path, value)
    print(f"> saved cache: {path}")
    return value


def get_codebook(
    config: dict[str, Any], files_train: list[str], n_clusters: int | None = None
) -> np.ndarray:
    k = int(config["n_clusters"] if n_clusters is None else n_clusters)
    tag = base_cache_tag(config, k)
    cache_dir = Path(config["cache_dir"])
    codebook_path = cache_dir / f"codebook_{tag}.pkl.gz"

    descriptor_path = cache_dir / (
        f"descriptors_d{config['max_descriptors']}_"
        f"f{config['max_dictionary_files']}_data{dataset_key(config)}.pkl.gz"
    )

    def train_codebook() -> np.ndarray:
        memory_key = str(descriptor_path.resolve())
        if memory_key in _DESCRIPTOR_MEMORY_CACHE:
            descriptors = _DESCRIPTOR_MEMORY_CACHE[memory_key]
        else:
            descriptors = cached_value(
                descriptor_path,
                lambda: loadRandomDescriptors(
                    files_train,
                    config["max_descriptors"],
                    config["max_dictionary_files"],
                ),
                overwrite=config["overwrite"],
                validator=lambda x: isinstance(x, np.ndarray)
                and x.ndim == 2
                and x.shape[1] == 128,
            )
            _DESCRIPTOR_MEMORY_CACHE[memory_key] = descriptors
        if len(descriptors) < k:
            raise ValueError(f"need at least {k} descriptors, got {len(descriptors)}")
        return dictionary(descriptors.astype(np.float32, copy=False), k)

    codebook = cached_value(
        codebook_path,
        train_codebook,
        overwrite=config["overwrite"],
        validator=lambda x: isinstance(x, np.ndarray) and x.shape == (k, 128),
    )
    return codebook.astype(np.float32, copy=False)


def encode_intranorm(files: list[str], codebook: np.ndarray, powernorm: bool) -> np.ndarray:
    """VLAD with per-visual-word L2 normalization before final normalization."""
    k, dimension = codebook.shape
    encodings = np.empty((len(files), k * dimension), dtype=np.float32)
    eps = np.finfo(np.float32).eps

    for index, filename in enumerate(tqdm(files, desc="VLAD intra-normalization")):
        descriptors = computeDescs(filename, True, True)
        residual_matrix = np.zeros((k, dimension), dtype=np.float32)
        if len(descriptors):
            nearest = nearest_cluster_indices(descriptors, codebook)
            np.add.at(residual_matrix, nearest, descriptors - codebook[nearest])
        block_norms = np.linalg.norm(residual_matrix, axis=1, keepdims=True)
        residual_matrix /= np.maximum(block_norms, eps)
        encodings[index] = normalize_vlad(residual_matrix.ravel(), powernorm)
    return encodings


def get_vlad_encodings(
    config: dict[str, Any],
    files: list[str],
    codebook: np.ndarray,
    split: str,
    *,
    intranorm: bool = False,
    n_clusters: int | None = None,
) -> tuple[np.ndarray, Path]:
    k = int(config["n_clusters"] if n_clusters is None else n_clusters)
    tag = feature_cache_tag(config, n_clusters=k, intranorm=intranorm)
    path = Path(config["cache_dir"]) / f"vlad_{split}_{tag}.pkl.gz"
    expected_shape = (len(files), k * codebook.shape[1])

    def compute() -> np.ndarray:
        if intranorm:
            result = encode_intranorm(files, codebook, config["powernorm"])
        else:
            result = vlad(files, codebook, config["powernorm"])
        return result.astype(np.float32, copy=False)

    result = cached_value(
        path,
        compute,
        overwrite=config["overwrite"],
        validator=lambda x: isinstance(x, np.ndarray) and x.shape == expected_shape,
    )
    return result.astype(np.float32, copy=False), path


def l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    result = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return (result / np.maximum(norms, np.finfo(np.float32).eps)).astype(
        np.float32, copy=False
    )


def get_esvm_encodings(
    config: dict[str, Any],
    test_features: np.ndarray,
    train_features: np.ndarray,
    feature_tag: str,
) -> tuple[np.ndarray, Path]:
    c_tag = format(float(config["C"]), "g").replace(".", "p")
    path = Path(config["cache_dir"]) / f"esvm_{feature_tag}_c{c_tag}.pkl.gz"
    expected_shape = test_features.shape
    result = cached_value(
        path,
        lambda: esvm(test_features, train_features, config["C"]).astype(np.float32),
        overwrite=config["overwrite"],
        validator=lambda x: isinstance(x, np.ndarray) and x.shape == expected_shape,
    )
    return result.astype(np.float32, copy=False), path


def evaluate_row(
    experiment: str,
    method: str,
    features: np.ndarray,
    labels: list[str],
    parameters: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    top1, mean_ap = evaluate(features, labels)
    return {
        "experiment": experiment,
        "method": method,
        "top1": float(top1),
        "map": float(mean_ap),
        "parameters": parameters,
        "runtime_seconds": float(time.perf_counter() - started_at),
    }


def save_result_rows(
    rows: list[dict[str, Any]], results_dir: str, filename: str
) -> Path:
    path = Path(results_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "method", "top1", "map", "parameters", "runtime_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["parameters"] = json.dumps(
                row.get("parameters", {}), sort_keys=True, ensure_ascii=False
            )
            writer.writerow({key: serializable.get(key) for key in fieldnames})
    return path


def experiment_result(
    experiment: str,
    rows: list[dict[str, Any]],
    result_file: Path,
    parameters: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("experiment produced no result rows")
    best = max(rows, key=lambda row: row["map"])
    return {
        "experiment": experiment,
        "method": best["method"],
        "top1": best["top1"],
        "map": best["map"],
        "parameters": parameters,
        "runtime_seconds": float(time.perf_counter() - started_at),
        "result_file": str(result_file),
        "results": rows,
    }


def public_parameters(config: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    keys = (
        "n_clusters",
        "max_descriptors",
        "max_dictionary_files",
        "powernorm",
        "esvm",
        "C",
    )
    result = {key: config[key] for key in keys}
    if extra:
        result.update(extra)
    return result


def print_summary(result: dict[str, Any]) -> None:
    print(f"\nResults saved to {result['result_file']}")
    for row in result["results"]:
        print(
            f"{row['method']}: Top-1={row['top1']:.6f}, "
            f"mAP={row['map']:.6f}"
        )
