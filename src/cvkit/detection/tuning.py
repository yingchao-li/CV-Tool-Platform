"""Bounded two-stage parameter search for the balloon detector.

Edit ``SEARCH_CONFIG`` and run this file from the project root. Stage 1 samples
Selective Search configurations and ranks them by validation MABO. Stage 2
trains a small number of SVMs for the best proposal configurations and tests
several evaluation settings per model. Nothing in the baseline output folders
is read as a parameter-search cache or overwritten.

Ground-truth boxes and their jittered variants, when enabled, are training-only
positive augmentation. They are never stored as Selective Search proposals and
never participate in proposal MABO.
"""

from __future__ import annotations

import csv
from itertools import product
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import joblib



from .datasets import ground_truth_boxes, load_coco_split
from .evaluation import compute_mabo, evaluate
from .proposals import (
    generate_proposals_for_image,
    load_proposal_cache,
    read_rgb,
    save_proposal_cache,
)
from .trainer import train
from ..paths import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Configuration: no command-line arguments are required.
# ---------------------------------------------------------------------------
SEARCH_CONFIG = {
    "dataset_root": None,
    "results_root": "outputs/detection",
    "search_root": "outputs/detection/parameter_search",
    "tune_split": "valid",
    "final_test_split": "test",
    "max_proposal_configs": 30,
    "top_k_proposal_configs": 5,
    "max_total_trials": 50,
    # At most 5 * 3 = 15 models are trained by default. Multiple evaluation
    # parameter sets reuse each model, which is much cheaper than 50 trainings.
    "max_training_configs_per_proposal": 3,
    "random_seed": 42,
    "baseline": {
        "MABO": 0.5567,
        "mAP": 0.1246,
    },
    "objective": {
        "map_weight": 1.0,
        "mabo_weight": 0.2,
    },
    # The supplied baseline used fallback_101_point. Keep this True for a
    # directly comparable metric even when pycocotools is installed.
    "force_fallback_map": True,
    "proposal_grid": {
        "scale": [50, 100, 200, 300, 500, 800],
        "sigma": [0.5, 0.8, 1.0],
        "min_size": [5, 10, 20, 30],
        "max_proposals": [500, 1000, 2000, 3000],
    },
    "training_grid": {
        "tp": [0.5, 0.6, 0.7, 0.75],
        "tn": [0.1, 0.2, 0.25, 0.3],
        "max_positive": [32, 64, 128],
        "max_negative": [64, 96, 128, 256],
        "negative_positive_ratio": [1, 3, 5],
        "class_weight": [None, "balanced"],
        "svm_C": [0.1, 1.0, 10.0],
        "add_gt_as_positive": [True, False],
        "jitter_gt_boxes": [True, False],
    },
    "evaluation_grid": {
        "score_threshold": [-1000000.0, -2.0, -1.0, 0.0],
        "nms_threshold": [0.2, 0.3, 0.5, 0.7],
        "max_detections": [50, 100, 200],
    },
    "fixed_training": {
        "resize_size": (64, 64),
        "orientations": 9,
        "pixels_per_cell": 8,
        "cells_per_block": 2,
        "min_crop_size": 4,
        "max_iter": 10000,
        "jitter_count_per_box": 2,
        "jitter_fraction": 0.08,
    },
}


TRIAL_FIELDS = (
    "trial_id",
    "phase",
    "split",
    "proposal_params",
    "training_params",
    "evaluation_params",
    "positive_samples",
    "negative_samples",
    "MABO",
    "mAP",
    "objective_score",
    "model_path",
    "metrics_path",
    "failed",
    "failure_reason",
)


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def prepare_paths(config):
    dataset_root = project_path(config["dataset_root"])
    results_root = project_path(config["results_root"])
    search_root = project_path(config["search_root"])
    expected_search_root = results_root / "parameter_search"
    if search_root != expected_search_root:
        raise ValueError(
            "search_root must be results_root/parameter_search so baseline outputs "
            f"cannot be overwritten; expected {expected_search_root}, got {search_root}"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    paths = {
        "dataset": dataset_root,
        "results": results_root,
        "search": search_root,
        "proposals": search_root / "proposals",
        "models": search_root / "models",
        "metrics": search_root / "metrics",
        "trials_csv": search_root / "trials.csv",
        "trials_json": search_root / "trials.json",
        "best_config": search_root / "best_config.json",
        "final_test_metrics": search_root / "final_test_metrics.json",
        "final_test_detections": search_root / "metrics" / "final_test_detections.json",
    }
    for key in ("search", "proposals", "models", "metrics"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, allow_nan=False)
    temporary.replace(path)


def save_trials(paths, trials):
    """Persist all accumulated trials after every completed or failed trial."""
    write_json(paths["trials_json"], trials)
    temporary = paths["trials_csv"].with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        for trial in trials:
            row = {}
            for field in TRIAL_FIELDS:
                value = trial.get(field)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(json_ready(value), sort_keys=True)
                row[field] = value
            writer.writerow(row)
    temporary.replace(paths["trials_csv"])


def compact_number(value):
    return f"{float(value):g}"


def normalized_proposal_params(params):
    return {
        "scale": float(params["scale"]),
        "sigma": float(params["sigma"]),
        "min_size": int(params["min_size"]),
        "max_proposals": int(params["max_proposals"]),
    }


def proposal_key(params):
    params = normalized_proposal_params(params)
    return (
        f"scale_{compact_number(params['scale'])}"
        f"_sigma_{compact_number(params['sigma'])}"
        f"_min_{params['min_size']}"
        f"_max_{params['max_proposals']}"
    )


def proposal_root_for(paths, params):
    return paths["proposals"] / proposal_key(params)


def proposal_cache_file(paths, params, split, file_name):
    return proposal_root_for(paths, params) / split / f"{Path(file_name).stem}.pkl"


def cache_matches(payload, params, split, image_record):
    expected = normalized_proposal_params(params)
    actual = payload.get("parameters", {})
    try:
        actual = normalized_proposal_params(actual)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        payload.get("bbox_format") == "xyxy_exclusive"
        and payload.get("split") == split
        and int(payload.get("image_id", -1)) == int(image_record["id"])
        and payload.get("file_name") == image_record["file_name"]
        and actual == expected
    )


def ensure_proposals(config, paths, params, split):
    """Generate/reuse only caches whose embedded parameters exactly match."""
    params = normalized_proposal_params(params)
    coco = load_coco_split(split, paths["dataset"])
    proposals_by_image = {}
    generated = 0
    reused = 0

    for image_id in sorted(coco["images"]):
        image_record = coco["images"][image_id]
        cache_path = proposal_cache_file(
            paths, params, split, image_record["file_name"]
        )
        payload = None
        if cache_path.is_file():
            try:
                candidate = load_proposal_cache(cache_path)
                if cache_matches(candidate, params, split, image_record):
                    payload = candidate
            except (OSError, ValueError, KeyError):
                payload = None

        if payload is None:
            image = read_rgb(image_record["path"])
            boxes = generate_proposals_for_image(
                image,
                scale=params["scale"],
                sigma=params["sigma"],
                min_size=params["min_size"],
                max_proposals=params["max_proposals"],
            )
            save_proposal_cache(
                cache_path,
                boxes=boxes,
                image_record=image_record,
                split=split,
                parameters=params,
            )
            generated += 1
            print(
                f"  [proposal generated] {split}/{image_record['file_name']}: "
                f"{len(boxes)}"
            )
        else:
            boxes = payload["boxes_xyxy"]
            reused += 1
        proposals_by_image[image_id] = boxes

    ground_truth_by_image = {
        image_id: ground_truth_boxes(coco, image_id)
        for image_id in sorted(coco["images"])
    }
    mabo, _ = compute_mabo(proposals_by_image, ground_truth_by_image)
    return {
        "coco": coco,
        "proposals": proposals_by_image,
        "ground_truth": ground_truth_by_image,
        "MABO": float(mabo),
        "generated_images": generated,
        "reused_images": reused,
        "proposal_root": proposal_root_for(paths, params),
    }


def objective_score(config, mabo, mean_ap):
    weights = config["objective"]
    return float(weights["map_weight"] * mean_ap + weights["mabo_weight"] * mabo)


def grid_combinations(grid):
    names = tuple(grid)
    return [
        dict(zip(names, values))
        for values in product(*(grid[name] for name in names))
    ]


def choose_proposal_configs(config, rng):
    combinations = grid_combinations(config["proposal_grid"])
    baseline = normalized_proposal_params(
        {"scale": 500, "sigma": 0.8, "min_size": 20, "max_proposals": 2000}
    )
    normalized = [normalized_proposal_params(item) for item in combinations]
    normalized = [item for item in normalized if item != baseline]
    rng.shuffle(normalized)
    maximum = min(max(1, int(config["max_proposal_configs"])), len(combinations))
    return [baseline, *normalized[: maximum - 1]]


def training_is_valid(params):
    return 0.0 <= float(params["tn"]) < float(params["tp"]) <= 1.0


def sample_unique_configs(grid, count, rng, *, predicate=lambda _: True, seeds=()):
    selected = []
    seen = set()

    def add(candidate):
        candidate = dict(candidate)
        key = json.dumps(candidate, sort_keys=True)
        if key in seen or not predicate(candidate):
            return False
        seen.add(key)
        selected.append(candidate)
        return True

    for seed in seeds:
        add(seed)
    names = tuple(grid)
    attempts = 0
    maximum_attempts = max(100, count * 100)
    while len(selected) < count and attempts < maximum_attempts:
        attempts += 1
        add({name: rng.choice(grid[name]) for name in names})
    return selected[:count]


def baseline_training_config():
    return {
        "tp": 0.75,
        "tn": 0.25,
        "max_positive": 32,
        "max_negative": 96,
        "negative_positive_ratio": 5,
        "class_weight": "balanced",
        "svm_C": 1.0,
        "add_gt_as_positive": False,
        "jitter_gt_boxes": False,
    }


def enhanced_training_config():
    """Guarantee that each top proposal config tests GT-based augmentation."""
    return {
        "tp": 0.7,
        "tn": 0.25,
        "max_positive": 64,
        "max_negative": 128,
        "negative_positive_ratio": 3,
        "class_weight": "balanced",
        "svm_C": 1.0,
        "add_gt_as_positive": True,
        "jitter_gt_boxes": True,
    }


def baseline_evaluation_config():
    return {
        "score_threshold": -1000000.0,
        "nms_threshold": 0.3,
        "max_detections": 100,
    }


def build_full_trial_plan(config, top_proposals, rng):
    """Create <= max_total_trials while sharing models across eval settings."""
    if not top_proposals:
        return []
    max_trials = max(1, int(config["max_total_trials"]))
    models_per_proposal = max(1, int(config["max_training_configs_per_proposal"]))
    model_groups = []
    for proposal in top_proposals:
        training_configs = sample_unique_configs(
            config["training_grid"],
            models_per_proposal,
            rng,
            predicate=training_is_valid,
            seeds=(baseline_training_config(), enhanced_training_config()),
        )
        for training_params in training_configs:
            model_groups.append(
                {
                    "proposal_params": proposal,
                    "training_params": training_params,
                }
            )

    evals_per_model = max(1, math.ceil(max_trials / len(model_groups)))
    for group in model_groups:
        group["evaluation_configs"] = sample_unique_configs(
            config["evaluation_grid"],
            evals_per_model,
            rng,
            seeds=(baseline_evaluation_config(),),
        )

    # Round-robin prevents the trial cap from excluding later proposal configs.
    plan = []
    for evaluation_index in range(evals_per_model):
        for group_index, group in enumerate(model_groups):
            if evaluation_index >= len(group["evaluation_configs"]):
                continue
            plan.append(
                {
                    "model_group": group_index,
                    "proposal_params": group["proposal_params"],
                    "training_params": group["training_params"],
                    "evaluation_params": group["evaluation_configs"][evaluation_index],
                }
            )
            if len(plan) >= max_trials:
                return plan
    return plan


def train_search_model(config, paths, trial_id, proposal_params, training_params):
    proposal_root = proposal_root_for(paths, proposal_params)
    model_path = paths["models"] / f"trial_{trial_id:04d}_balloon_svm.joblib"
    fixed = config["fixed_training"]
    args = SimpleNamespace(
        dataset_root=paths["dataset"],
        proposal_root=proposal_root,
        model_path=model_path,
        tp=float(training_params["tp"]),
        tn=float(training_params["tn"]),
        max_positive=int(training_params["max_positive"]),
        max_negative=int(training_params["max_negative"]),
        negative_positive_ratio=float(training_params["negative_positive_ratio"]),
        class_weight=training_params["class_weight"],
        c=float(training_params["svm_C"]),
        add_gt_as_positive=bool(training_params["add_gt_as_positive"]),
        jitter_gt_boxes=bool(training_params["jitter_gt_boxes"]),
        resize_size=tuple(fixed["resize_size"]),
        orientations=int(fixed["orientations"]),
        pixels_per_cell=int(fixed["pixels_per_cell"]),
        cells_per_block=int(fixed["cells_per_block"]),
        min_crop_size=int(fixed["min_crop_size"]),
        max_iter=int(fixed["max_iter"]),
        jitter_count_per_box=int(fixed["jitter_count_per_box"]),
        jitter_fraction=float(fixed["jitter_fraction"]),
        seed=int(config["random_seed"]) + trial_id,
    )
    train(args)
    bundle = joblib.load(model_path)
    metadata = bundle["training"]
    return {
        "model_path": model_path,
        "positive_samples": int(metadata["num_positive"]),
        "negative_samples": int(metadata["num_negative"]),
    }


def evaluate_search_model(
    config,
    paths,
    trial_id,
    split,
    proposal_params,
    evaluation_params,
    model_path,
    *,
    final=False,
):
    if final:
        metrics_path = paths["final_test_metrics"]
        detections_path = paths["final_test_detections"]
    else:
        metrics_path = paths["metrics"] / f"trial_{trial_id:04d}_{split}.json"
        detections_path = (
            paths["metrics"] / f"trial_{trial_id:04d}_{split}_detections.json"
        )
    args = SimpleNamespace(
        dataset_root=paths["dataset"],
        split=split,
        proposal_root=proposal_root_for(paths, proposal_params),
        model_path=model_path,
        score_threshold=float(evaluation_params["score_threshold"]),
        nms_threshold=float(evaluation_params["nms_threshold"]),
        max_detections=int(evaluation_params["max_detections"]),
        metrics_path=metrics_path,
        detections_path=detections_path,
        force_fallback=bool(config.get("force_fallback_map", True)),
    )
    metrics = evaluate(args)
    return metrics, metrics_path


def proposal_trial_record(trial_id, split, params):
    return {
        "trial_id": f"P{trial_id:04d}",
        "phase": "proposal_prescreen",
        "split": split,
        "proposal_params": params,
        "training_params": {},
        "evaluation_params": {},
        "positive_samples": 0,
        "negative_samples": 0,
        "MABO": None,
        "mAP": None,
        "objective_score": None,
        "model_path": None,
        "metrics_path": None,
        "failed": False,
        "failure_reason": "",
    }


def full_trial_record(trial_id, split, plan):
    return {
        "trial_id": f"T{trial_id:04d}",
        "phase": "full_validation",
        "split": split,
        "proposal_params": plan["proposal_params"],
        "training_params": plan["training_params"],
        "evaluation_params": plan["evaluation_params"],
        "positive_samples": 0,
        "negative_samples": 0,
        "MABO": None,
        "mAP": None,
        "objective_score": None,
        "model_path": None,
        "metrics_path": None,
        "failed": False,
        "failure_reason": "",
    }


def main():
    config = SEARCH_CONFIG
    paths = prepare_paths(config)
    tune_split = config["tune_split"]
    final_split = config["final_test_split"]
    if tune_split == "test":
        print(
            "Warning: parameter search is performed on test set, so the result "
            "may be overfitted."
        )
    elif tune_split != "valid":
        raise ValueError("tune_split must be 'valid' or 'test'")
    if final_split != "test":
        raise ValueError("final_test_split must be 'test'")

    rng = random.Random(int(config["random_seed"]))
    trials = []
    print("Bounded parameter search")
    print(f"Tune split: {tune_split}")
    print(f"Search output: {paths['search']}")
    print("Baseline folders are not used as search outputs.")

    # Stage 1: proposal-only MABO prescreen.
    successful_proposals = []
    proposal_configs = choose_proposal_configs(config, rng)
    print(f"\nStage 1: {len(proposal_configs)} proposal configurations")
    for index, proposal_params in enumerate(proposal_configs, start=1):
        record = proposal_trial_record(index, tune_split, proposal_params)
        print(f"\n[{record['trial_id']}] proposal params={proposal_params}")
        try:
            result = ensure_proposals(
                config, paths, proposal_params, tune_split
            )
            record["MABO"] = result["MABO"]
            record["objective_score"] = objective_score(
                config, result["MABO"], 0.0
            )
            proposal_metrics_path = (
                paths["metrics"] / f"proposal_{record['trial_id']}_{tune_split}.json"
            )
            write_json(
                proposal_metrics_path,
                {
                    "trial_id": record["trial_id"],
                    "split": tune_split,
                    "proposal_params": proposal_params,
                    "MABO": result["MABO"],
                    "generated_images": result["generated_images"],
                    "reused_images": result["reused_images"],
                    "proposal_root": result["proposal_root"],
                },
            )
            record["metrics_path"] = str(proposal_metrics_path)
            successful_proposals.append(
                {"params": proposal_params, "MABO": result["MABO"]}
            )
            print(
                f"[{record['trial_id']}] MABO={result['MABO']:.4f} "
                f"generated={result['generated_images']} reused={result['reused_images']}"
            )
        except Exception as exc:
            record["failed"] = True
            record["failure_reason"] = f"{type(exc).__name__}: {exc}"
            print(f"[{record['trial_id']}] FAILED: {record['failure_reason']}")
        trials.append(record)
        save_trials(paths, trials)

    successful_proposals.sort(key=lambda item: item["MABO"], reverse=True)
    top_k = min(int(config["top_k_proposal_configs"]), len(successful_proposals))
    top_proposals = [item["params"] for item in successful_proposals[:top_k]]
    if not top_proposals:
        raise RuntimeError("No proposal configuration completed successfully")
    print("\nTop proposal configurations:")
    for rank, item in enumerate(successful_proposals[:top_k], start=1):
        print(f"  {rank}. MABO={item['MABO']:.4f} params={item['params']}")

    # Stage 2: bounded training configurations, with model reuse across eval settings.
    plan = build_full_trial_plan(config, top_proposals, rng)
    print(
        f"\nStage 2: {len(plan)} full validation trials; "
        f"at most {top_k * int(config['max_training_configs_per_proposal'])} models"
    )
    model_cache = {}
    prepared_training_proposals = set()
    best = None
    baseline = config["baseline"]
    for trial_id, item in enumerate(plan, start=1):
        record = full_trial_record(trial_id, tune_split, item)
        print(
            f"\n[{record['trial_id']}] proposal={item['proposal_params']} "
            f"training={item['training_params']} evaluation={item['evaluation_params']}"
        )
        try:
            proposal_cache_key = proposal_key(item["proposal_params"])
            if proposal_cache_key not in prepared_training_proposals:
                ensure_proposals(config, paths, item["proposal_params"], "train")
                prepared_training_proposals.add(proposal_cache_key)
            model_group = int(item["model_group"])
            if model_group not in model_cache:
                model_result = train_search_model(
                    config,
                    paths,
                    trial_id,
                    item["proposal_params"],
                    item["training_params"],
                )
                model_cache[model_group] = model_result
            else:
                model_result = model_cache[model_group]
                print(f"  [model reused] {model_result['model_path']}")
            record.update(
                {
                    "positive_samples": model_result["positive_samples"],
                    "negative_samples": model_result["negative_samples"],
                    "model_path": str(model_result["model_path"]),
                }
            )

            metrics, metrics_path = evaluate_search_model(
                config,
                paths,
                trial_id,
                tune_split,
                item["proposal_params"],
                item["evaluation_params"],
                model_result["model_path"],
            )
            mabo = float(metrics["mabo"])
            mean_ap = float(metrics["map_50_95"])
            score = objective_score(config, mabo, mean_ap)
            record.update(
                {
                    "MABO": mabo,
                    "mAP": mean_ap,
                    "objective_score": score,
                    "metrics_path": str(metrics_path),
                }
            )
            print(
                f"[{record['trial_id']}] MABO={mabo:.4f} mAP={mean_ap:.4f} "
                f"objective={score:.4f}"
            )
            if mean_ap > float(baseline["mAP"]):
                print("New best result found! mAP exceeds the baseline reference.")
            if best is None or score > best["objective_score"]:
                best = dict(record)
                write_json(
                    paths["best_config"],
                    {
                        "selection_split": tune_split,
                        "baseline": baseline,
                        "objective": config["objective"],
                        "best_trial": best,
                    },
                )
        except Exception as exc:
            record["failed"] = True
            record["failure_reason"] = f"{type(exc).__name__}: {exc}"
            print(f"[{record['trial_id']}] FAILED: {record['failure_reason']}")
        trials.append(record)
        save_trials(paths, trials)

    if best is None:
        raise RuntimeError("No full validation trial completed successfully")

    print("\nFinal test evaluation using the best validation configuration")
    ensure_proposals(config, paths, best["proposal_params"], final_split)
    final_metrics, _ = evaluate_search_model(
        config,
        paths,
        int(best["trial_id"][1:]),
        final_split,
        best["proposal_params"],
        best["evaluation_params"],
        Path(best["model_path"]),
        final=True,
    )
    best_payload = {
        "selection_split": tune_split,
        "baseline": baseline,
        "objective": config["objective"],
        "best_trial": best,
        "final_test": {
            "MABO": float(final_metrics["mabo"]),
            "mAP": float(final_metrics["map_50_95"]),
            "metrics_path": str(paths["final_test_metrics"]),
        },
    }
    write_json(paths["best_config"], best_payload)

    print("\nSearch complete")
    print(f"Baseline: MABO={baseline['MABO']:.4f} mAP={baseline['mAP']:.4f}")
    print(
        f"Best validation: MABO={best['MABO']:.4f} "
        f"mAP={best['mAP']:.4f} objective={best['objective_score']:.4f}"
    )
    print(
        f"Final test: MABO={final_metrics['mabo']:.4f} "
        f"mAP={final_metrics['map_50_95']:.4f}"
    )
    print(f"Best config: {paths['best_config']}")


if __name__ == "__main__":
    main()
