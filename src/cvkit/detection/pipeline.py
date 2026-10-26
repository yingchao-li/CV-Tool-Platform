"""End-to-end detection pipeline: proposals -> SVM -> inference -> evaluation.

Edit ``CONFIG`` below and call :func:`run_full_pipeline`, or use the CLI:

    cvkit detection run

The step functions are also imported by ``apps/pages/5_detection.py`` so the UI
and the CLI execute exactly the same pipeline.
"""

from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace



import joblib

from .datasets import (
    SPLITS,
    ground_truth_boxes,
    load_coco_split,
    proposal_cache_path,
)
from .evaluation import evaluate as evaluate_pipeline
from .detector import run_inference as inference_pipeline
from .config import ensure_output_tree, resolve_dataset_root
from .proposals import generate_split, load_proposal_cache
from .trainer import train as train_pipeline
from ..paths import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Configuration: edit these values; no command-line arguments are required.
# ---------------------------------------------------------------------------
CONFIG = {
    "dataset_root": None,
    "results_root": None,
    "run_proposal_generation": True,
    "run_training": True,
    "run_inference": True,
    "run_evaluation": True,
    "proposal": {
        "split": "all",
        "scale": 500,
        "sigma": 0.8,
        "min_size": 20,
        "max_proposals": 2000,
        "force": False,
    },
    "training": {
        "tp": 0.75,
        "tn": 0.25,
        "max_positive": 32,
        "max_negative": 96,
        "negative_positive_ratio": 5,
    },
    "inference": {
        "input_image": (
            "data/balloons/test/"
            "17156759330_5af4f5a5b8_k_jpg.rf.52e622bd9600a6dc4652c3622c439616.jpg"
        ),
        "model_path": "outputs/detection/models/balloon_svm.joblib",
        "score_threshold": 0.5,
        "nms_threshold": 0.3,
        "reuse_proposals": True,
    },
    "evaluation": {
        "model_path": "outputs/detection/models/balloon_svm.joblib",
        "score_threshold": 0.0,
        "nms_threshold": 0.3,
        "max_detections": 100,
    },
}


class StepExecutionError(RuntimeError):
    """A pipeline step failed; ``log`` contains output produced before failure."""

    def __init__(self, step, cause, log=""):
        self.step = step
        self.cause = cause
        self.log = log
        super().__init__(f"{step} failed: {type(cause).__name__}: {cause}")


def project_path(value):
    """Resolve a configured path relative to the project root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def prepare_paths(config):
    """Validate dataset/output roots and create the individual output tree."""
    dataset_root = resolve_dataset_root(config.get("dataset_root"))
    output_dirs = ensure_output_tree(config.get("results_root"))
    return dataset_root, output_dirs


def captured_step(step, action):
    """Execute one step and return its structured result plus captured log."""
    stream = StringIO()
    try:
        with redirect_stdout(stream):
            result = action()
    except Exception as exc:
        raise StepExecutionError(step, exc, stream.getvalue()) from exc
    result["executed"] = True
    result["log"] = stream.getvalue()
    return result


def proposal_inventory(output_dirs):
    """Count cached images and proposals for every dataset split."""
    inventory = {}
    for split in SPLITS:
        cache_files = sorted(output_dirs[f"proposals_{split}"].glob("*.pkl"))
        proposal_count = sum(
            len(load_proposal_cache(cache_path)["boxes_xyxy"])
            for cache_path in cache_files
        )
        inventory[split] = {
            "cached_images": len(cache_files),
            "proposals": int(proposal_count),
            "path": output_dirs[f"proposals_{split}"],
        }
    return inventory


def run_proposal_generation(config):
    dataset_root, output_dirs = prepare_paths(config)
    settings = config["proposal"]
    requested_split = settings.get("split", "all")
    if requested_split not in (*SPLITS, "all"):
        raise ValueError(f"Unknown proposal split: {requested_split}")
    maximum = int(settings.get("max_proposals", 2000))
    if maximum == 0 or maximum < -1:
        raise ValueError("max_proposals must be positive, or -1 for no limit")
    maximum = None if maximum == -1 else maximum
    selected_splits = SPLITS if requested_split == "all" else (requested_split,)

    def action():
        current_run = {
            split: {"generated": 0, "skipped": 0, "proposals": 0}
            for split in SPLITS
        }
        for split in selected_splits:
            current_run[split] = generate_split(
                split,
                dataset_root=dataset_root,
                results_root=output_dirs["root"],
                scale=float(settings.get("scale", 500)),
                sigma=float(settings.get("sigma", 0.8)),
                min_size=int(settings.get("min_size", 20)),
                max_proposals=maximum,
                overwrite=bool(settings.get("force", False)),
            )
        inventory = proposal_inventory(output_dirs)
        for split in SPLITS:
            inventory[split].update(
                generated=current_run[split]["generated"],
                skipped=current_run[split]["skipped"],
            )
        return {
            "splits": inventory,
            "generated_images": sum(item["generated"] for item in current_run.values()),
            "skipped_images": sum(item["skipped"] for item in current_run.values()),
            "proposal_count": sum(item["proposals"] for item in inventory.values()),
            "proposal_root": output_dirs["proposals"],
        }

    return captured_step("proposal generation", action)


def run_training(config):
    dataset_root, output_dirs = prepare_paths(config)
    settings = config["training"]
    arguments = SimpleNamespace(
        dataset_root=dataset_root,
        tp=float(settings.get("tp", 0.75)),
        tn=float(settings.get("tn", 0.25)),
        max_positive=int(settings.get("max_positive", 32)),
        max_negative=int(settings.get("max_negative", 96)),
        negative_positive_ratio=float(settings.get("negative_positive_ratio", 5)),
        resize_size=tuple(settings.get("resize_size", (64, 64))),
        orientations=int(settings.get("orientations", 9)),
        pixels_per_cell=int(settings.get("pixels_per_cell", 8)),
        cells_per_block=int(settings.get("cells_per_block", 2)),
        min_crop_size=int(settings.get("min_crop_size", 4)),
        c=float(settings.get("c", 1.0)),
        max_iter=int(settings.get("max_iter", 10000)),
        seed=int(settings.get("seed", 0)),
    )

    def action():
        model_path = Path(train_pipeline(arguments))
        training = joblib.load(model_path)["training"]
        return {
            "positive_samples": int(training["num_positive"]),
            "negative_samples": int(training["num_negative"]),
            "total_samples": int(training["num_samples"]),
            "feature_length": int(training["feature_length"]),
            "model_path": model_path,
            "models_root": output_dirs["models"],
        }

    return captured_step("SVM training", action)


def run_inference(config):
    _, output_dirs = prepare_paths(config)
    settings = config["inference"]
    proposal_settings = config["proposal"]
    maximum = int(proposal_settings.get("max_proposals", 2000))
    maximum = None if maximum == -1 else maximum
    image_path = project_path(settings["input_image"])
    arguments = SimpleNamespace(
        input=image_path,
        model_path=project_path(settings["model_path"]),
        score_threshold=float(settings.get("score_threshold", 0.5)),
        nms_threshold=float(settings.get("nms_threshold", 0.3)),
        reuse_proposals=bool(settings.get("reuse_proposals", True)),
        scale=float(proposal_settings.get("scale", 500)),
        sigma=float(proposal_settings.get("sigma", 0.8)),
        min_size=int(proposal_settings.get("min_size", 20)),
        max_proposals=maximum,
        max_detections=int(settings.get("max_detections", 100)),
    )

    def action():
        inference_pipeline(arguments)
        image_output = output_dirs["detections"] / f"{image_path.stem}_detection.jpg"
        json_output = output_dirs["detections"] / f"{image_path.stem}_detection.json"
        if not image_output.is_file() or not json_output.is_file():
            raise RuntimeError("Inference completed without producing both output files")
        with json_output.open("r", encoding="utf-8") as handle:
            detection_data = json.load(handle)
        return {
            "detection_count": len(detection_data.get("detections", [])),
            "detection_image_path": image_output,
            "detection_json_path": json_output,
            "detections": detection_data.get("detections", []),
        }

    return captured_step("single-image inference", action)


def run_evaluation(config):
    dataset_root, output_dirs = prepare_paths(config)
    settings = config["evaluation"]
    arguments = SimpleNamespace(
        dataset_root=dataset_root,
        model_path=project_path(settings["model_path"]),
        score_threshold=float(settings.get("score_threshold", 0.0)),
        nms_threshold=float(settings.get("nms_threshold", 0.3)),
        max_detections=int(settings.get("max_detections", 100)),
    )

    def action():
        metrics = evaluate_pipeline(arguments)
        detections_path = output_dirs["evaluation"] / "test_detections.json"
        metrics_path = output_dirs["evaluation"] / "metrics.json"
        with detections_path.open("r", encoding="utf-8") as handle:
            detections = json.load(handle)
        detection_counts = Counter(item["image_id"] for item in detections)
        coco = load_coco_split("test", dataset_root)
        per_image = []
        for image_id in sorted(coco["images"]):
            image_record = coco["images"][image_id]
            cache_path = proposal_cache_path(
                output_dirs["root"], "test", image_record["file_name"]
            )
            proposals = load_proposal_cache(cache_path)["boxes_xyxy"]
            per_image.append(
                {
                    "file_name": image_record["file_name"],
                    "proposals": len(proposals),
                    "detections": detection_counts[image_id],
                    "gt": len(ground_truth_boxes(coco, image_id)),
                }
            )
        return {
            "mabo": float(metrics["mabo"]),
            "map": float(metrics["map_50_95"]),
            "map_backend": metrics["map_backend"],
            "metrics": metrics,
            "per_image": per_image,
            "metrics_path": metrics_path,
            "detections_path": detections_path,
        }

    return captured_step("test-set evaluation", action)


def run_full_pipeline(config):
    """Run enabled steps in dependency order."""
    steps = (
        ("proposal_generation", "run_proposal_generation", run_proposal_generation),
        ("training", "run_training", run_training),
        ("inference", "run_inference", run_inference),
        ("evaluation", "run_evaluation", run_evaluation),
    )
    results = {}
    for result_name, flag_name, function in steps:
        if config.get(flag_name, True):
            results[result_name] = function(config)
        else:
            results[result_name] = {
                "executed": False,
                "reason": f"{flag_name}=False",
                "log": "",
            }
    return results


def print_log(title, result):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if not result["executed"]:
        print(f"SKIPPED: {result['reason']}")
        return
    if result["log"]:
        print(result["log"], end="" if result["log"].endswith("\n") else "\n")


def main():
    print("Balloon detection pipeline")
    for flag in (
        "run_proposal_generation",
        "run_training",
        "run_inference",
        "run_evaluation",
    ):
        print(f"{flag}: {bool(CONFIG.get(flag, True))}")

    try:
        results = run_full_pipeline(CONFIG)
    except StepExecutionError as exc:
        print(f"\nERROR: {exc}")
        if exc.log:
            print("Output produced before failure:")
            print(exc.log)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    proposal = results["proposal_generation"]
    print_log("1. Proposal generation", proposal)
    if proposal["executed"]:
        for split, summary in proposal["splits"].items():
            print(
                f"{split}: images={summary['cached_images']}, "
                f"proposals={summary['proposals']}, generated={summary['generated']}, "
                f"skipped={summary['skipped']}"
            )
        print(f"Proposal root: {proposal['proposal_root']}")

    training = results["training"]
    print_log("2. SVM training", training)
    if training["executed"]:
        print(
            f"Training samples: positive={training['positive_samples']}, "
            f"negative={training['negative_samples']}, total={training['total_samples']}"
        )
        print(f"Model path: {training['model_path']}")

    inference = results["inference"]
    print_log("3. Single-image inference", inference)
    if inference["executed"]:
        print(f"Detections: {inference['detection_count']}")
        print(f"Detection image: {inference['detection_image_path']}")
        print(f"Detection JSON: {inference['detection_json_path']}")

    evaluation = results["evaluation"]
    print_log("4. Test-set evaluation", evaluation)
    if evaluation["executed"]:
        print(f"MABO: {evaluation['mabo']:.4f}")
        print(f"mAP: {evaluation['map']:.4f} ({evaluation['map_backend']})")
        print(f"Metrics JSON: {evaluation['metrics_path']}")

    print("\nTotal test completed successfully.")


if __name__ == "__main__":
    main()
