"""Reproducible benchmarks for the face-analysis module.

Runs four experiments on the bundled face sequences and writes the CSV /
figure artefacts used in ``docs/benchmarks.md``:

1. tracking      - MTCNN detection versus template-matching tracking;
2. identification - closed-set rank-1 and open-set rejection rules;
3. clustering    - k-means purity and initialisation sensitivity;
4. dir           - detection-and-identification rate versus false alarms.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cvkit.faces.config import Config
from cvkit.faces.dir_eval import OpenSetEvaluation
from cvkit.faces.detector import FaceDetector
from cvkit.faces.recognition import FaceClustering, FaceNet, FaceRecognizer

OUT = ROOT / "docs"
FIGURES = OUT / "assets" / "faces"
RESULTS = OUT / "results" / "faces"

RANDOM_SEED = 42
KNOWN_LABELS = ["Alan_Ball", "Manuel_Pellegrini", "Marina_Silva", "Nancy_Sinatra", "Peter_Gilmour"]
UNKNOWN_LABELS = ["Al_Pacino", "Maria_Shriver"]
GALLERY_FACES_PER_LABEL = 5
KNOWN_TEST_FACES_PER_LABEL = 8
UNKNOWN_TEST_FACES_PER_LABEL = 10
CLUSTER_LABELS = ["Alan_Ball", "Marina_Silva", "Nancy_Sinatra"]
CLUSTER_FACES_PER_LABEL = 15
CLUSTER_SEEDS = [0, 1, 2, 42, 100]


def ensure_dirs() -> None:
    for path in (FIGURES, RESULTS):
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resize_like_training(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if width < 640:
        scale = 640.0 / width
        return cv2.resize(image, (int(scale * width), int(scale * height)))
    return image


def sorted_images(root: Path, label: str) -> list[Path]:
    return sorted((root / label).glob("*.jpg"))


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_face_result(frame: np.ndarray, face, title: str, path: Path) -> None:
    display = frame.copy()
    if face is not None:
        x, y, w, h = face.rect
        cv2.rectangle(display, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(to_rgb(display))
    axes[0].set_title(title)
    axes[0].axis("off")
    if face is not None:
        axes[1].imshow(to_rgb(face.aligned))
        axes[1].set_title("Aligned 224x224")
    else:
        axes[1].text(0.5, 0.5, "No face", ha="center", va="center")
        axes[1].set_title("Aligned 224x224")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def get_aligned_samples(
    root: Path,
    labels: list[str],
    count_per_label: int,
    scan_limit: int = 80,
    preprocess: bool = False,
) -> list[dict]:
    detector = FaceDetector()
    samples: list[dict] = []
    for label in labels:
        found = 0
        for image_path in sorted_images(root, label)[:scan_limit]:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            detect_image = resize_like_training(image) if preprocess else image
            face = detector.detect_face(detect_image)
            if face is None:
                continue
            samples.append({"label": label, "path": image_path, "aligned": face.aligned, "face": face})
            found += 1
            if found >= count_per_label:
                break
    return samples


def count_by_label_text(labels: list[str], counts: Counter) -> str:
    return "; ".join(f"{label}:{int(counts.get(label, 0))}" for label in labels)


def gallery_metadata(recognizer: FaceRecognizer, train_samples: list[dict]) -> dict:
    face_counts = Counter(sample["label"] for sample in train_samples)
    embedding_counts = Counter(recognizer.labels)
    return {
        "gallery_face_samples_total": len(train_samples),
        "gallery_embedding_total": len(recognizer.labels),
        "gallery_face_samples_by_identity": count_by_label_text(KNOWN_LABELS, face_counts),
        "gallery_embeddings_by_identity": count_by_label_text(KNOWN_LABELS, embedding_counts),
    }


def majority_vote(labels: np.ndarray, distances: np.ndarray):
    best_label = None
    best_count = -1
    best_distance = math.inf
    for label in sorted(set(labels), key=str):
        label_distances = distances[labels == label]
        count = len(label_distances)
        min_distance = float(np.min(label_distances))
        if count > best_count or (count == best_count and min_distance < best_distance):
            best_label = label
            best_count = count
            best_distance = min_distance
    return best_label


def raw_knn_prediction(
    gallery_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_embeddings: list[np.ndarray],
    k: int,
) -> tuple[str, float, float]:
    distances: list[float] = []
    labels: list[str] = []
    for query_embedding in query_embeddings:
        query_distances = np.linalg.norm(gallery_embeddings - query_embedding, axis=1)
        valid = np.isfinite(query_distances)
        distances.extend(query_distances[valid])
        labels.extend(gallery_labels[valid])

    distances_array = np.asarray(distances, dtype=float)
    labels_array = np.asarray(labels, dtype=object)
    k = min(k, len(distances_array))
    nearest = np.argsort(distances_array, kind="mergesort")[:k]
    nearest_labels = labels_array[nearest]
    nearest_distances = distances_array[nearest]
    pred = majority_vote(nearest_labels, nearest_distances)
    mask = nearest_labels == pred
    prob = float(np.count_nonzero(mask) / k)
    distance = float(np.min(nearest_distances[mask]))
    return str(pred), prob, distance


def apply_open_set_rule(label: str, prob: float, distance: float, rule: str, tau_d: float, tau_p: float) -> str:
    reject_by_distance = distance > tau_d
    reject_by_prob = prob < tau_p
    if rule == "distance" and reject_by_distance:
        return "unknown"
    if rule == "posterior" and reject_by_prob:
        return "unknown"
    if rule == "both" and (reject_by_distance or reject_by_prob):
        return "unknown"
    return label


def run_tracking() -> dict:
    windows = [15, 25, 40, 60]
    thresholds = [0.50, 0.60, 0.70, 0.80]
    frames = []
    for path in sorted_images(Config.TRAIN_DATA, "Alan_Ball")[:45]:
        image = cv2.imread(str(path))
        if image is not None:
            frames.append((path, resize_like_training(image)))

    rows = []
    representative = None
    for window in windows:
        for threshold in thresholds:
            detector = FaceDetector(tm_threshold=threshold)
            detector.tm_search_margin = window
            original_detect = detector.detect_face
            state = {"calls": 0, "successes": 0}

            def counted_detect(image, _original=original_detect):
                state["calls"] += 1
                face = _original(image)
                if face is not None:
                    state["successes"] += 1
                return face

            detector.detect_face = counted_detect
            found = 0
            missing = 0
            template_success = 0
            start = time.perf_counter()
            last_face = None
            last_frame = None
            for _, frame in frames:
                before = state["calls"]
                face = detector.track_face(frame)
                after = state["calls"]
                if face is None:
                    missing += 1
                else:
                    found += 1
                    last_face = face
                    last_frame = frame
                    if after == before:
                        template_success += 1
            elapsed = time.perf_counter() - start
            total = len(frames)
            rows.append(
                {
                    "search_window_px": window,
                    "threshold": threshold,
                    "total_frames": total,
                    "face_frames": found,
                    "template_successes": template_success,
                    "mtcnn_detection_calls": state["calls"],
                    "mtcnn_successes": state["successes"],
                    "mtcnn_redetection_calls_after_first": max(0, state["calls"] - 1),
                    "no_face_frames": missing,
                    "avg_time_s": elapsed / total if total else 0.0,
                }
            )
            if representative is None and last_face is not None and last_frame is not None:
                representative = (last_frame, last_face, window, threshold)

    result_path = RESULTS / "tracking_results.csv"
    write_csv(result_path, rows)

    best = max(
        rows,
        key=lambda r: (
            int(r["template_successes"]) / max(1, int(r["total_frames"])),
            -int(r["no_face_frames"]),
            -float(r["avg_time_s"]),
        ),
    )
    if representative is not None:
        frame, face, window, threshold = representative
        draw_face_result(
            frame,
            face,
            f"Tracking result, window={window}, threshold={threshold}",
            FIGURES / "tracking_example.png",
        )

    return {"rows": rows, "best": best}


def build_recognition_data() -> tuple[FaceRecognizer, list[dict], list[dict], list[dict]]:
    recognizer = FaceRecognizer()
    recognizer.labels = []
    recognizer.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)

    train_samples = get_aligned_samples(
        Config.TRAIN_DATA,
        KNOWN_LABELS,
        count_per_label=GALLERY_FACES_PER_LABEL,
        scan_limit=120,
    )
    for sample in train_samples:
        recognizer.partial_fit(sample["aligned"], sample["label"])

    known_test = get_aligned_samples(
        Config.TEST_DATA,
        KNOWN_LABELS,
        count_per_label=KNOWN_TEST_FACES_PER_LABEL,
        scan_limit=160,
    )
    unknown_test = get_aligned_samples(
        Config.TEST_DATA,
        UNKNOWN_LABELS,
        count_per_label=UNKNOWN_TEST_FACES_PER_LABEL,
        scan_limit=120,
    )
    return recognizer, train_samples, known_test, unknown_test


def evaluate_recognition_config(
    recognizer: FaceRecognizer,
    query_records: list[dict],
    k: int,
    rule: str,
    tau_d: float,
    tau_p: float,
) -> dict:
    gallery_labels = np.asarray(recognizer.labels, dtype=object)
    gallery_embeddings = np.asarray(recognizer.embeddings, dtype=np.float32)
    known_total = 0
    known_raw_correct = 0
    known_correct_after_rule = 0
    known_rejected = 0
    unknown_total = 0
    unknown_rejected = 0
    predictions = []

    for record in query_records:
        raw_label, prob, distance = raw_knn_prediction(
            gallery_labels, gallery_embeddings, record["embeddings"], k
        )
        final_label = apply_open_set_rule(raw_label, prob, distance, rule, tau_d, tau_p)
        is_known = record["is_known"]
        true_label = record["label"]
        if is_known:
            known_total += 1
            known_raw_correct += int(raw_label == true_label)
            known_correct_after_rule += int(final_label == true_label)
            known_rejected += int(final_label == "unknown")
        else:
            unknown_total += 1
            unknown_rejected += int(final_label == "unknown")
        predictions.append((record, raw_label, final_label, prob, distance))

    return {
        "k": k,
        "rule": rule,
        "distance_threshold": tau_d,
        "posterior_threshold": tau_p,
        "known_samples": known_total,
        "unknown_samples": unknown_total,
        "closed_set_rank1": known_raw_correct / known_total if known_total else 0.0,
        "known_identification_rate": known_correct_after_rule / known_total if known_total else 0.0,
        "known_false_reject_rate": known_rejected / known_total if known_total else 0.0,
        "unknown_rejection_rate": unknown_rejected / unknown_total if unknown_total else 0.0,
        "unknown_false_accept_rate": 1.0 - (unknown_rejected / unknown_total if unknown_total else 0.0),
        "_predictions": predictions,
    }


def transform_aligned(aligned: np.ndarray, transform: str) -> np.ndarray:
    if transform == "original":
        return aligned.copy()
    if transform in ("dark", "synthetic_darkening"):
        return np.clip(aligned.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    if transform in ("occluded", "synthetic_lower_face_occlusion"):
        output = aligned.copy()
        h, w = output.shape[:2]
        cv2.rectangle(output, (w // 4, h // 2), (3 * w // 4, 3 * h // 4), (0, 0, 0), -1)
        return output
    if transform in ("side_crop", "synthetic_lateral_crop"):
        output = aligned.copy()
        w = output.shape[1]
        output[:, : w // 3] = 0
        return output
    raise ValueError(transform)


ROBUSTNESS_TRANSFORMS = [
    ("original", "Original"),
    ("synthetic_darkening", "Darkening"),
    ("synthetic_lower_face_occlusion", "Lower-face occlusion"),
    ("synthetic_lateral_crop", "Lateral crop"),
]


def evaluate_robustness_transform(
    recognizer: FaceRecognizer,
    samples: list[dict],
    transform: str,
    k: int,
    rule: str,
    tau_d: float,
    tau_p: float,
) -> dict:
    gallery_labels = np.asarray(recognizer.labels, dtype=object)
    gallery_embeddings = np.asarray(recognizer.embeddings, dtype=np.float32)
    total = 0
    raw_correct = 0
    accepted = 0
    final_correct = 0
    distances = []
    posteriors = []

    for sample in samples:
        aligned = transform_aligned(sample["aligned"], transform)
        query_embeddings = recognizer._extract_color_and_gray_embeddings(aligned)
        if not query_embeddings:
            continue
        raw_label, posterior, distance = raw_knn_prediction(
            gallery_labels,
            gallery_embeddings,
            query_embeddings,
            k,
        )
        final_label = apply_open_set_rule(raw_label, posterior, distance, rule, tau_d, tau_p)
        total += 1
        raw_correct += int(raw_label == sample["label"])
        accepted += int(final_label != "unknown")
        final_correct += int(final_label == sample["label"])
        distances.append(float(distance))
        posteriors.append(float(posterior))

    known_acceptance_rate = accepted / total if total else 0.0
    open_set_correct = final_correct / total if total else 0.0
    return {
        "transform": transform,
        "display_name": dict(ROBUSTNESS_TRANSFORMS).get(transform, transform),
        "samples": total,
        "k": k,
        "rule": rule,
        "distance_threshold": tau_d,
        "posterior_threshold": tau_p,
        "identification_rate": open_set_correct,
        "known_acceptance_rate": known_acceptance_rate,
        "known_false_rejection_rate": 1.0 - known_acceptance_rate if total else 0.0,
        "average_distance": float(np.mean(distances)) if distances else float("nan"),
        "average_posterior_probability": float(np.mean(posteriors)) if posteriors else float("nan"),
        "correct_class_prediction_rate": raw_correct / total if total else 0.0,
        "open_set_correct_identification_rate": open_set_correct,
    }


def save_robustness_performance_figure(robustness_rows: list[dict]) -> None:
    labels = [row["display_name"] for row in robustness_rows]
    x = np.arange(len(robustness_rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        [float(row["identification_rate"]) for row in robustness_rows],
        width=width,
        label="Identification rate",
    )
    ax.bar(
        x + width / 2,
        [float(row["known_acceptance_rate"]) for row in robustness_rows],
        width=width,
        label="Known acceptance rate",
    )
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Open-set identification robustness performance")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "robustness_performance.png", dpi=160)
    plt.close(fig)


def run_identification() -> dict:
    recognizer, train_samples, known_test, unknown_test = build_recognition_data()
    metadata = gallery_metadata(recognizer, train_samples)
    query_records = []
    for sample in known_test:
        query_records.append(
            {
                "label": sample["label"],
                "path": sample["path"],
                "is_known": True,
                "embeddings": recognizer._extract_color_and_gray_embeddings(sample["aligned"]),
                "aligned": sample["aligned"],
            }
        )
    for sample in unknown_test:
        query_records.append(
            {
                "label": sample["label"],
                "path": sample["path"],
                "is_known": False,
                "embeddings": recognizer._extract_color_and_gray_embeddings(sample["aligned"]),
                "aligned": sample["aligned"],
            }
        )

    ks = [1, 3, 5, 7]
    distance_thresholds = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35]
    posterior_thresholds = [0.00, 0.34, 0.50, 0.60, 0.67, 0.75, 0.80, 1.00]
    rules = ["distance", "posterior", "both"]
    rows = []
    full_results = []
    for k in ks:
        for rule in rules:
            for tau_d in distance_thresholds:
                for tau_p in posterior_thresholds:
                    if rule == "distance" and tau_p != 0.00:
                        continue
                    if rule == "posterior" and tau_d != distance_thresholds[0]:
                        continue
                    result = evaluate_recognition_config(recognizer, query_records, k, rule, tau_d, tau_p)
                    full_results.append(result)
                    row = {key: value for key, value in result.items() if not key.startswith("_")}
                    row.update(metadata)
                    row["known_test_samples"] = len(known_test)
                    row["unknown_test_samples"] = len(unknown_test)
                    row["distance_threshold_active"] = rule in ("distance", "both")
                    row["posterior_threshold_active"] = rule in ("posterior", "both")
                    row["distance_threshold"] = tau_d if row["distance_threshold_active"] else "not_used"
                    row["posterior_threshold"] = tau_p if row["posterior_threshold_active"] else "not_used"
                    rows.append(row)

    write_csv(RESULTS / "identification_results.csv", rows)

    def result_key(result: dict) -> tuple[float, float, float, float]:
        balanced = (result["known_identification_rate"] + result["unknown_rejection_rate"]) / 2.0
        return (
            balanced,
            result["known_identification_rate"],
            result["unknown_rejection_rate"],
            -result["known_false_reject_rate"],
        )

    best = max(
        full_results,
        key=result_key,
    )
    best_by_k_rule = {}
    for k in ks:
        for rule in rules:
            candidates = [result for result in full_results if result["k"] == k and result["rule"] == rule]
            best_by_k_rule[(k, rule)] = max(candidates, key=result_key)

    k_gt1_posterior = max(
        [result for result in full_results if result["k"] > 1 and result["rule"] == "posterior"],
        key=result_key,
    )
    k_gt1_combined = max(
        [result for result in full_results if result["k"] > 1 and result["rule"] == "both"],
        key=result_key,
    )
    k1_posterior_values = sorted(
        {
            raw_knn_prediction(
                np.asarray(recognizer.labels, dtype=object),
                np.asarray(recognizer.embeddings, dtype=np.float32),
                record["embeddings"],
                1,
            )[1]
            for record in query_records
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    marker_by_rule = {"distance": "o", "posterior": "s", "both": "^"}
    color_by_k = {1: "#1f77b4", 3: "#ff7f0e", 5: "#2ca02c", 7: "#d62728"}
    for k in ks:
        for rule in rules:
            subset = [result for result in full_results if result["rule"] == rule and int(result["k"]) == k]
            ax.scatter(
                [float(result["unknown_rejection_rate"]) for result in subset],
                [float(result["known_identification_rate"]) for result in subset],
                label=f"k={k}, {rule}",
                alpha=0.45,
                marker=marker_by_rule[rule],
                color=color_by_k[k],
                s=24,
            )
    ax.set_xlabel("Unknown rejection rate")
    ax.set_ylabel("Known identification rate")
    ax.set_title("Threshold trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    comparison = [best_by_k_rule[(k, rule)] for k in ks for rule in rules]
    x = np.arange(len(comparison))
    width = 0.38
    ax.bar(
        x - width / 2,
        [result["known_identification_rate"] for result in comparison],
        width=width,
        label="Known ID",
    )
    ax.bar(
        x + width / 2,
        [result["unknown_rejection_rate"] for result in comparison],
        width=width,
        label="Unknown rejection",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={result['k']}\n{result['rule']}" for result in comparison], rotation=45, ha="right")
    ax.set_title("Best setting per k/rule")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "threshold_tradeoff.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    balanced_scores = [
        (result["known_identification_rate"] + result["unknown_rejection_rate"]) / 2.0
        for result in comparison
    ]
    ax.bar(np.arange(len(comparison)), balanced_scores, color=[color_by_k[result["k"]] for result in comparison])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(comparison)))
    ax.set_xticklabels([f"k={result['k']}\n{result['rule']}" for result in comparison], rotation=45, ha="right")
    ax.set_ylabel("Mean of known ID and unknown rejection")
    ax.set_title("Open-set identification k/rule comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "rule_k_comparison.png", dpi=160)
    plt.close(fig)

    robustness_samples = known_test
    robustness_rows = [
        evaluate_robustness_transform(
            recognizer,
            robustness_samples,
            transform,
            int(best["k"]),
            str(best["rule"]),
            float(best["distance_threshold"]),
            float(best["posterior_threshold"]),
        )
        for transform, _title in ROBUSTNESS_TRANSFORMS
    ]
    write_csv(RESULTS / "robustness_results.csv", robustness_rows)
    save_robustness_performance_figure(robustness_rows)

    if known_test:
        base = known_test[0]["aligned"]
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for ax, (transform, title) in zip(axes, ROBUSTNESS_TRANSFORMS):
            ax.imshow(to_rgb(transform_aligned(base, transform)))
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIGURES / "robustness_examples.png", dpi=160)
        plt.close(fig)

    return {
        "train_samples": train_samples,
        "known_test": known_test,
        "unknown_test": unknown_test,
        "metadata": metadata,
        "rows": rows,
        "best": {key: value for key, value in best.items() if not key.startswith("_")},
        "best_by_k_rule": {
            f"k={k},{rule}": {key: value for key, value in result.items() if not key.startswith("_")}
            for (k, rule), result in best_by_k_rule.items()
        },
        "k_gt1_posterior": {
            key: value for key, value in k_gt1_posterior.items() if not key.startswith("_")
        },
        "k_gt1_combined": {
            key: value for key, value in k_gt1_combined.items() if not key.startswith("_")
        },
        "k1_posterior_values": k1_posterior_values,
        "robustness_rows": robustness_rows,
    }


def cluster_purity(memberships: list[int], labels: list[str]) -> tuple[float, dict[int, Counter]]:
    cluster_counts: dict[int, Counter] = defaultdict(Counter)
    for cluster_id, label in zip(memberships, labels):
        cluster_counts[int(cluster_id)][label] += 1
    correct = sum(max(counter.values()) for counter in cluster_counts.values() if counter)
    return correct / len(labels) if labels else 0.0, cluster_counts


def run_clustering() -> dict:
    labels = CLUSTER_LABELS
    samples = get_aligned_samples(
        Config.TRAIN_DATA,
        labels,
        count_per_label=CLUSTER_FACES_PER_LABEL,
        scan_limit=160,
    )
    true_labels = [sample["label"] for sample in samples]
    sample_counts = Counter(true_labels)
    rows = []
    histories = {}
    clusters_by_seed = {}
    for seed in CLUSTER_SEEDS:
        cluster = FaceClustering(num_clusters=len(labels), max_iter=60)
        cluster.num_clusters = len(labels)
        cluster.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)
        cluster.cluster_center = np.empty((len(labels), FaceNet.embedding_dimensionality), dtype=np.float32)
        cluster.cluster_membership = []
        cluster.objective_history = []
        cluster.n_iter_ = 0
        cluster.converged_ = False
        cluster.is_fitted_ = False
        cluster.random_seed = seed
        for sample in samples:
            cluster.partial_fit(sample["aligned"])
        cluster.fit()
        final_objective = cluster.objective_history[-1] if cluster.objective_history else math.inf
        purity, counts = cluster_purity(cluster.cluster_membership, true_labels)
        size_text = "; ".join(
            f"{cluster_id}:{sum(counts.get(cluster_id, Counter()).values())}"
            for cluster_id in range(len(labels))
        )
        label_text = "; ".join(
            f"{cluster_id}:{dict(counts.get(cluster_id, Counter()))}"
            for cluster_id in range(len(labels))
        )
        empty_clusters = sum(
            1 for cluster_id in range(len(labels)) if sum(counts.get(cluster_id, Counter()).values()) == 0
        )
        rows.append(
            {
                "seed": seed,
                "samples": len(samples),
                "samples_by_identity": count_by_label_text(labels, sample_counts),
                "clusters": len(labels),
                "iterations": cluster.n_iter_,
                "converged": cluster.converged_,
                "final_objective": final_objective,
                "objective_history": ";".join(f"{value:.6f}" for value in cluster.objective_history),
                "objective_non_increasing": all(
                    cluster.objective_history[i + 1] <= cluster.objective_history[i] + 1e-8
                    for i in range(len(cluster.objective_history) - 1)
                ),
                "cluster_sizes": size_text,
                "empty_clusters": empty_clusters,
                "cluster_label_counts": label_text,
                "purity_analysis_only": purity,
            }
        )
        histories[seed] = cluster.objective_history

        clusters_by_seed[seed] = cluster

    valid_rows = [
        row
        for row in rows
        if int(row["empty_clusters"]) == 0 and bool(row["converged"])
    ]
    if valid_rows:
        representative = max(
            valid_rows,
            key=lambda row: (
                float(row["purity_analysis_only"]),
                -float(row["final_objective"]),
                -int(row["iterations"]),
            ),
        )
    else:
        representative = max(
            rows,
            key=lambda row: (
                -int(row["empty_clusters"]),
                bool(row["converged"]),
                float(row["purity_analysis_only"]),
                -float(row["final_objective"]),
            ),
        )
    for row in rows:
        row["representative_run"] = int(row["seed"]) == int(representative["seed"])

    write_csv(RESULTS / "clustering_results.csv", rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    for seed, history in histories.items():
        ax.plot(range(1, len(history) + 1), history, marker="o", label=f"seed {seed}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("K-means objective")
    ax.set_title("Face clustering objective convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "clustering_objective.png", dpi=160)
    plt.close(fig)

    representative_cluster = clusters_by_seed.get(int(representative["seed"]))
    if representative_cluster is not None and samples:
        test_samples = get_aligned_samples(Config.TEST_DATA, [samples[0]["label"]], count_per_label=1, scan_limit=80)
        query = test_samples[0] if test_samples else samples[0]
        cluster_id, distances = representative_cluster.predict(query["aligned"])
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        axes[0].imshow(to_rgb(query["aligned"]))
        axes[0].set_title(f"Seed {representative['seed']} query -> Cluster {cluster_id}")
        axes[0].axis("off")
        axes[1].bar(range(len(distances)), distances)
        axes[1].set_xlabel("Cluster id")
        axes[1].set_ylabel("Distance")
        axes[1].set_title("Re-identification distances")
        fig.tight_layout()
        fig.savefig(FIGURES / "reid_example.png", dpi=160)
        plt.close(fig)

    return {
        "rows": rows,
        "representative": representative,
        "samples_by_identity": count_by_label_text(labels, sample_counts),
    }


def run_dir() -> dict:
    false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)
    evaluation = OpenSetEvaluation(false_alarm_rate_range=false_alarm_rate_range)
    evaluation.prepare_input_data(Config.EVAL_TRAIN_DATA, Config.EVAL_TEST_DATA)
    results = evaluation.run()
    far = np.asarray(results["false_alarm_rates"], dtype=float)
    thresholds = np.asarray(results["similarity_thresholds"], dtype=float)
    identification = np.asarray(results["identification_rates"], dtype=float)
    rows = [
        {
            "false_alarm_rate": far_value,
            "identification_rate": id_value,
            "similarity_threshold": threshold,
        }
        for far_value, id_value, threshold in zip(far, identification, thresholds)
    ]
    write_csv(RESULTS / "dir_results.csv", rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(far, identification, linewidth=2.5)
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Identification rate")
    ax.set_title("DIR evaluation DIR curve")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "dir_curve.png", dpi=160)
    plt.close(fig)

    mask_a = far <= 0.01
    if np.any(mask_a):
        candidates = np.flatnonzero(mask_a)
        idx_a = candidates[np.argmax(identification[candidates])]
        condition_a = {
            "false_alarm_rate": far[idx_a],
            "identification_rate": identification[idx_a],
            "similarity_threshold": thresholds[idx_a],
        }
    else:
        condition_a = None

    mask_b = identification >= 0.90
    if np.any(mask_b):
        candidates = np.flatnonzero(mask_b)
        idx_b = candidates[np.argmin(far[candidates])]
        condition_b = {
            "false_alarm_rate": far[idx_b],
            "identification_rate": identification[idx_b],
            "similarity_threshold": thresholds[idx_b],
        }
    else:
        condition_b = None

    return {
        "rows": rows,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "ident_min": float(np.min(identification)),
        "ident_max": float(np.max(identification)),
    }


def main() -> None:
    ensure_dirs()
    start = time.perf_counter()

    summary = {
        "tracking": run_tracking(),
        "identification": run_identification(),
        "clustering": run_clustering(),
        "dir": run_dir(),
    }

    (RESULTS / "summary.csv").write_text(
        "experiment,recommended\n"
        + "\n".join(f"{name},{value}" for name, value in summary.items()),
        encoding="utf-8",
    )

    print(f"Benchmark artefacts written to {RESULTS}")
    print(f"Total runtime: {time.perf_counter() - start:.2f}s")
    print(f"  tracking:       {summary['tracking']}")
    print(f"  identification: {summary['identification']}")
    print(f"  clustering:     {summary['clustering']}")
    print(f"  dir:            {summary['dir']}")


if __name__ == "__main__":
    main()
