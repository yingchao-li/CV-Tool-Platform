import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import loadmat

from cvkit.geometry.dataset import valid_points_with_indices, list_mat_files
from cvkit.geometry.measurement import estimate_box_dimensions, get_box_top_corners_pca
from cvkit.geometry.masks import (
    create_mask_from_indices,
    filter_floor_mask,
    get_non_floor_mask,
    get_points_from_mask,
    largest_connected_component,
)
from cvkit.geometry.mlesac import mlesac_plane
from cvkit.geometry.preemptive import preemptive_ransac_plane
from cvkit.geometry.ransac import ransac_plane
from cvkit.geometry.render import get_rectangle_from_mask


METHOD_NAMES = [
    "Ordinary RANSAC",
    "MLESAC",
    "Preemptive RANSAC",
]


def load_from_mat_dict(data):
    """Extract the A, D, and PC arrays from a loadmat result dictionary."""

    a_key = None
    d_key = None
    pc_key = None

    for key in data.keys():
        if key.startswith("__"):
            continue
        if key.startswith("amplitudes"):
            a_key = key
        elif key.startswith("distances"):
            d_key = key
        elif key.startswith("cloud"):
            pc_key = key

    if a_key is None or d_key is None or pc_key is None:
        raise ValueError(
            "Could not find amplitude/distance/cloud fields in this mat file."
        )

    a = np.squeeze(data[a_key])
    d = np.squeeze(data[d_key])
    pc = data[pc_key]

    if a.shape != d.shape:
        raise ValueError(f"A and D shape mismatch: {a.shape} vs {d.shape}")
    if pc.ndim != 3 or pc.shape[2] != 3:
        raise ValueError(f"PC should be HxWx3, got {pc.shape}")
    if a.shape != pc.shape[:2]:
        raise ValueError(f"A/D and PC shape mismatch: {a.shape} vs {pc.shape[:2]}")

    return a, d, pc


def load_mat_from_file(path_or_buffer):
    """Read a .mat file from a path or binary buffer and return A/D/PC."""

    data = loadmat(path_or_buffer)
    return load_from_mat_dict(data)


def percentile_range(image, low=1, high=99):
    """Compute a percentile display range over the finite pixels of an image."""

    valid = image[np.isfinite(image)]
    if valid.size == 0:
        return None, None
    return np.percentile(valid, low), np.percentile(valid, high)


def draw_gray_image(image, title):
    """Draw a 2D image in grayscale."""

    fig, ax = plt.subplots(figsize=(6, 4))
    vmin, vmax = percentile_range(image, 1, 99)
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def draw_mask(mask, title):
    """Draw a boolean mask image."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mask, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def draw_final_overlay(floor_mask, box_top_mask, method_name):
    """Draw the final detection overlay."""

    h, w = floor_mask.shape
    result = np.zeros((h, w, 3), dtype=float)

    floor_color = np.array([0.55, 1.00, 0.40])
    non_floor_color = np.array([0.00, 0.00, 0.45])
    box_top_color = np.array([0.60, 0.00, 0.00])

    result[:, :] = non_floor_color
    result[floor_mask] = floor_color
    result[box_top_mask] = box_top_color

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result)
    ax.set_title(f"{method_name} Final Detection Overlay")

    if np.any(box_top_mask):
        try:
            corners = get_rectangle_from_mask(box_top_mask)
        except ValueError:
            corners = None

        if corners is not None:
            closed_corners = np.vstack([corners, corners[0]])

            top_point = corners[np.argmin(corners[:, 1])]
            bottom_point = corners[np.argmax(corners[:, 1])]
            left_point = corners[np.argmin(corners[:, 0])]
            right_point = corners[np.argmax(corners[:, 0])]

            ax.plot(
                closed_corners[:, 0],
                closed_corners[:, 1],
                color="cyan",
                linewidth=3,
            )
            ax.scatter(
                corners[:, 0],
                corners[:, 1],
                color="cyan",
                s=30,
            )

            ax.text(
                top_point[0],
                top_point[1] - 35,
                "top",
                color="black",
                fontsize=12,
                ha="center",
            )
            ax.text(
                bottom_point[0],
                bottom_point[1] + 35,
                "bottom",
                color="black",
                fontsize=12,
                ha="center",
            )
            ax.text(
                left_point[0] - 45,
                left_point[1],
                "left",
                color="black",
                fontsize=12,
                ha="center",
            )
            ax.text(
                right_point[0] + 45,
                right_point[1],
                "right",
                color="black",
                fontsize=12,
                ha="center",
            )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def format_plane_equation(plane):
    """Format a plane equation as ax + by + cz + d = 0."""

    a, b, c, d = plane
    return f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0"


def normal_d_to_equation_plane(model):
    """Convert normal · x = d into ax + by + cz + d = 0."""

    normal, d = model
    return np.array([normal[0], normal[1], normal[2], -d])


def point_to_model_distances(points, model):
    """Compute point-to-plane distances for planes in normal · x = d form."""

    normal, d = model
    normal = np.asarray(normal, dtype=float)
    points = np.asarray(points, dtype=float)

    normal_norm = np.linalg.norm(normal)
    if (not np.isfinite(normal_norm)) or normal_norm < 1e-12:
        raise ValueError("Plane normal must not be zero.")

    finite_points = np.all(np.isfinite(points), axis=1)
    distances = np.full(points.shape[0], np.inf, dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        finite_distances = np.abs(points[finite_points] @ normal - d) / normal_norm

    finite_distances[~np.isfinite(finite_distances)] = np.inf
    distances[finite_points] = finite_distances

    return distances


def residual_rmse(points, model, inliers):
    """Return the RMSE residual of the inliers to the fitted plane."""

    distances = point_to_model_distances(points, model)
    residuals = distances[inliers]
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(residuals ** 2)))


def format_score_value(value):
    """Format RANSAC outlier counts and costs consistently."""

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}"


def score_summary(floor_result, box_result):
    """Compress the floor and box-top scores into one comparison-table field."""

    floor_score = format_score_value(floor_result["score"])
    box_score = format_score_value(box_result["score"])
    return (
        f"Floor {floor_result['score_label']}: {floor_score}; "
        f"Box {box_result['score_label']}: {box_score}"
    )


def run_plane_detector(
    points,
    method_name,
    threshold,
    random_seed,
    ransac_iterations,
    mlesac_iterations,
    gamma,
    preemptive_m,
    preemptive_b,
    preemptive_sample_size,
):
    """Fit a single plane with the method selected by the user."""

    if method_name == "Ordinary RANSAC":
        np.random.seed(random_seed)
        model, inliers, score = ransac_plane(
            points,
            threshold=threshold,
            max_iterations=ransac_iterations,
            random_seed=random_seed,
        )
        return {
            "model": model,
            "equation_plane": normal_d_to_equation_plane(model),
            "inliers": inliers,
            "score_label": "Outlier Count",
            "score": int(score),
        }

    if method_name == "MLESAC":
        model, inliers, cost = mlesac_plane(
            points,
            num_iterations=mlesac_iterations,
            epsilon=threshold,
            gamma=gamma,
            random_seed=random_seed,
        )
        return {
            "model": model,
            "equation_plane": normal_d_to_equation_plane(model),
            "inliers": inliers,
            "score_label": "MLESAC Cost",
            "score": float(cost),
        }

    if method_name == "Preemptive RANSAC":
        model, inliers, cost = preemptive_ransac_plane(
            points,
            num_hypotheses=preemptive_m,
            batch_size=preemptive_b,
            sample_size=preemptive_sample_size,
            threshold=threshold,
            random_seed=random_seed,
        )
        return {
            "model": model,
            "equation_plane": normal_d_to_equation_plane(model),
            "inliers": inliers,
            "score_label": "Preemptive Cost",
            "score": int(cost),
        }

    raise ValueError(f"Unknown detection method: {method_name}")


def run_detection_pipeline(
    pc,
    points,
    rows,
    cols,
    method_name,
    floor_threshold,
    box_threshold,
    random_seed,
    ransac_iterations,
    mlesac_iterations,
    floor_gamma,
    box_gamma,
    preemptive_m,
    preemptive_b,
    preemptive_sample_size,
    min_floor_size,
    max_hole_size,
    size_method,
):
    """Run the full detection pipeline for one method and return all results."""

    start_time = time.perf_counter()

    floor_result = run_plane_detector(
        points,
        method_name=method_name,
        threshold=floor_threshold,
        random_seed=random_seed,
        ransac_iterations=ransac_iterations,
        mlesac_iterations=mlesac_iterations,
        gamma=floor_gamma,
        preemptive_m=preemptive_m,
        preemptive_b=preemptive_b,
        preemptive_sample_size=preemptive_sample_size,
    )
    floor_model = floor_result["model"]
    floor_inliers = floor_result["inliers"]

    raw_floor_mask = create_mask_from_indices(
        pc.shape[:2],
        rows,
        cols,
        floor_inliers,
    )
    if not np.any(raw_floor_mask):
        raise RuntimeError(f"{method_name} did not find any floor inliers.")

    filtered_floor_mask = filter_floor_mask(
        raw_floor_mask,
        min_floor_size=min_floor_size,
        max_hole_size=max_hole_size,
    )

    non_floor_mask = get_non_floor_mask(filtered_floor_mask)
    non_floor_points, non_floor_rows, non_floor_cols = get_points_from_mask(
        pc,
        non_floor_mask,
    )
    if non_floor_points.shape[0] < 3:
        raise RuntimeError("Not enough non-floor points to detect box top plane.")

    box_result = run_plane_detector(
        non_floor_points,
        method_name=method_name,
        threshold=box_threshold,
        random_seed=random_seed + 1,
        ransac_iterations=ransac_iterations,
        mlesac_iterations=mlesac_iterations,
        gamma=box_gamma,
        preemptive_m=preemptive_m,
        preemptive_b=preemptive_b,
        preemptive_sample_size=preemptive_sample_size,
    )
    box_model = box_result["model"]
    box_inliers = box_result["inliers"]

    raw_box_mask = create_mask_from_indices(
        pc.shape[:2],
        non_floor_rows,
        non_floor_cols,
        box_inliers,
    )
    if not np.any(raw_box_mask):
        raise RuntimeError(f"{method_name} did not find any box-top inliers.")

    box_top_mask = largest_connected_component(raw_box_mask)
    if not np.any(box_top_mask):
        raise RuntimeError("Box top mask is empty after connected-component filtering.")

    height, length, width = estimate_box_dimensions(
        pc,
        floor_model,
        box_model,
        box_top_mask,
        method=size_method,
    )
    corners = get_box_top_corners_pca(pc, box_top_mask)

    runtime = time.perf_counter() - start_time

    return {
        "method": method_name,
        "runtime": runtime,
        "floor_result": floor_result,
        "box_result": box_result,
        "floor_model": floor_model,
        "box_model": box_model,
        "floor_inliers": floor_inliers,
        "box_inliers": box_inliers,
        "floor_rmse": residual_rmse(points, floor_model, floor_inliers),
        "box_rmse": residual_rmse(non_floor_points, box_model, box_inliers),
        "raw_floor_mask": raw_floor_mask,
        "filtered_floor_mask": filtered_floor_mask,
        "raw_box_mask": raw_box_mask,
        "box_top_mask": box_top_mask,
        "height": height,
        "length": length,
        "width": width,
        "corners": corners,
    }


def comparison_row(result):
    """Build one row of the method-comparison table."""

    return {
        "Method": result["method"],
        "Runtime (s)": f"{result['runtime']:.4f}",
        "Floor inliers": int(np.sum(result["floor_inliers"])),
        "Box top inliers": int(np.sum(result["box_inliers"])),
        "Floor residual RMSE": f"{result['floor_rmse']:.6f}",
        "Box top residual RMSE": f"{result['box_rmse']:.6f}",
        "Height / Length / Width": (
            f"{result['height']:.4f} / "
            f"{result['length']:.4f} / "
            f"{result['width']:.4f}"
        ),
        "Score / Cost": score_summary(
            result["floor_result"],
            result["box_result"],
        ),
        "Final overlay visualization": "Shown below",
    }


def build_qualitative_comparison(results):
    """Derive a brief qualitative comparison from the numeric metrics."""

    if not results:
        return []

    fastest = min(results, key=lambda item: item["runtime"])
    best_floor_rmse = min(results, key=lambda item: item["floor_rmse"])
    best_box_rmse = min(results, key=lambda item: item["box_rmse"])
    most_floor_inliers = max(results, key=lambda item: np.sum(item["floor_inliers"]))
    most_box_inliers = max(results, key=lambda item: np.sum(item["box_inliers"]))

    return [
        (
            f"Fastest runtime: {fastest['method']} "
            f"({fastest['runtime']:.4f} s)."
        ),
        (
            f"Lowest floor residual RMSE: {best_floor_rmse['method']} "
            f"({best_floor_rmse['floor_rmse']:.6f})."
        ),
        (
            f"Lowest box-top residual RMSE: {best_box_rmse['method']} "
            f"({best_box_rmse['box_rmse']:.6f})."
        ),
        (
            f"Most floor inliers: {most_floor_inliers['method']} "
            f"({int(np.sum(most_floor_inliers['floor_inliers']))})."
        ),
        (
            f"Most box-top inliers: {most_box_inliers['method']} "
            f"({int(np.sum(most_box_inliers['box_inliers']))})."
        ),
        (
            "Use the final overlays below for the qualitative part: a cleaner "
            "box-top region and a cyan boundary aligned with the visible top "
            "surface indicate the better segmentation."
        ),
    ]


def parameter_study_row(result):
    """Extract the numeric metrics shared by the parameter studies."""

    return {
        "Runtime (s)": result["runtime"],
        "Floor inliers": int(np.sum(result["floor_inliers"])),
        "Box top inliers": int(np.sum(result["box_inliers"])),
        "Floor residual RMSE": result["floor_rmse"],
        "Box top residual RMSE": result["box_rmse"],
        "Height (m)": result["height"],
        "Length (m)": result["length"],
        "Width (m)": result["width"],
        "Floor score/cost": result["floor_result"]["score"],
        "Box score/cost": result["box_result"]["score"],
    }


def build_mlesac_epsilon_analysis(study_results):
    """Summarize trends across the different epsilon experiments."""

    if not study_results:
        return []

    ordered = sorted(study_results, key=lambda item: item["floor_epsilon"])
    tightest = ordered[0]
    loosest = ordered[-1]
    best_floor = min(ordered, key=lambda item: item["result"]["floor_rmse"])
    best_box = min(ordered, key=lambda item: item["result"]["box_rmse"])

    heights = [item["result"]["height"] for item in ordered]
    lengths = [item["result"]["length"] for item in ordered]
    widths = [item["result"]["width"] for item in ordered]

    return [
        (
            "Changing epsilon from the tightest to the loosest setting changed "
            f"floor inliers from {int(np.sum(tightest['result']['floor_inliers']))} "
            f"to {int(np.sum(loosest['result']['floor_inliers']))}, and box-top "
            f"inliers from {int(np.sum(tightest['result']['box_inliers']))} to "
            f"{int(np.sum(loosest['result']['box_inliers']))}."
        ),
        (
            f"Lowest floor residual RMSE: {best_floor['label']} "
            f"({best_floor['result']['floor_rmse']:.6f})."
        ),
        (
            f"Lowest box-top residual RMSE: {best_box['label']} "
            f"({best_box['result']['box_rmse']:.6f})."
        ),
        (
            "Estimated dimension ranges across epsilon settings: "
            f"height {min(heights):.4f}-{max(heights):.4f} m, "
            f"length {min(lengths):.4f}-{max(lengths):.4f} m, and "
            f"width {min(widths):.4f}-{max(widths):.4f} m."
        ),
        (
            "A larger epsilon normally accepts more points, but it may also "
            "increase residual error or include points from nearby surfaces. "
            "The preferred epsilon should balance support, RMSE, dimension "
            "stability, and the final segmentation."
        ),
    ]


def build_preemptive_parameter_analysis(study_results):
    """Summarize trends across the different M/B experiments."""

    if not study_results:
        return []

    fastest = min(study_results, key=lambda item: item["result"]["runtime"])
    best_floor = min(study_results, key=lambda item: item["result"]["floor_rmse"])
    best_box = min(study_results, key=lambda item: item["result"]["box_rmse"])
    most_box = max(
        study_results,
        key=lambda item: np.sum(item["result"]["box_inliers"]),
    )

    return [
        (
            f"Fastest configuration: M={fastest['M']}, B={fastest['B']} "
            f"({fastest['result']['runtime']:.4f} s)."
        ),
        (
            f"Lowest floor residual RMSE: M={best_floor['M']}, "
            f"B={best_floor['B']} "
            f"({best_floor['result']['floor_rmse']:.6f})."
        ),
        (
            f"Lowest box-top residual RMSE: M={best_box['M']}, "
            f"B={best_box['B']} "
            f"({best_box['result']['box_rmse']:.6f})."
        ),
        (
            f"Most box-top inliers: M={most_box['M']}, B={most_box['B']} "
            f"({int(np.sum(most_box['result']['box_inliers']))})."
        ),
        (
            "Increasing M provides more initial model candidates and may improve "
            "the probability of including a strong plane, but it increases the "
            "first evaluation rounds. Increasing B gives each pruning decision "
            "more evidence, but also increases the work performed before weak "
            "hypotheses are removed."
        ),
        (
            "The best M/B pair should therefore be selected using runtime, "
            "residual error, dimension stability, and visual correctness rather "
            "than using the final preemptive cost alone."
        ),
    ]


st.set_page_config(page_title="Box Detection Method Tuner", layout="wide")
st.title("Box Detection Method Tuner")
st.caption(
    "Run Ordinary RANSAC, MLESAC, and Preemptive RANSAC together for direct "
    "runtime, inlier, residual, size, score, and visualization comparison."
)

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input", ["Use local data file", "Upload .mat file"])

    selected_path = None
    uploaded = None

    if source == "Use local data file":
        if os.path.exists("data"):
            mat_files = list_mat_files("data")
        else:
            mat_files = []
        if not mat_files:
            st.warning("No .mat file found in ./data")
        else:
            selected_path = st.selectbox("Select .mat file", mat_files)
    else:
        uploaded = st.file_uploader("Upload .mat", type=["mat"])

    st.header("Detection Methods")
    st.write("Runs all three methods in one click.")
    random_seed = st.number_input("Random seed", value=42, step=1)

    st.header("Inlier Thresholds")
    floor_threshold = st.slider("Floor threshold / epsilon", 0.001, 0.2, 0.05, 0.001)
    box_threshold = st.slider("Box top threshold / epsilon", 0.001, 0.05, 0.005, 0.001)

    ransac_iterations = 1000
    mlesac_iterations = 2000
    floor_gamma = 0.1
    box_gamma = 0.02
    preemptive_m = 200
    preemptive_b = 100
    preemptive_sample_size = 3

    with st.expander("Ordinary RANSAC Parameters", expanded=True):
        st.header("Ordinary RANSAC Parameters")
        ransac_iterations = st.slider("Max iterations", 100, 10000, 1000, 100)

    with st.expander("MLESAC Parameters", expanded=True):
        st.header("MLESAC Parameters")
        mlesac_iterations = st.slider("Sampling iterations", 100, 10000, 2000, 100)
        floor_gamma = st.slider("Floor gamma", 0.001, 0.5, 0.1, 0.001)
        box_gamma = st.slider("Box top gamma", 0.001, 0.2, 0.02, 0.001)

    with st.expander("Preemptive RANSAC Parameters", expanded=True):
        st.header("Preemptive RANSAC Parameters")
        preemptive_m = st.slider("Initial hypotheses M", 10, 2000, 200, 10)
        preemptive_b = st.slider("Batch size B", 10, 5000, 100, 10)
        preemptive_sample_size = st.slider("Sample size", 3, 10, 3, 1)

    st.header("Mask Cleaning Parameters")
    min_floor_size = st.slider("Min floor component size", 10, 5000, 100, 10)
    max_hole_size = st.slider("Max internal hole size", 10, 5000, 300, 10)

    st.header("Geometry Parameters")
    size_method = st.selectbox("Length/width estimation method", ["pca", "simple"])

    run = st.button("Run All Methods", type="primary")

    with st.expander("Parameter Study Settings", expanded=False):
        st.caption(
            "MLESAC uses fixed floor epsilon values 0.03, 0.05, 0.07, 0.10, "
            "and 0.50. Box-top epsilon is scaled by the current box/floor "
            "threshold ratio. Preemptive RANSAC evaluates every combination "
            "of the three M and three B values."
        )

        study_m_1 = st.number_input(
            "M value 1",
            min_value=10,
            max_value=5000,
            value=50,
            step=10,
        )
        study_m_2 = st.number_input(
            "M value 2",
            min_value=10,
            max_value=5000,
            value=200,
            step=10,
        )
        study_m_3 = st.number_input(
            "M value 3",
            min_value=10,
            max_value=5000,
            value=500,
            step=10,
        )

        study_b_1 = st.number_input(
            "B value 1",
            min_value=10,
            max_value=10000,
            value=50,
            step=10,
        )
        study_b_2 = st.number_input(
            "B value 2",
            min_value=10,
            max_value=10000,
            value=100,
            step=10,
        )
        study_b_3 = st.number_input(
            "B value 3",
            min_value=10,
            max_value=10000,
            value=200,
            step=10,
        )

    run_parameter_study = st.button("Run Parameter Studies")


if run:
    try:
        if floor_gamma <= floor_threshold:
            st.error("Floor gamma must be larger than floor threshold / epsilon.")
            st.stop()
        if box_gamma <= box_threshold:
            st.error("Box top gamma must be larger than box top threshold / epsilon.")
            st.stop()

        if source == "Use local data file":
            if not selected_path:
                st.error("Please choose a local .mat file first.")
                st.stop()
            a, d, pc = load_mat_from_file(selected_path)
            st.success(f"Loaded: {selected_path}")
        else:
            if uploaded is None:
                st.error("Please upload a .mat file first.")
                st.stop()
            buffer = io.BytesIO(uploaded.read())
            a, d, pc = load_mat_from_file(buffer)
            st.success(f"Loaded uploaded file: {uploaded.name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Image Height", int(pc.shape[0]))
        col2.metric("Image Width", int(pc.shape[1]))
        col3.metric("Valid Points", int(np.sum(pc[:, :, 2] != 0)))

        points, rows, cols = valid_points_with_indices(pc)
        if points.shape[0] < 3:
            st.error("Not enough valid points to run plane fitting.")
            st.stop()

        results = []
        failures = []

        for method_name in METHOD_NAMES:
            with st.spinner(f"Running {method_name}..."):
                try:
                    result = run_detection_pipeline(
                        pc=pc,
                        points=points,
                        rows=rows,
                        cols=cols,
                        method_name=method_name,
                        floor_threshold=floor_threshold,
                        box_threshold=box_threshold,
                        random_seed=int(random_seed),
                        ransac_iterations=ransac_iterations,
                        mlesac_iterations=mlesac_iterations,
                        floor_gamma=floor_gamma,
                        box_gamma=box_gamma,
                        preemptive_m=preemptive_m,
                        preemptive_b=preemptive_b,
                        preemptive_sample_size=preemptive_sample_size,
                        min_floor_size=min_floor_size,
                        max_hole_size=max_hole_size,
                        size_method=size_method,
                    )
                    results.append(result)
                except Exception as exc:
                    failures.append((method_name, exc))

        if not results:
            st.error("All methods failed. Adjust thresholds or mask parameters.")
            for method_name, exc in failures:
                st.exception(RuntimeError(f"{method_name}: {exc}"))
            st.stop()

        if failures:
            st.warning("Some methods failed and are excluded from the comparison table.")
            for method_name, exc in failures:
                st.error(f"{method_name}: {exc}")

        st.subheader("Comparison Table")
        comparison_df = pd.DataFrame([comparison_row(result) for result in results])
        st.dataframe(comparison_df, width="stretch", hide_index=True)

        st.subheader("Qualitative Comparison")
        for item in build_qualitative_comparison(results):
            st.write(f"- {item}")

        st.subheader("Plane Equations")
        for result in results:
            method_name = result["method"]
            floor_result = result["floor_result"]
            box_result = result["box_result"]
            st.code(
                "\n".join(
                    [
                        f"{method_name}",
                        f"Floor plane:  {format_plane_equation(floor_result['equation_plane'])}",
                        f"Box top plane: {format_plane_equation(box_result['equation_plane'])}",
                    ]
                ),
                language="text",
            )

        st.subheader("Input Images")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            draw_gray_image(a, "Amplitude")
        with img_col2:
            d_disp = d.copy().astype(float)
            d_disp[d_disp == 0] = np.nan
            draw_gray_image(d_disp, "Distance")

        st.subheader("Final Overlay Visualization")
        overlay_columns = st.columns(len(results))
        for column, result in zip(overlay_columns, results):
            with column:
                draw_final_overlay(
                    result["filtered_floor_mask"],
                    result["box_top_mask"],
                    result["method"],
                )

        st.subheader("Detailed Masks and Corners")
        for result in results:
            method_name = result["method"]
            with st.expander(f"{method_name} details"):
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    draw_mask(result["raw_floor_mask"], f"{method_name} Raw Floor")
                with m2:
                    draw_mask(result["filtered_floor_mask"], "Filtered Floor")
                with m3:
                    draw_mask(result["raw_box_mask"], f"{method_name} Raw Box Top")
                with m4:
                    draw_mask(result["box_top_mask"], "Largest Box Top")

                st.write("Estimated Box Top Corners (3D)")
                st.dataframe(result["corners"], width="stretch")

        st.code(
            "\n".join(
                [
                    "Required outputs are included in the comparison table:",
                    "Method",
                    "Runtime",
                    "Floor inliers",
                    "Box top inliers",
                    "Floor residual RMSE",
                    "Box top residual RMSE",
                    "Height / Length / Width",
                    "Score / Cost",
                    "Final overlay visualization",
                ]
            ),
            language="text",
        )

    except Exception as exc:
        st.exception(exc)
else:
    if not run_parameter_study:
        st.info(
            "Run the three-method comparison or the parameter sensitivity studies "
            "from the sidebar."
        )


if run_parameter_study:
    try:
        m_values = sorted(
            {
                int(study_m_1),
                int(study_m_2),
                int(study_m_3),
            }
        )
        b_values = sorted(
            {
                int(study_b_1),
                int(study_b_2),
                int(study_b_3),
            }
        )

        if len(m_values) != 3:
            st.error("Enter three different M values for the parameter study.")
            st.stop()
        if len(b_values) != 3:
            st.error("Enter three different B values for the parameter study.")
            st.stop()

        floor_epsilon_values = [0.03, 0.05, 0.07, 0.10, 0.50]
        box_to_floor_ratio = box_threshold / floor_threshold
        epsilon_settings = [
            (
                f"epsilon = {floor_epsilon:.2f}",
                floor_epsilon,
                floor_epsilon * box_to_floor_ratio,
            )
            for floor_epsilon in floor_epsilon_values
        ]

        max_floor_epsilon = max(setting[1] for setting in epsilon_settings)
        max_box_epsilon = max(setting[2] for setting in epsilon_settings)
        study_floor_gamma = max(float(floor_gamma), max_floor_epsilon + 0.10)
        study_box_gamma = max(float(box_gamma), max_box_epsilon + 0.01)

        if source == "Use local data file":
            if not selected_path:
                st.error("Please choose a local .mat file first.")
                st.stop()
            a, d, pc = load_mat_from_file(selected_path)
            st.success(f"Loaded for parameter studies: {selected_path}")
        else:
            if uploaded is None:
                st.error("Please upload a .mat file first.")
                st.stop()
            buffer = io.BytesIO(uploaded.read())
            a, d, pc = load_mat_from_file(buffer)
            st.success(f"Loaded for parameter studies: {uploaded.name}")

        points, rows, cols = valid_points_with_indices(pc)
        if points.shape[0] < 3:
            st.error("Not enough valid points to run the parameter studies.")
            st.stop()

        total_experiments = len(epsilon_settings) + len(m_values) * len(b_values)
        completed_experiments = 0
        progress = st.progress(0.0, text="Starting parameter studies...")

        st.header("MLESAC Epsilon Sensitivity Study")
        st.caption(
            "The first column explicitly shows floor epsilon. Box-top epsilon "
            "is scaled using the current box/floor threshold ratio. Study gamma "
            f"is held fixed at {study_floor_gamma:.2f} for the floor and "
            f"{study_box_gamma:.3f} for the box top; the random seed, iteration "
            "count, mask processing, and geometry settings also remain fixed."
        )

        mlesac_study_results = []
        mlesac_failures = []

        for label, floor_epsilon, box_epsilon in epsilon_settings:
            progress.progress(
                completed_experiments / total_experiments,
                text=f"Running MLESAC: {label}...",
            )
            try:
                result = run_detection_pipeline(
                    pc=pc,
                    points=points,
                    rows=rows,
                    cols=cols,
                    method_name="MLESAC",
                    floor_threshold=floor_epsilon,
                    box_threshold=box_epsilon,
                    random_seed=int(random_seed),
                    ransac_iterations=ransac_iterations,
                    mlesac_iterations=mlesac_iterations,
                    floor_gamma=study_floor_gamma,
                    box_gamma=study_box_gamma,
                    preemptive_m=preemptive_m,
                    preemptive_b=preemptive_b,
                    preemptive_sample_size=preemptive_sample_size,
                    min_floor_size=min_floor_size,
                    max_hole_size=max_hole_size,
                    size_method=size_method,
                )
                mlesac_study_results.append(
                    {
                        "label": label,
                        "floor_epsilon": floor_epsilon,
                        "box_epsilon": box_epsilon,
                        "result": result,
                    }
                )
            except Exception as exc:
                mlesac_failures.append((label, exc))

            completed_experiments += 1

        if mlesac_study_results:
            mlesac_rows = []
            for item in mlesac_study_results:
                row = {
                    "ε (floor)": item["floor_epsilon"],
                    "ε (box top, scaled)": item["box_epsilon"],
                }
                row.update(parameter_study_row(item["result"]))
                mlesac_rows.append(row)

            mlesac_df = pd.DataFrame(mlesac_rows).round(6)
            st.dataframe(mlesac_df, width="stretch", hide_index=True)

            st.subheader("MLESAC Epsilon Analysis")
            for item in build_mlesac_epsilon_analysis(mlesac_study_results):
                st.write(f"- {item}")

        for label, exc in mlesac_failures:
            st.warning(f"MLESAC {label} failed: {exc}")

        st.header("Preemptive RANSAC M/B Sensitivity Study")
        st.caption(
            "Every M/B combination uses the same thresholds, random seed, sample "
            "size, mask processing, and geometry settings."
        )

        preemptive_study_results = []
        preemptive_failures = []

        for study_m in m_values:
            for study_b in b_values:
                progress.progress(
                    completed_experiments / total_experiments,
                    text=f"Running Preemptive RANSAC: M={study_m}, B={study_b}...",
                )
                try:
                    result = run_detection_pipeline(
                        pc=pc,
                        points=points,
                        rows=rows,
                        cols=cols,
                        method_name="Preemptive RANSAC",
                        floor_threshold=floor_threshold,
                        box_threshold=box_threshold,
                        random_seed=int(random_seed),
                        ransac_iterations=ransac_iterations,
                        mlesac_iterations=mlesac_iterations,
                        floor_gamma=floor_gamma,
                        box_gamma=box_gamma,
                        preemptive_m=study_m,
                        preemptive_b=study_b,
                        preemptive_sample_size=preemptive_sample_size,
                        min_floor_size=min_floor_size,
                        max_hole_size=max_hole_size,
                        size_method=size_method,
                    )
                    preemptive_study_results.append(
                        {
                            "M": study_m,
                            "B": study_b,
                            "result": result,
                        }
                    )
                except Exception as exc:
                    preemptive_failures.append((study_m, study_b, exc))

                completed_experiments += 1

        progress.progress(1.0, text="Parameter studies complete.")

        if preemptive_study_results:
            preemptive_rows = []
            for item in preemptive_study_results:
                row = {
                    "M": item["M"],
                    "B": item["B"],
                }
                row.update(parameter_study_row(item["result"]))
                preemptive_rows.append(row)

            preemptive_df = pd.DataFrame(preemptive_rows).round(6)
            st.dataframe(preemptive_df, width="stretch", hide_index=True)

            st.subheader("Preemptive RANSAC M/B Analysis")
            for item in build_preemptive_parameter_analysis(
                preemptive_study_results
            ):
                st.write(f"- {item}")

            visualization_b = b_values[1]
            visualization_results = [
                item
                for item in preemptive_study_results
                if item["B"] == visualization_b
            ]
            visualization_results.sort(key=lambda item: item["M"])

            st.subheader("Three-M Visualization Comparison")
            st.caption(
                f"B is fixed at {visualization_b}. Only M changes, so the "
                "overlays show the effect of the initial hypothesis count."
            )

            visualization_columns = st.columns(len(m_values))
            visualization_lookup = {
                item["M"]: item for item in visualization_results
            }
            for column, study_m in zip(visualization_columns, m_values):
                with column:
                    item = visualization_lookup.get(study_m)
                    if item is None:
                        st.warning(
                            f"No visualization available for M={study_m}, "
                            f"B={visualization_b}."
                        )
                        continue

                    result = item["result"]
                    draw_final_overlay(
                        result["filtered_floor_mask"],
                        result["box_top_mask"],
                        f"Preemptive RANSAC M={study_m}, B={visualization_b}",
                    )
                    st.caption(
                        f"Runtime {result['runtime']:.4f} s | "
                        f"Floor RMSE {result['floor_rmse']:.6f} | "
                        f"Box RMSE {result['box_rmse']:.6f}"
                    )

            st.write(
                "Visual analysis: compare the continuity of the green floor, "
                "the compactness of the red box-top region, false-positive "
                "areas, and the alignment of the cyan boundary. A larger M may "
                "improve hypothesis diversity, but similar overlays across all "
                "three values indicate that the method is stable for this scene."
            )

        for study_m, study_b, exc in preemptive_failures:
            st.warning(
                f"Preemptive RANSAC M={study_m}, B={study_b} failed: {exc}"
            )

    except Exception as exc:
        st.exception(exc)
