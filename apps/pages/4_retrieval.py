"""Streamlit dashboard for Computer Vision Project Sheet 3 experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


import pandas as pd
import streamlit as st

from cvkit.retrieval.experiments import aqe, codebook_ablation, intranorm, pca_whitening


def parse_integer_list(value: str, label: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated integer list") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{label} must contain positive integers")
    return values


st.set_page_config(page_title="VLAD Individual Experiments", layout="wide")
st.title("Computer Vision Project Sheet 3 — Individual Experiments")
st.caption("CPU-only VLAD experiments with parameter-aware disk caching.")

with st.sidebar:
    st.header("Dataset")
    in_train = st.text_input("Train image folder", "icdar2017-training-binary")
    in_test = st.text_input("Test image folder", "ScriptNet-HistoricalWI-2017-binarized")
    labels_train = st.text_input("Train label file", "icdar17_labels_train.txt")
    labels_test = st.text_input("Test label file", "icdar17_labels_test.txt")
    suffix_train = st.text_input("Train suffix", ".png")
    suffix_test = st.text_input("Test suffix", ".jpg")

    st.header("Storage")
    cache_dir = st.text_input("Cache directory", "cache")
    results_dir = st.text_input("Results directory", "results")

    st.header("Common parameters")
    n_clusters = st.number_input("K", min_value=2, value=100, step=1)
    max_descriptors = st.number_input(
        "Max descriptors", min_value=1, value=500000, step=10000
    )
    max_dictionary_files = st.number_input(
        "Max dictionary files (0 = all)", min_value=0, value=0, step=1
    )
    powernorm = st.checkbox("Power normalization", value=True)
    use_esvm = st.checkbox("E-SVM", value=False)
    svm_c = st.number_input("E-SVM C", min_value=0.0001, value=1000.0)
    overwrite = st.checkbox("Overwrite caches", value=False)

st.subheader("Select experiments")
selection_columns = st.columns(4)
with selection_columns[0]:
    run_intranorm = st.checkbox("Intra-normalization", value=True)
with selection_columns[1]:
    run_pca = st.checkbox("PCA whitening", value=True)
with selection_columns[2]:
    run_aqe = st.checkbox("AQE / re-ranking", value=True)
with selection_columns[3]:
    run_ablation = st.checkbox("Codebook size ablation", value=False)

parameter_columns = st.columns(3)
with parameter_columns[0]:
    pca_dims_text = st.text_input("PCA dims", "2048,4096")
with parameter_columns[1]:
    aqe_ks_text = st.text_input("AQE ks", "5,10,20")
with parameter_columns[2]:
    k_values_text = st.text_input("K values", "64,100,128,200")


def common_config() -> dict:
    return {
        "in_train": in_train,
        "in_test": in_test,
        "labels_train": labels_train,
        "labels_test": labels_test,
        "suffix_train": suffix_train,
        "suffix_test": suffix_test,
        "n_clusters": int(n_clusters),
        "max_descriptors": int(max_descriptors),
        "max_dictionary_files": int(max_dictionary_files),
        "powernorm": powernorm,
        "esvm": use_esvm,
        "C": float(svm_c),
        "overwrite": overwrite,
        "cache_dir": cache_dir,
        "results_dir": results_dir,
    }


if st.button("Run selected experiments", type="primary"):
    selected = []
    try:
        if run_intranorm:
            selected.append(("Intra-normalization", intranorm.run_experiment, {}))
        if run_pca:
            selected.append(
                (
                    "PCA whitening",
                    pca_whitening.run_experiment,
                    {"pca_dims": parse_integer_list(pca_dims_text, "PCA dims")},
                )
            )
        if run_aqe:
            selected.append(
                (
                    "AQE / re-ranking",
                    aqe.run_experiment,
                    {"aqe_ks": parse_integer_list(aqe_ks_text, "AQE ks")},
                )
            )
        if run_ablation:
            selected.append(
                (
                    "Codebook size ablation",
                    codebook_ablation.run_experiment,
                    {"k_values": parse_integer_list(k_values_text, "K values")},
                )
            )
    except ValueError as exc:
        st.error(str(exc))
        selected = []

    if not selected:
        st.warning("Select at least one experiment and provide valid parameters.")
    else:
        all_rows = []
        progress = st.progress(0, text="Starting experiments")
        for index, (name, runner, extra) in enumerate(selected):
            progress.progress(index / len(selected), text=f"Running {name}")
            try:
                with st.spinner(f"Running {name}..."):
                    result = runner({**common_config(), **extra})
                all_rows.extend(result.get("results", [result]))
                st.success(f"{name} completed: {result['result_file']}")
            except Exception as exc:  # Keep other selected experiments running.
                st.error(f"{name} failed: {type(exc).__name__}: {exc}")
        progress.progress(1.0, text="Finished")

        if all_rows:
            comparison = pd.DataFrame(all_rows)
            comparison_file = Path(results_dir) / "streamlit_comparison.csv"
            comparison_file.parent.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(comparison_file, index=False)

            st.subheader("Comparison")
            st.dataframe(
                comparison[
                    ["experiment", "method", "top1", "map", "runtime_seconds"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Saved to {comparison_file}")

            chart_data = comparison.copy()
            chart_data["label"] = (
                chart_data["experiment"].astype(str)
                + " | "
                + chart_data["method"].astype(str)
            )
            chart_data = chart_data.set_index("label")
            chart_columns = st.columns(2)
            with chart_columns[0]:
                st.subheader("mAP")
                st.bar_chart(chart_data[["map"]])
            with chart_columns[1]:
                st.subheader("Top-1")
                st.bar_chart(chart_data[["top1"]])
