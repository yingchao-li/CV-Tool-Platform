"""Benchmark viewer for the face-analysis module."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cvkit.paths import DOCS_DIR

ASSETS = DOCS_DIR / "assets" / "faces"
RESULTS = DOCS_DIR / "results" / "faces"

st.set_page_config(page_title="Faces · CVKit", page_icon="🙂", layout="wide")
st.title("Face analysis")
st.caption("Detection, tracking, recognition, clustering and open-set rejection")

st.markdown(
    """
The module wraps MTCNN for detection/alignment and a 128-D FaceNet ONNX
model for embeddings. Recognition uses a k-NN gallery; clustering uses
k-means with empty-cluster repair; open-set rejection compares SPL and MPL
pseudo-labelling strategies.
"""
)

tabs = st.tabs(["Tracking", "Identification", "Clustering", "DIR"])

with tabs[0]:
    st.image(ASSETS / "tracking_example.png", use_container_width=True)
    table = RESULTS / "tracking_results.csv"
    if table.exists():
        st.dataframe(pd.read_csv(table), use_container_width=True)

with tabs[1]:
    col_a, col_b = st.columns(2)
    col_a.image(ASSETS / "threshold_tradeoff.png", use_container_width=True)
    col_b.image(ASSETS / "robustness_performance.png", use_container_width=True)
    table = RESULTS / "identification_results.csv"
    if table.exists():
        st.dataframe(pd.read_csv(table), use_container_width=True)

with tabs[2]:
    st.image(ASSETS / "clustering_objective.png", use_container_width=True)
    table = RESULTS / "clustering_results.csv"
    if table.exists():
        st.dataframe(pd.read_csv(table), use_container_width=True)

with tabs[3]:
    st.image(ASSETS / "dir_curve.png", use_container_width=True)
    table = RESULTS / "dir_results.csv"
    if table.exists():
        st.dataframe(pd.read_csv(table), use_container_width=True)
