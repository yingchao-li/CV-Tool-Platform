"""CVKit Studio - a Streamlit front end for every subsystem.

Run from the repository root::

    streamlit run apps/studio.py

Each page in ``apps/pages`` is a thin view over the corresponding
``cvkit`` module, so the UI and the CLI always share one implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cvkit import __version__
from cvkit.paths import BALLOON_DIR, FACE_DIR, PAINTING_DIR, POINTCLOUD_DIR, RAW_DIR

st.set_page_config(page_title="CVKit Studio", page_icon="📷", layout="wide")

st.title("CVKit Studio")
st.caption(
    f"v{__version__} · classical computer vision toolkit · "
    "point clouds, RAW imaging, retrieval, detection and faces"
)

st.markdown(
    """
    Five independent pipelines share one data layout, one command-line
    interface and one interactive front end. Pick a page in the sidebar, or
    run any stage from the terminal with `python -m cvkit <module> ...`.
    """
)

left, right = st.columns([3, 2])

with left:
    st.subheader("Modules")
    st.markdown(
        """
| Module | What it does | Signature method |
|---|---|---|
| **Geometry** | Measures a box from a depth-camera point cloud | RANSAC / MLESAC / Preemptive RANSAC |
| **Imaging** | Develops RAW captures and merges exposure brackets | Bayer demosaicing, sensor linearity, gamma response, iCAM06 |
| **Retrieval** | Finds instances of the same writer or scene | RootSIFT + VLAD + exemplar SVM |
| **Detection** | Locates balloons without a deep network | Selective Search + HOG + linear SVM |
| **Faces** | Detects, tracks, clusters and rejects unknown faces | MTCNN + FaceNet + SPL/MPL |
"""
    )

with right:
    st.subheader("Data availability")
    datasets = {
        "point clouds": POINTCLOUD_DIR,
        "RAW brackets": RAW_DIR,
        "balloon set": BALLOON_DIR,
        "paintings": PAINTING_DIR,
        "face sequences": FACE_DIR,
    }
    for name, path in datasets.items():
        status = "available" if path.is_dir() else "missing"
        st.write(f"{'✅' if path.is_dir() else '⚠️'} **{name}** — {status}")

st.divider()

st.subheader("Quick start")
st.code(
    """# install
pip install -e .

# point-cloud measurement
python -m cvkit geometry --example 1 --method preemptive

# RAW demosaicing and HDR
python -m cvkit imaging demosaic
python -m cvkit imaging hdr

# retrieval ablation
python -m cvkit retrieval experiment pca --help

# object detection
python -m cvkit detection run

# face benchmarks
python benchmarks/run_faces.py""",
    language="bash",
)

st.info(
    "Large datasets (RAW captures and face frames) are not committed to git. "
    "See `docs/data.md` for the expected layout."
)
