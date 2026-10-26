"""
Streamlit page: tune the RAW-to-JPEG pipeline and its display parameters.

Run from the project root:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


import numpy as np
import streamlit as st
from PIL import Image

from cvkit.imaging.config import SAMPLE_RAW, PATTERN_CR3
from cvkit.imaging.pipeline import process_bayer_raw_to_rgb, process_raw
from cvkit.imaging.image_ops import load_raw_cr3

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def _load_bayer_raw_cached(path: str, mtime: float) -> np.ndarray:
    return load_raw_cr3(path)


def _rgb_float_to_pil(rgb: np.ndarray) -> Image.Image:
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(u8)


def main() -> None:
    st.set_page_config(page_title="RAW processing", layout="wide")
    st.title("Demosaic / HDR — RAW processing controls")
    st.caption(
        "Pipeline: `pipeline.process_raw` — black level → demosaic → gray-world WB → "
        "percentile stretch → gamma → stylistic boost → final brightness/contrast."
    )

    default_disk = PROJECT_ROOT / SAMPLE_RAW
    default_hint = str(default_disk) if default_disk.is_file() else str(PROJECT_ROOT / SAMPLE_RAW)

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload .CR3 (optional)", type=["cr3", "CR3"])
        rel_or_abs = st.text_input(
            "RAW path (relative to project root or absolute)",
            value=SAMPLE_RAW,
            help=f"Default sample: {SAMPLE_RAW}; resolves to: {default_hint}",
        )

        raw_path_resolved: str | None = None
        raw_array: np.ndarray | None = None

        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".CR3"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.getbuffer())
            tmp.close()
            raw_path_resolved = tmp.name
            st.session_state["_upload_tmp"] = raw_path_resolved
            try:
                raw_array = load_raw_cr3(raw_path_resolved)
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")
                raw_array = None
        else:
            p = Path(rel_or_abs)
            if not p.is_file():
                p = PROJECT_ROOT / rel_or_abs
            if p.is_file():
                raw_path_resolved = str(p.resolve())
                mtime = p.stat().st_mtime
                try:
                    raw_array = _load_bayer_raw_cached(raw_path_resolved, mtime)
                except Exception as e:
                    st.error(f"Could not read RAW: {e}")
                    raw_array = None
            else:
                st.warning(f"File not found: {rel_or_abs}")
                raw_array = None

        st.divider()
        patterns = ["RGGB", "GRBG", "GBRG", "BGGR"]
        st.header("Core parameters")
        pattern = st.selectbox(
            "Bayer pattern",
            options=patterns,
            index=patterns.index(PATTERN_CR3) if PATTERN_CR3 in patterns else 0,
        )
        black_pct = st.slider("Black level percentile (low)", 0.001, 1.0, 0.01, 0.001, format="%.3f")
        norm_low = st.slider("Stretch lower percentile", 0.001, 5.0, 0.01, 0.001, format="%.3f")
        norm_high = st.slider("Stretch upper percentile", 95.0, 99.999, 99.99, 0.001, format="%.3f")
        gamma = st.slider("Gamma (<1 brightens shadows)", 0.1, 1.0, 0.38, 0.01)

        st.subheader("Stylistic boost (in-pipeline)")
        enh_bright = st.slider("Boost: brightness offset", -0.2, 0.4, 0.10, 0.01)
        enh_contrast = st.slider("Boost: contrast", 0.5, 3.0, 1.75, 0.05)
        enh_sat = st.slider("Boost: saturation", 0.0, 2.0, 1.25, 0.05)

        st.subheader("Final output (display)")
        final_bright = st.slider("Final brightness offset", -0.35, 0.35, 0.0, 0.01)
        final_contrast = st.slider("Final contrast", 0.2, 3.0, 1.0, 0.05)

        st.divider()
        jpg_quality = st.slider("Export JPEG quality", 70, 100, 99, 1)
        out_name = st.text_input("Export filename", value="streamlit_export.jpg")

    col_preview, col_export = st.columns([2, 1])

    if raw_array is None:
        st.info("Upload a CR3 or enter a valid RAW path.")
        return

    with st.spinner("Processing…"):
        rgb = process_bayer_raw_to_rgb(
            raw_array,
            pattern=pattern,
            black_percentile=black_pct,
            norm_low=norm_low,
            norm_high=norm_high,
            gamma=gamma,
            enhance_brightness=enh_bright,
            enhance_contrast=enh_contrast,
            enhance_saturation=enh_sat,
            final_brightness=final_bright,
            final_contrast=final_contrast,
            verbose=False,
        )

    pil_img = _rgb_float_to_pil(rgb)

    with col_preview:
        st.subheader("Preview")
        st.image(pil_img, use_container_width=True)

    with col_export:
        st.subheader("Export")
        buf_path = PROJECT_ROOT / out_name
        if st.button("Save JPEG to project root", type="primary"):
            try:
                if not raw_path_resolved:
                    st.error("No valid RAW path for export.")
                else:
                    process_raw(
                        raw_path_resolved,
                        str(buf_path),
                        pattern=pattern,
                        black_percentile=black_pct,
                        norm_low=norm_low,
                        norm_high=norm_high,
                        gamma=gamma,
                        enhance_brightness=enh_bright,
                        enhance_contrast=enh_contrast,
                        enhance_saturation=enh_sat,
                        final_brightness=final_bright,
                        final_contrast=final_contrast,
                        jpg_quality=jpg_quality,
                        verbose=False,
                    )
                    st.success(f"Saved: {buf_path}")
            except Exception as e:
                st.error(str(e))

        st.download_button(
            label="Download JPEG (in-memory export)",
            data=_pil_to_bytes_jpg(pil_img, quality=jpg_quality),
            file_name=out_name,
            mime="image/jpeg",
        )

        st.caption(
            'After an upload, "Save to project root" reloads from the temp file on disk; '
            "result matches preview. You can also export with download only."
        )


def _pil_to_bytes_jpg(img: Image.Image, quality: int) -> bytes:
    import io

    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=int(quality))
    return bio.getvalue()


if __name__ == "__main__":
    main()
