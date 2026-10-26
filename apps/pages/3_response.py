"""Response estimation page: compare three JPG response estimators and their HDR output."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# Data layer.
from cvkit.imaging.response_io import (
    DEFAULT_DATA_DIR,
    DEFAULT_EXPOSURE_TEXT,
    collect_luminance_samples,
    format_exposure_times,
    image_shape,
    list_jpg_paths,
    parse_exposure_times,
    sample_shared_pixels,
)
# HDR merge and result file layer.
from cvkit.imaging.response_hdr import (
    HDR_DESCRIPTION,
    merge_hdr_from_jpg,
    output_paths,
    save_jpg_float,
    save_summary_txt,
    tone_map_log,
)
# Response curve layer.
from cvkit.imaging.response import (
    camera_response_arrays,
    response_curve_arrays,
    response_equation_text,
    save_response_curve_csv,
    save_response_curve_plot,
    validate_method_inputs,
)
from cvkit.imaging.response_gamma import (
    METHOD_DESCRIPTION as GAMMA_DESCRIPTION,
    estimate_gamma_exposure_consistency,
    save_loss_plot as save_gamma_diagnostic,
)
from cvkit.imaging.response_pairwise import (
    METHOD_DESCRIPTION as PAIRWISE_DESCRIPTION,
    estimate_gamma_pairwise,
    save_loss_plot as save_pairwise_diagnostic,
)
from cvkit.imaging.response_quantile import (
    METHOD_DESCRIPTION as QUANTILE_DESCRIPTION,
    estimate_gamma_quantile,
    save_quantile_alignment_plot,
)


from cvkit.paths import OUTPUT_IMAGING

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_OUTPUT_DIR = OUTPUT_IMAGING / "response"

METHODS = {
    "Gamma exposure consistency": {
        "prefix": "gamma",
        "description": GAMMA_DESCRIPTION,
        "estimator": estimate_gamma_exposure_consistency,
    },
    "Pairwise exposure ratio": {
        "prefix": "pairwise",
        "description": PAIRWISE_DESCRIPTION,
        "estimator": estimate_gamma_pairwise,
    },
    "Quantile matching": {
        "prefix": "quantile",
        "description": QUANTILE_DESCRIPTION,
        "estimator": estimate_gamma_quantile,
    },
}


def _resolve_data_dir(value: str) -> Path:
    """Resolve the page's data directory input to an absolute path."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _thumbnail(path: Path, size: tuple[int, int] = (420, 280)) -> Image.Image:
    """Create a small preview so the page does not hold full 32MP images."""
    with Image.open(path) as image:
        preview = image.convert("RGB")
        preview.thumbnail(size)
        return preview.copy()


def _save_method_outputs(
    method: str,
    result: dict[str, object],
    paths: list[Path],
    exposure_times: np.ndarray,
    min_valid: float,
    max_valid: float,
    requested_samples: int,
) -> dict[str, object]:
    """Save one method's curve, diagnostic, HDR and summary; return light state."""
    config = METHODS[method]
    outputs = output_paths(UI_OUTPUT_DIR, str(config["prefix"]))
    gamma = float(result["gamma"])
    error = float(result["error"])
    forward_equation, inverse_equation = response_equation_text(gamma)

    save_response_curve_csv(outputs["csv"], method, gamma, error)
    save_response_curve_plot(outputs["curve_plot"], method, gamma)
    if method == "Gamma exposure consistency":
        save_gamma_diagnostic(outputs["diagnostic"], result)
    elif method == "Pairwise exposure ratio":
        save_pairwise_diagnostic(outputs["diagnostic"], result)
    else:
        save_quantile_alignment_plot(outputs["diagnostic"], result, exposure_times)

    # The three methods differ only in the estimator; HDR merging is shared.
    hdr = merge_hdr_from_jpg(paths, exposure_times, gamma, min_valid, max_valid)
    display = tone_map_log(hdr, in_place=True)
    save_jpg_float(display, outputs["hdr"], quality=98)
    del hdr, display

    summary = {
        "method": method,
        "method description": config["description"],
        "HDR merge method": HDR_DESCRIPTION,
        "input files": ", ".join(path.name for path in paths),
        "exposure times [s]": format_exposure_times(exposure_times),
        "requested shared pixels": requested_samples,
        "valid sample count": result["valid_sample_count"],
        "estimated gamma": f"{gamma:.8f}",
        "estimated camera transformation": forward_equation,
        "inverse response used for linearization": inverse_equation,
        str(result["metric_name"]): f"{error:.8g}",
        "response curve CSV": outputs["csv"],
        "response curve plot": outputs["curve_plot"],
        "diagnostic plot": outputs["diagnostic"],
        "HDR result": outputs["hdr"],
        "compliance": "No Debevec / Robertson / specialized HDR calibration function was used.",
    }
    save_summary_txt(outputs["summary"], summary)
    return {
        "method": method,
        "gamma": gamma,
        "error": error,
        "metric_name": result["metric_name"],
        "forward_equation": forward_equation,
        "inverse_equation": inverse_equation,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "summary": summary,
    }


def _download(path: str, label: str, mime: str, key: str) -> None:
    file_path = Path(path)
    st.download_button(
        label,
        data=file_path.read_bytes(),
        file_name=file_path.name,
        mime=mime,
        key=key,
        width="stretch",
    )


def main() -> None:
    """Build the input area, per-method result area and cross-method comparison."""
    st.set_page_config(page_title="JPG Response Curve Estimation and HDR Merge", layout="wide")
    st.title("JPG Response Curve Estimation and HDR Merge")

    # The sidebar groups the inputs, valid luminance range and gamma search range.
    with st.sidebar:
        st.header("Inputs")
        data_dir_text = st.text_input("Data directory", value=DEFAULT_DATA_DIR)
        selected_methods = st.multiselect(
            "Methods",
            options=list(METHODS),
            default=list(METHODS),
        )
        exposure_text = st.text_input("Exposure times [s]", value=DEFAULT_EXPOSURE_TEXT)
        samples = st.number_input("Samples", min_value=1000, max_value=500000, value=50000, step=1000)
        seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=7, step=1)
        min_valid = st.number_input("min_valid", min_value=0.0, max_value=0.49, value=0.02, step=0.01)
        max_valid = st.number_input("max_valid", min_value=0.51, max_value=1.0, value=0.95, step=0.01)
        st.header("Gamma search")
        gamma_min = st.number_input("gamma_min", min_value=0.05, max_value=5.0, value=0.4, step=0.05)
        gamma_max = st.number_input("gamma_max", min_value=0.1, max_value=8.0, value=3.5, step=0.1)
        gamma_steps = st.number_input("gamma_steps", min_value=3, max_value=2000, value=250, step=10)
        run_clicked = st.button("Run selected methods", type="primary", width="stretch")

    data_dir = _resolve_data_dir(data_dir_text)
    try:
        paths = list_jpg_paths(data_dir)
        exposure_times = parse_exposure_times(exposure_text, paths)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    st.subheader("Input JPG files")
    st.dataframe(
        [
            {"file": path.name, "exposure_seconds": float(exposure)}
            for path, exposure in zip(paths, exposure_times)
        ],
        width="stretch",
        hide_index=True,
    )
    thumbnail_columns = st.columns(3)
    for index, (path, exposure) in enumerate(zip(paths, exposure_times)):
        with thumbnail_columns[index % 3]:
            st.image(_thumbnail(path), caption=f"{path.name} — {exposure:.6g} s", width="stretch")

    # Clicking the button samples once; all estimators share the same luminance samples.
    if run_clicked:
        if not selected_methods:
            st.warning("Select at least one method.")
        else:
            try:
                validate_method_inputs(
                    paths,
                    exposure_times,
                    float(min_valid),
                    float(max_valid),
                    float(gamma_min),
                    float(gamma_max),
                    int(gamma_steps),
                )
                height, width = image_shape(paths)
                rows, cols = sample_shared_pixels(height, width, int(samples), int(seed))
                with st.spinner("Sampling shared JPG pixels..."):
                    luminance_samples = collect_luminance_samples(paths, rows, cols)

                saved_results = []
                progress = st.progress(0.0)
                for index, method in enumerate(selected_methods):
                    with st.spinner(f"Estimating response and merging HDR: {method}"):
                        estimator = METHODS[method]["estimator"]
                        estimated = estimator(
                            luminance_samples,
                            exposure_times,
                            float(min_valid),
                            float(max_valid),
                            float(gamma_min),
                            float(gamma_max),
                            int(gamma_steps),
                        )
                        saved_results.append(
                            _save_method_outputs(
                                method,
                                estimated,
                                paths,
                                exposure_times,
                                float(min_valid),
                                float(max_valid),
                                len(rows),
                            )
                        )
                    progress.progress((index + 1) / len(selected_methods))
                st.session_state["response_results"] = saved_results
                st.success("Selected methods completed.")
            except Exception as exc:
                st.exception(exc)

    results = st.session_state.get("response_results", [])
    if not results:
        return

    # Show metrics, forward/inverse curves, diagnostic, HDR and downloads per method.
    st.subheader("Method results")
    for result in results:
        with st.expander(result["method"], expanded=True):
            metric_col, error_col = st.columns(2)
            metric_col.metric("Estimated gamma", f"{result['gamma']:.6f}")
            error_col.metric(str(result["metric_name"]), f"{result['error']:.6g}")
            st.code(
                f"Camera transformation: {result['forward_equation']}\n"
                f"Inverse linearization: {result['inverse_equation']}",
                language="text",
            )
            curve_col, diagnostic_col, hdr_col = st.columns(3)
            curve_col.image(
                result["outputs"]["curve_plot"],
                caption="Camera transformation and inverse response",
            )
            diagnostic_col.image(result["outputs"]["diagnostic"], caption="Diagnostic")
            hdr_col.image(result["outputs"]["hdr"], caption="Tone-mapped JPG HDR result")

            download_columns = st.columns(5)
            specs = [
                ("csv", "Curve CSV", "text/csv"),
                ("curve_plot", "Curve PNG", "image/png"),
                ("diagnostic", "Diagnostic PNG", "image/png"),
                ("hdr", "HDR JPG", "image/jpeg"),
                ("summary", "Summary TXT", "text/plain"),
            ]
            for column, (output_key, label, mime) in zip(download_columns, specs):
                with column:
                    _download(
                        result["outputs"][output_key],
                        label,
                        mime,
                        f"{result['method']}-{output_key}",
                    )

    st.subheader("Comparison")
    st.dataframe(
        [
            {
                "method": result["method"],
                "gamma": result["gamma"],
                "error metric": result["metric_name"],
                "error": result["error"],
                "output HDR path": result["outputs"]["hdr"],
            }
            for result in results
        ],
        width="stretch",
        hide_index=True,
    )

    # Compare the required forward f(x) with the inverse response used by HDR merging.
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for result in results:
        gamma = float(result["gamma"])
        linear_x, camera_y = camera_response_arrays(gamma, 512)
        jpg_y, inverse_x = response_curve_arrays(gamma, 512)
        label = f"{result['method']} ({gamma:.4f})"
        axes[0].plot(linear_x, camera_y, linewidth=2, label=label)
        axes[1].plot(jpg_y, inverse_x, linewidth=2, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Linear reference")
    axes[0].set(xlabel="Linear luminosity x", ylabel="JPG value y")
    axes[0].set_title("Camera transformation y = f(x)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Linear reference")
    axes[1].set(xlabel="JPG value y", ylabel="Linear luminosity x")
    axes[1].set_title("Inverse response x = h(y)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    hdr_columns = st.columns(len(results))
    for column, result in zip(hdr_columns, results):
        column.image(
            result["outputs"]["hdr"],
            caption=f"{result['method']} — gamma {result['gamma']:.4f}",
            width="stretch",
        )


if __name__ == "__main__":
    main()
