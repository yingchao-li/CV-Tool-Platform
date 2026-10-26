"""Response estimation from luminance quantile matching.

Chooses the gamma that best aligns the luminance distributions of all
exposures after exposure normalisation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data layer: files, exposure times and shared pixel samples.
from .response_io import (
    DEFAULT_DATA_DIR,
    DEFAULT_EXPOSURE_TEXT,
    collect_luminance_samples,
    format_exposure_times,
    image_shape,
    list_jpg_paths,
    parse_exposure_times,
    print_input_report,
    sample_shared_pixels,
)
# HDR layer: reuse the shared merge and persist the outputs.
from ..paths import OUTPUT_IMAGING
from .response_hdr import (
    HDR_DESCRIPTION,
    merge_hdr_from_jpg,
    output_paths,
    save_jpg_float,
    save_summary_txt,
    tone_map_log,
)
# Response layer: curve output and shared parameter checks.
from .response import (
    response_equation_text,
    save_response_curve_csv,
    save_response_curve_plot,
    validate_method_inputs,
)


METHOD_NAME = "Quantile matching"
METHOD_DESCRIPTION = (
    "Custom distribution-consistency fitting with h(y)=y^gamma. It aligns luminance "
    "quantiles after exposure normalization, derives f(x)=x^(1/gamma), and does not "
    "estimate per-pixel irradiance."
)
DEFAULT_QUANTILES = np.arange(0.1, 1.0, 0.1, dtype=np.float64)


def _quantile_matrix(
    values: np.ndarray,
    exposure_times: np.ndarray,
    valid_masks: np.ndarray,
    gamma: float,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Compute luminance quantiles per exposure after linearization and exposure normalization."""
    rows = []
    for index, exposure in enumerate(exposure_times):
        corrected = np.power(np.clip(values[index, valid_masks[index]], 1e-8, 1.0), gamma)
        normalized = corrected / exposure
        rows.append(np.quantile(normalized, quantiles))
    return np.asarray(rows, dtype=np.float64)


def _quantile_loss(matrix: np.ndarray) -> float:
    # Compare multiplicative differences in log space so overall scale does not drive the choice.
    logged = np.log(np.maximum(matrix, 1e-12))
    center = logged.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.square(logged - center))))


def estimate_gamma_quantile(
    luminance_samples: np.ndarray,
    exposure_times: Sequence[float],
    min_valid: float = 0.02,
    max_valid: float = 0.95,
    gamma_min: float = 0.4,
    gamma_max: float = 3.5,
    gamma_steps: int = 250,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, object]:
    """Estimate gamma by aligning the normalized luminance distributions of the exposures."""
    values = np.asarray(luminance_samples, dtype=np.float64)
    times = np.asarray(exposure_times, dtype=np.float64)
    quantile_values = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(times):
        raise ValueError("luminance_samples must have shape (exposures, samples)")
    if np.any(quantile_values <= 0) or np.any(quantile_values >= 1):
        raise ValueError("Quantiles must lie strictly between 0 and 1")

    # Reject dark, blown-out and non-finite samples independently per image.
    valid_masks = np.isfinite(values) & (values >= min_valid) & (values <= max_valid)
    valid_counts = np.count_nonzero(valid_masks, axis=1)
    if np.any(valid_counts < len(quantile_values)):
        raise ValueError("Not enough valid sampled pixels in every exposure for quantile fitting")

    def loss(gamma: float) -> float:
        return _quantile_loss(
            _quantile_matrix(values, times, valid_masks, gamma, quantile_values)
        )

    # The quantile method also uses a coarse search plus a local fine search.
    coarse_gammas = np.linspace(gamma_min, gamma_max, gamma_steps)
    coarse_losses = np.asarray([loss(gamma) for gamma in coarse_gammas])
    coarse_best = int(np.argmin(coarse_losses))
    coarse_step = float(coarse_gammas[1] - coarse_gammas[0])
    fine_min = max(gamma_min, float(coarse_gammas[coarse_best] - 2 * coarse_step))
    fine_max = min(gamma_max, float(coarse_gammas[coarse_best] + 2 * coarse_step))
    fine_gammas = np.linspace(fine_min, fine_max, max(101, gamma_steps))
    fine_losses = np.asarray([loss(gamma) for gamma in fine_gammas])
    fine_best = int(np.argmin(fine_losses))
    best_gamma = float(fine_gammas[fine_best])
    best_matrix = _quantile_matrix(values, times, valid_masks, best_gamma, quantile_values)

    return {
        "gamma": best_gamma,
        "error": float(fine_losses[fine_best]),
        "metric_name": "quantile log-domain mismatch RMSE",
        "valid_sample_count": int(valid_counts.sum()),
        "valid_counts": valid_counts,
        "quantiles": quantile_values,
        "quantile_matrix": best_matrix,
        "coarse_gammas": coarse_gammas,
        "coarse_losses": coarse_losses,
        "fine_gammas": fine_gammas,
        "fine_losses": fine_losses,
    }


def save_quantile_alignment_plot(
    path: str | Path,
    result: dict[str, object],
    exposure_times: Sequence[float],
) -> Path:
    output = Path(path)
    quantiles = np.asarray(result["quantiles"]) * 100.0
    matrix = np.asarray(result["quantile_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for index, (row, exposure) in enumerate(zip(matrix, exposure_times)):
        axes[0].plot(quantiles, row, marker="o", label=f"image {index}: {exposure:.6g} s")
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="Luminance quantile [%]",
        ylabel="Linearized luminance / exposure",
        title="Quantile alignment at fitted gamma",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(result["coarse_gammas"], result["coarse_losses"], label="Coarse grid")
    axes[1].plot(result["fine_gammas"], result["fine_losses"], label="Fine grid")
    axes[1].scatter([result["gamma"]], [result["error"]], color="red", zorder=3, label="Best")
    axes[1].set(
        xlabel="Gamma",
        ylabel="Log-quantile mismatch RMSE",
        title="Quantile mismatch versus gamma",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def run_method(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = str(OUTPUT_IMAGING),
    output_prefix: str = "quantile",
    samples: int = 50000,
    seed: int = 7,
    exposure_times: str | Sequence[float] | None = DEFAULT_EXPOSURE_TEXT,
    min_valid: float = 0.02,
    max_valid: float = 0.95,
    gamma_min: float = 0.4,
    gamma_max: float = 3.5,
    gamma_steps: int = 250,
    estimate_only: bool = False,
) -> dict[str, object]:
    """Run quantile fitting and optionally merge the bracket into HDR."""
    paths = list_jpg_paths(data_dir)
    times = parse_exposure_times(exposure_times, paths)
    validate_method_inputs(paths, times, min_valid, max_valid, gamma_min, gamma_max, gamma_steps)
    print_input_report(paths, times, samples)

    height, width = image_shape(paths)
    rows, cols = sample_shared_pixels(height, width, samples, seed)
    luminance_samples = collect_luminance_samples(paths, rows, cols)
    result = estimate_gamma_quantile(
        luminance_samples, times, min_valid, max_valid, gamma_min, gamma_max, gamma_steps
    )
    outputs = output_paths(output_dir, output_prefix)
    gamma = float(result["gamma"])
    error = float(result["error"])
    forward_equation, inverse_equation = response_equation_text(gamma)

    # Output the forward camera curve, the inverse curve and the quantile diagnostic plot.
    save_response_curve_csv(outputs["csv"], METHOD_NAME, gamma, error)
    save_response_curve_plot(outputs["curve_plot"], METHOD_NAME, gamma)
    save_quantile_alignment_plot(outputs["diagnostic"], result, times)

    if not estimate_only:
        # Response estimation and HDR merge are separate layers; group HDR starts here.
        print("Merging JPG exposures with the threshold-replacement HDR merge...")
        hdr = merge_hdr_from_jpg(paths, times, gamma, min_valid, max_valid)
        display = tone_map_log(hdr, in_place=True)
        save_jpg_float(display, outputs["hdr"], quality=98)

    summary_fields = {
        "method": METHOD_NAME,
        "method description": METHOD_DESCRIPTION,
        "HDR merge method": HDR_DESCRIPTION,
        "input files": ", ".join(path.name for path in paths),
        "exposure times [s]": format_exposure_times(times),
        "requested shared pixels": len(rows),
        "valid sample observations": result["valid_sample_count"],
        "valid counts by image": ", ".join(str(int(v)) for v in result["valid_counts"]),
        "quantiles": ", ".join(f"{100 * q:.0f}%" for q in result["quantiles"]),
        "estimated gamma": f"{gamma:.8f}",
        "estimated camera transformation": forward_equation,
        "inverse response used for linearization": inverse_equation,
        str(result["metric_name"]): f"{error:.8g}",
        "response curve CSV": outputs["csv"],
        "response curve plot": outputs["curve_plot"],
        "diagnostic plot": outputs["diagnostic"],
        "HDR result": "not generated (--estimate-only)" if estimate_only else outputs["hdr"],
        "compliance": (
            "No Debevec / Robertson / specialized HDR calibration function was used."
        ),
    }
    save_summary_txt(outputs["summary"], summary_fields)
    result.update({"paths": paths, "exposure_times": times, "outputs": outputs, "summary": summary_fields})

    print(f"Valid sample observations: {result['valid_sample_count']}")
    print(f"Estimated gamma: {gamma:.8f}")
    print(f"Quantile mismatch score: {error:.8g}")
    print("Saved outputs:")
    for key, path in outputs.items():
        if key != "hdr" or not estimate_only:
            print(f"  {path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=METHOD_DESCRIPTION)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=str(OUTPUT_IMAGING))
    parser.add_argument("--output-prefix", default="quantile")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exposure-times", default=DEFAULT_EXPOSURE_TEXT)
    parser.add_argument("--min-valid", type=float, default=0.02)
    parser.add_argument("--max-valid", type=float, default=0.95)
    parser.add_argument("--gamma-min", type=float, default=0.4)
    parser.add_argument("--gamma-max", type=float, default=3.5)
    parser.add_argument("--gamma-steps", type=int, default=250)
    parser.add_argument("--estimate-only", action="store_true")
    return parser


def main() -> None:
    run_method(**vars(build_parser().parse_args()))


if __name__ == "__main__":
    main()
