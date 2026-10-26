"""Response estimation from pairwise exposure ratios.

Chooses the gamma that makes adjacent frames agree with the known ratio
of their exposure times.
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
# Response layer: weights, curve output and parameter checks.
from .response import (
    response_equation_text,
    save_response_curve_csv,
    save_response_curve_plot,
    valid_weights,
    validate_method_inputs,
)


METHOD_NAME = "Pairwise exposure ratio"
METHOD_DESCRIPTION = (
    "Custom adjacent-exposure ratio fitting with h(y)=y^gamma. It compares only "
    "paired JPG measurements, derives f(x)=x^(1/gamma), and does not estimate any "
    "per-pixel irradiance variable."
)


def _pairwise_loss(log_differences: np.ndarray, targets: np.ndarray, weights: np.ndarray, gamma: float) -> float:
    """Weighted log-ratio RMSE of all adjacent exposure pairs for a candidate gamma."""
    # gamma only scales the observed JPG values; the target comes from the known exposure ratio.
    residual = gamma * log_differences - targets
    denominator = weights.sum()
    if denominator <= 0:
        return float("inf")
    return float(np.sqrt(np.sum(weights * residual * residual) / denominator))


def estimate_gamma_pairwise(
    luminance_samples: np.ndarray,
    exposure_times: Sequence[float],
    min_valid: float = 0.02,
    max_valid: float = 0.95,
    gamma_min: float = 0.4,
    gamma_max: float = 3.5,
    gamma_steps: int = 250,
) -> dict[str, object]:
    """Fit gamma from adjacent-exposure ratio constraints without per-pixel irradiance unknowns."""
    values = np.asarray(luminance_samples, dtype=np.float64)
    times = np.asarray(exposure_times, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(times):
        raise ValueError("luminance_samples must have shape (exposures, samples)")

    all_weights = valid_weights(values, min_valid, max_valid).astype(np.float64)
    log_differences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    pair_weights: list[np.ndarray] = []
    pair_counts: list[int] = []

    # Build only (0,1), (1,2), ... adjacent pairs and drop positions dark or blown in either image.
    for first in range(len(times) - 1):
        second = first + 1
        valid = (all_weights[first] > 0) & (all_weights[second] > 0)
        count = int(np.count_nonzero(valid))
        pair_counts.append(count)
        if count == 0:
            continue
        log_differences.append(
            np.log(np.clip(values[first, valid], 1e-8, 1.0))
            - np.log(np.clip(values[second, valid], 1e-8, 1.0))
        )
        targets.append(np.full(count, np.log(times[first] / times[second]), dtype=np.float64))
        pair_weights.append(np.sqrt(all_weights[first, valid] * all_weights[second, valid]))

    if not log_differences:
        raise ValueError("No sampled pixel is valid in an adjacent exposure pair")
    differences = np.concatenate(log_differences)
    target_values = np.concatenate(targets)
    weights = np.concatenate(pair_weights)

    # Full coarse search first, then refine around the best value.
    coarse_gammas = np.linspace(gamma_min, gamma_max, gamma_steps)
    coarse_losses = np.asarray(
        [_pairwise_loss(differences, target_values, weights, gamma) for gamma in coarse_gammas]
    )
    coarse_best = int(np.argmin(coarse_losses))
    coarse_step = float(coarse_gammas[1] - coarse_gammas[0])
    fine_min = max(gamma_min, float(coarse_gammas[coarse_best] - 2 * coarse_step))
    fine_max = min(gamma_max, float(coarse_gammas[coarse_best] + 2 * coarse_step))
    fine_gammas = np.linspace(fine_min, fine_max, max(101, gamma_steps))
    fine_losses = np.asarray(
        [_pairwise_loss(differences, target_values, weights, gamma) for gamma in fine_gammas]
    )
    fine_best = int(np.argmin(fine_losses))

    return {
        "gamma": float(fine_gammas[fine_best]),
        "error": float(fine_losses[fine_best]),
        "metric_name": "pairwise log-ratio RMSE",
        "valid_sample_count": int(differences.size),
        "pair_counts": pair_counts,
        "coarse_gammas": coarse_gammas,
        "coarse_losses": coarse_losses,
        "fine_gammas": fine_gammas,
        "fine_losses": fine_losses,
    }


def save_loss_plot(path: str | Path, result: dict[str, object]) -> Path:
    output = Path(path)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(result["coarse_gammas"], result["coarse_losses"], label="Coarse grid")
    axis.plot(result["fine_gammas"], result["fine_losses"], label="Fine grid")
    axis.scatter([result["gamma"]], [result["error"]], color="red", zorder=3, label="Best")
    axis.set(xlabel="Gamma", ylabel="Pairwise log-ratio RMSE")
    axis.set_title("Adjacent exposure-ratio loss versus gamma")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def run_method(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = str(OUTPUT_IMAGING),
    output_prefix: str = "pairwise",
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
    """Run pairwise fitting and optionally merge the bracket into HDR."""
    paths = list_jpg_paths(data_dir)
    times = parse_exposure_times(exposure_times, paths)
    validate_method_inputs(paths, times, min_valid, max_valid, gamma_min, gamma_max, gamma_steps)
    print_input_report(paths, times, samples)

    height, width = image_shape(paths)
    rows, cols = sample_shared_pixels(height, width, samples, seed)
    luminance_samples = collect_luminance_samples(paths, rows, cols)
    result = estimate_gamma_pairwise(
        luminance_samples, times, min_valid, max_valid, gamma_min, gamma_max, gamma_steps
    )
    outputs = output_paths(output_dir, output_prefix)
    gamma = float(result["gamma"])
    error = float(result["error"])
    forward_equation, inverse_equation = response_equation_text(gamma)

    # The CSV/curve plot shows both the forward f(x) and the inverse h(y).
    save_response_curve_csv(outputs["csv"], METHOD_NAME, gamma, error)
    save_response_curve_plot(outputs["curve_plot"], METHOD_NAME, gamma)
    save_loss_plot(outputs["diagnostic"], result)

    if not estimate_only:
        # Fitting does not merge HDR; the shared group HDR code runs only after the curve is done.
        print("Merging JPG exposures with the threshold-replacement HDR merge...")
        hdr = merge_hdr_from_jpg(paths, times, gamma, min_valid, max_valid)
        display = tone_map_log(hdr, in_place=True)
        save_jpg_float(display, outputs["hdr"], quality=98)

    pair_labels = [f"{i}-{i + 1}:{count}" for i, count in enumerate(result["pair_counts"])]
    summary_fields = {
        "method": METHOD_NAME,
        "method description": METHOD_DESCRIPTION,
        "HDR merge method": HDR_DESCRIPTION,
        "input files": ", ".join(path.name for path in paths),
        "exposure times [s]": format_exposure_times(times),
        "requested shared pixels": len(rows),
        "valid pair observations": result["valid_sample_count"],
        "valid counts by adjacent pair": ", ".join(pair_labels),
        "estimated gamma": f"{gamma:.8f}",
        "estimated camera transformation": forward_equation,
        "inverse response used for linearization": inverse_equation,
        str(result["metric_name"]): f"{error:.8g}",
        "response curve CSV": outputs["csv"],
        "response curve plot": outputs["curve_plot"],
        "diagnostic plot": outputs["diagnostic"],
        "HDR result": "not generated (--estimate-only)" if estimate_only else outputs["hdr"],
    }
    save_summary_txt(outputs["summary"], summary_fields)
    result.update({"paths": paths, "exposure_times": times, "outputs": outputs, "summary": summary_fields})

    print(f"Valid pair observations: {result['valid_sample_count']}")
    print(f"Estimated gamma: {gamma:.8f}")
    print(f"Pairwise log-ratio RMSE: {error:.8g}")
    print("Saved outputs:")
    for key, path in outputs.items():
        if key != "hdr" or not estimate_only:
            print(f"  {path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=METHOD_DESCRIPTION)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=str(OUTPUT_IMAGING))
    parser.add_argument("--output-prefix", default="pairwise")
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
