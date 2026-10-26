"""Inverse camera-response model shared by the JPG estimators.

Defines the gamma response curve, sample weighting and the CSV/plot
output format.  It does not read images or merge exposures; the three
estimators in ``response_gamma``, ``response_pairwise`` and
``response_quantile`` all build on this module.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Inverse response model and sample weights
# ============================================================

def valid_weights(
    values: np.ndarray,
    min_valid: float = 0.02,
    max_valid: float = 0.95,
) -> np.ndarray:
    """Compute mid-tone weights; very dark or blown-out samples get weight 0."""
    if not 0.0 <= min_valid < max_valid <= 1.0:
        raise ValueError("Require 0 <= min_valid < max_valid <= 1")
    values = np.asarray(values, dtype=np.float32)
    weights = np.square(1.0 - np.abs(2.0 * values - 1.0)).astype(np.float32)
    valid = np.isfinite(values) & (values >= min_valid) & (values <= max_valid)
    weights[~valid] = 0.0
    return weights


def gamma_inverse_response(
    values: np.ndarray,
    gamma: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply h(y)=y**gamma to map JPG values into an approximately linear domain."""
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and greater than zero")
    clipped = np.clip(np.asarray(values, dtype=np.float32), eps, 1.0)
    return np.power(clipped, gamma).astype(np.float32)


def response_curve_arrays(
    gamma: float,
    points: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the inverse response curve ``h(y)=y**gamma`` at uniform intervals.

    ``values`` are normalized JPG pixel values y; the returned ``inverse`` is the
    recovered approximate linear luminosity x. The name is kept for compatibility
    with the Streamlit curve comparison.
    """
    values = np.linspace(0.0, 1.0, points, dtype=np.float32)
    inverse = np.power(values, gamma).astype(np.float32)
    return values, inverse


def camera_response_arrays(
    gamma: float,
    points: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the forward camera curve ``y=f(x)=x**(1/gamma)``.

    All three methods fit the gamma of the inverse response directly. Because
    that power function is monotonic and invertible, the forward camera
    luminosity transformation follows from the same gamma.
    """
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and greater than zero")
    linear_luminosity = np.linspace(0.0, 1.0, points, dtype=np.float32)
    jpg_value = np.power(linear_luminosity, 1.0 / gamma).astype(np.float32)
    return linear_luminosity, jpg_value


def response_equation_text(gamma: float) -> tuple[str, str]:
    """Return the forward and inverse curve equations for the summary display."""
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and greater than zero")
    forward_exponent = 1.0 / gamma
    forward = f"y = f(x) = x^(1/gamma) = x^{forward_exponent:.8f}"
    inverse = f"x = f^(-1)(y) = h(y) = y^gamma = y^{gamma:.8f}"
    return forward, inverse


# ============================================================
# Response curve output
# ============================================================

def save_response_curve_csv(
    path: str | Path,
    method: str,
    estimated_gamma: float,
    error_metric: float,
    points: int = 256,
) -> Path:
    """Write the forward camera curve and the inverse response to one CSV.

    Each row contains both:

    - linear luminosity x and its JPG output y=f(x)
    - JPG input y and its linearized value h(y)

    This answers the requested ``y=f(x)`` directly while keeping the inverse
    response that HDR processing actually uses.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    jpg_values, inverse = response_curve_arrays(estimated_gamma, points)
    linear_values, camera_values = camera_response_arrays(estimated_gamma, points)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "linear_luminosity_x",
                "camera_response_y",
                "normalized_jpg_value",
                "inverse_response_value",
                "method",
                "estimated_gamma",
                "error_metric",
            ]
        )
        for linear_x, camera_y, jpg_y, corrected_x in zip(
            linear_values,
            camera_values,
            jpg_values,
            inverse,
        ):
            writer.writerow(
                [
                    f"{linear_x:.8f}",
                    f"{camera_y:.8f}",
                    f"{jpg_y:.8f}",
                    f"{corrected_x:.8f}",
                    method,
                    f"{estimated_gamma:.10g}",
                    f"{error_metric:.10g}",
                ]
            )
    return output


def save_response_curve_plot(
    path: str | Path,
    method: str,
    estimated_gamma: float,
) -> Path:
    """Plot the forward camera response and the inverse response side by side."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    jpg_values, inverse = response_curve_arrays(estimated_gamma, 512)
    linear_values, camera_values = camera_response_arrays(estimated_gamma, 512)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: the requested y=f(x), where linear luminosity becomes a JPG value.
    axes[0].plot(
        linear_values,
        camera_values,
        linewidth=2.2,
        label=f"f(x) = x^(1/{estimated_gamma:.4f})",
    )
    axes[0].plot(
        linear_values,
        linear_values,
        "--",
        linewidth=1.5,
        label="Linear reference y = x",
    )
    axes[0].set(
        xlabel="Linear luminosity x",
        ylabel="Normalized JPG value y",
        title="Estimated camera transformation y = f(x)",
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Right: the inverse x=h(y) actually called during HDR preprocessing.
    axes[1].plot(
        jpg_values,
        inverse,
        linewidth=2.2,
        label=f"h(y) = y^{estimated_gamma:.4f}",
    )
    axes[1].plot(
        jpg_values,
        jpg_values,
        "--",
        linewidth=1.5,
        label="Linear reference x = y",
    )
    axes[1].set(
        xlabel="Normalized JPG value y",
        ylabel="Approximate linear luminosity x",
        title="Inverse response x = h(y)",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(f"Camera response estimation — {method}")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


# ============================================================
# Parameter checks shared by the three methods
# ============================================================

def validate_method_inputs(
    paths: Sequence[str | Path],
    exposure_times: Sequence[float],
    min_valid: float,
    max_valid: float,
    gamma_min: float,
    gamma_max: float,
    gamma_steps: int,
) -> None:
    """Check image count, exposure times, valid range and gamma search parameters."""
    if len(paths) < 2:
        raise ValueError("At least two JPG exposures are required")
    times = np.asarray(exposure_times, dtype=float)
    if (
        len(times) != len(paths)
        or np.any(times <= 0)
        or not np.all(np.isfinite(times))
    ):
        raise ValueError("Exposure times must be positive and match the JPG count")
    if not 0 <= min_valid < max_valid <= 1:
        raise ValueError("Require 0 <= min_valid < max_valid <= 1")
    if not 0 < gamma_min < gamma_max:
        raise ValueError("Require 0 < gamma_min < gamma_max")
    if gamma_steps < 3:
        raise ValueError("gamma_steps must be at least 3")
