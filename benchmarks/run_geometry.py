"""Compare RANSAC, MLESAC and Preemptive RANSAC on the sample captures.

Produces ``docs/results/geometry_comparison.csv`` with the same metrics
used in ``docs/benchmarks.md``: runtime, inlier counts, plane RMSE, the
estimated box dimensions and the estimator-specific score.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cvkit.common.io import write_csv
from cvkit.geometry.pipeline import measure_example
from cvkit.paths import DOCS_DIR

METHODS = ("ransac", "mlesac", "preemptive")

PARAMETERS = {
    "floor_threshold": 0.05,
    "box_threshold": 0.005,
    "max_iterations": 1000,
    "random_seed": 42,
    "mlesac_gamma": 0.1,
    "preemptive_hypotheses": 200,
    "preemptive_batch": 100,
    "min_floor_size": 100,
    "max_hole_size": 300,
    "size_method": "pca",
}


def main() -> None:
    rows = []
    for method in METHODS:
        started = time.perf_counter()
        measurement = measure_example(1, method=method, **PARAMETERS)
        runtime = time.perf_counter() - started

        rows.append(
            {
                "method": method,
                "runtime_s": round(runtime, 4),
                "floor_inliers": int(measurement.floor_inliers.sum()),
                "box_inliers": int(measurement.box_inliers.sum()),
                "floor_rmse": round(measurement.floor_rmse, 6),
                "box_rmse": round(measurement.box_rmse, 6),
                "height_m": round(measurement.height, 4),
                "length_m": round(measurement.length, 4),
                "width_m": round(measurement.width, 4),
                "floor_score": measurement.floor_score,
                "box_score": measurement.box_score,
            }
        )
        print(f"{method:<11} {runtime:6.3f}s  h={measurement.height:.4f}m")

    output = DOCS_DIR / "results" / "geometry_comparison.csv"
    write_csv(output, rows)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
