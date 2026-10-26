"""Command-line access to the point-cloud measurement pipeline."""

from __future__ import annotations

import argparse
import json

from .dataset import list_mat_files
from .pipeline import measure_box, measure_example


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvkit geometry",
        description="Measure a box from a depth-camera point cloud.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--example", type=int, default=1, help="Sample capture id.")
    source.add_argument("--mat", help="Path to a .mat capture (overrides --example).")

    parser.add_argument("--method", choices=("ransac", "mlesac", "preemptive"), default="ransac")
    parser.add_argument("--floor-threshold", type=float, default=0.05)
    parser.add_argument("--box-threshold", type=float, default=0.005)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlesac-gamma", type=float, default=0.1)
    parser.add_argument("--hypotheses", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--json", help="Write the measurement summary to this file.")
    parser.add_argument("--list", action="store_true", help="List available captures.")
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.list:
        captures = list_mat_files()
        print("\n".join(captures))
        return {"captures": captures}

    kwargs = {
        "method": args.method,
        "floor_threshold": args.floor_threshold,
        "box_threshold": args.box_threshold,
        "max_iterations": args.iterations,
        "random_seed": args.seed,
        "mlesac_gamma": args.mlesac_gamma,
        "preemptive_hypotheses": args.hypotheses,
        "preemptive_batch": args.batch_size,
    }

    if args.mat:
        from .dataset import load_mat_file

        measurement = measure_box(load_mat_file(args.mat)[2], **kwargs)
    else:
        measurement = measure_example(args.example, **kwargs)

    summary = measurement.as_dict()
    print(json.dumps(summary, indent=2))

    if args.json:
        from pathlib import Path

        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
