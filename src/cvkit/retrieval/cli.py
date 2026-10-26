"""Command-line access to the retrieval pipeline and its ablations."""

from __future__ import annotations

import argparse
import sys

from .experiments import EXPERIMENTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvkit retrieval",
        description="RootSIFT + VLAD image retrieval and ablation experiments.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Run the baseline VLAD pipeline.")
    experiment = sub.add_parser("experiment", help="Run one ablation experiment.")
    experiment.add_argument("name", choices=EXPERIMENTS)
    return parser


def run(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return

    command, rest = argv[0], argv[1:]
    if command not in ("run", "experiment"):
        parser.error(f"unknown command {command!r}; choose from run, experiment")

    if command == "run":
        from . import pipeline

        pipeline.main(rest)
        return

    if not rest or rest[0] in ("-h", "--help"):
        parser.parse_args(["experiment", "--help"])
        return

    name, rest = rest[0], rest[1:]
    if name not in EXPERIMENTS:
        parser.error(f"unknown experiment {name!r}; choose from {', '.join(EXPERIMENTS)}")

    sys.argv = [f"cvkit retrieval experiment {name}", *rest]
    modules = {
        "intranorm": "intranorm",
        "pca": "pca_whitening",
        "aqe": "aqe",
        "codebook": "codebook_ablation",
        "inversion": "inversion",
    }
    module = __import__(
        f"cvkit.retrieval.experiments.{modules[name]}", fromlist=["main"]
    )
    module.main()


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
