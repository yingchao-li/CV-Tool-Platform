"""Command-line access to the detection pipeline.

Each subcommand forwards its remaining arguments to the module that owns
the functionality, so ``cvkit detection train --help`` shows the same
options as ``python -m cvkit.detection.trainer --help``.
"""

from __future__ import annotations

import argparse
import sys

COMMANDS = ("proposals", "train", "detect", "evaluate", "run", "tune")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvkit detection",
        description="Selective Search proposals, HOG/SVM training, inference, "
        "evaluation and parameter search.",
    )
    parser.add_argument("command", choices=COMMANDS)
    return parser


def run(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        parser.error(f"unknown command {command!r}; choose from {', '.join(COMMANDS)}")

    sys.argv = [f"cvkit detection {command}", *rest]

    if command == "proposals":
        from . import proposals

        proposals.main(rest)
    elif command == "train":
        from . import trainer

        trainer.main(rest)
    elif command == "detect":
        from . import detector

        detector.main(rest)
    elif command == "evaluate":
        from . import evaluation

        evaluation.main(rest)
    elif command == "run":
        from . import pipeline

        pipeline.main()
    elif command == "tune":
        from . import tuning

        tuning.main()


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
