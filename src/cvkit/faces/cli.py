"""Command-line access to the face-analysis tools."""

from __future__ import annotations

import argparse
import sys

COMMANDS = ("train", "infer", "camera", "dir", "openset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvkit faces",
        description="Face detection, recognition, clustering and open-set rejection.",
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

    sys.argv = [f"cvkit faces {command}", *rest]

    if command == "train":
        from . import train

        train.main(train.arguments())
    elif command == "infer":
        from . import infer

        infer.main(infer.arguments())
    elif command == "camera":
        from . import camera

        camera.main()
    elif command == "dir":
        from . import dir_eval

        dir_eval.main()
    elif command == "openset":
        from . import openset

        openset.main()


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
