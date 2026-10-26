"""Unified command-line entry point for CVKit.

Examples
--------
``cvkit geometry --example 1 --method preemptive``
``cvkit imaging response --method gamma``
``cvkit retrieval experiment pca --help``
``cvkit detection run``
``cvkit faces train --mode ident --label Alan_Ball --video ...``
"""

from __future__ import annotations

import argparse
import importlib
import sys

from . import __version__

MODULES = {
    "geometry": "Point-cloud box measurement and robust plane fitting.",
    "imaging": "Demosaicing, sensor calibration, response estimation and HDR.",
    "retrieval": "RootSIFT + VLAD image retrieval and ablations.",
    "detection": "Selective Search + HOG/SVM object detection.",
    "faces": "Face detection, recognition, clustering and open-set rejection.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvkit",
        description="A modular computer vision toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"cvkit {__version__}")
    parser.add_argument("module", choices=MODULES, help="Subsystem to run.")
    return parser


def run(argv: list[str] | None = None) -> None:
    """Dispatch ``argv`` to ``cvkit.<module>.cli``.

    The first token is inspected manually so that ``--help`` and options after
    the module name reach the sub-command parser instead of being swallowed by
    the top-level one.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return
    if argv[0] == "--version":
        print(f"cvkit {__version__}")
        return

    module_name, rest = argv[0], argv[1:]
    if module_name not in MODULES:
        parser.error(f"unknown module {module_name!r}; choose from {', '.join(MODULES)}")

    module = importlib.import_module(f"cvkit.{module_name}.cli")
    module.main(rest)


def main(argv: list[str] | None = None) -> None:
    run(argv)


if __name__ == "__main__":
    main()
