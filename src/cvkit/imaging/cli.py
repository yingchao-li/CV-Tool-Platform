"""Command-line access to the imaging stages."""

from __future__ import annotations

import argparse

from pathlib import Path

from ..paths import OUTPUT_IMAGING, ensure_dir
from .config import PATTERN_CR3, SAMPLE_JPG, SAMPLE_NPY, SAMPLE_RAW
from .pattern_detection import investigate_bayer_pattern_elegant
from .pipeline import (
    demosaic_raw,
    icam06_demo,
    plot_sensor_linearity,
    process_raw,
    run_hdr_merge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvkit imaging",
        description="Computational-photography stages: demosaicing, tone mapping, "
        "sensor calibration, response estimation and HDR merging.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demosaic = sub.add_parser("demosaic", help="Demosaic a single CR3 capture.")
    demosaic.add_argument("--raw", default=SAMPLE_RAW)
    demosaic.add_argument("--pattern", default=PATTERN_CR3)
    demosaic.add_argument("--out", default=str(OUTPUT_IMAGING / "demosaic_preview.jpg"))

    pattern = sub.add_parser(
        "detect-pattern", help="Estimate the Bayer phase from colour reference patches."
    )
    pattern.add_argument("--raw", default=SAMPLE_NPY)
    pattern.add_argument("--jpg", default=SAMPLE_JPG)

    linearity = sub.add_parser(
        "linearity", help="Plot mean sensor response against exposure time."
    )
    linearity.add_argument("--out", default=str(OUTPUT_IMAGING / "sensor_linearity.png"))

    hdr = sub.add_parser("hdr", help="Merge a RAW exposure bracket into an HDR image.")
    hdr.add_argument("--out", default=str(OUTPUT_IMAGING / "hdr_log_result.jpg"))

    icam = sub.add_parser("icam06", help="Tone-map an HDR bracket with iCAM06.")
    icam.add_argument("--out", default=str(OUTPUT_IMAGING / "icam06_result.jpg"))

    render = sub.add_parser("render", help="Run the full RAW-to-JPEG pipeline.")
    render.add_argument("--raw", default=SAMPLE_RAW)
    render.add_argument("--out", default=str(OUTPUT_IMAGING / "raw_pipeline.jpg"))
    render.add_argument("--pattern", default=PATTERN_CR3)

    response = sub.add_parser(
        "response", help="Estimate the JPEG inverse response (gamma) and merge HDR."
    )
    response.add_argument(
        "--method", choices=("gamma", "pairwise", "quantile"), default="gamma"
    )
    response.add_argument("--data-dir", default=None)
    response.add_argument("--output-dir", default=str(OUTPUT_IMAGING / "response"))
    response.add_argument("--samples", type=int, default=50000)
    response.add_argument("--estimate-only", action="store_true")

    return parser


def _prepare_output(path: str) -> None:
    ensure_dir(Path(path).parent)


def run(args: argparse.Namespace) -> dict | None:
    if args.command == "demosaic":
        _prepare_output(args.out)
        demosaic_raw(args.raw, args.out, pattern=args.pattern)
        return {"output": args.out}

    if args.command == "detect-pattern":
        return investigate_bayer_pattern_elegant(raw_path=args.raw, jpg_path=args.jpg)

    if args.command == "linearity":
        _prepare_output(args.out)
        plot_sensor_linearity(output_plot=args.out)
        return {"output": args.out}

    if args.command == "hdr":
        _prepare_output(args.out)
        run_hdr_merge(output_path=args.out)
        return {"output": args.out}

    if args.command == "icam06":
        _prepare_output(args.out)
        icam06_demo(output_path=args.out)
        return {"output": args.out}

    if args.command == "render":
        _prepare_output(args.out)
        process_raw(args.raw, args.out, pattern=args.pattern)
        return {"output": args.out}

    if args.command == "response":
        ensure_dir(args.output_dir)
        from .response_gamma import run_method as run_gamma
        from .response_pairwise import run_method as run_pairwise
        from .response_quantile import run_method as run_quantile

        runners = {"gamma": run_gamma, "pairwise": run_pairwise, "quantile": run_quantile}
        kwargs = {
            "output_dir": args.output_dir,
            "output_prefix": args.method,
            "samples": args.samples,
            "estimate_only": args.estimate_only,
        }
        if args.data_dir is not None:
            kwargs["data_dir"] = args.data_dir
        return runners[args.method](**kwargs)

    raise ValueError(f"Unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
