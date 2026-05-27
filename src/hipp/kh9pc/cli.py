import argparse
import logging
import sys
from pathlib import Path

from hipp.kh9pc.pipeline import preprocess_kh9pc


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stderr
    )
    logging.getLogger("hipp").setLevel(level)


def _cmd_preproc(args: argparse.Namespace) -> None:
    preprocess_kh9pc(
        input=args.input if len(args.input) > 1 else args.input[0],
        output_path=args.output,
        work_dir=args.work_dir,
        qc_dir=args.qc_dir,
        overwrite=args.overwrite,
    )


def _cmd_batch_preproc(_args: argparse.Namespace) -> None:
    # TODO: implement batch preprocessing
    raise NotImplementedError("batch_preproc is not yet implemented")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hipp-kh9pc", description="KH-9 Panoramic Camera preprocessing tools")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v INFO, -vv DEBUG)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- preproc ---
    p = subparsers.add_parser("preproc", help="Preprocess a single KH-9 PC scan")
    p.add_argument(
        "--input", "-i", nargs="+", required=True, metavar="FILE", help="Input archive (.tgz) or tile files (.tif)"
    )
    p.add_argument("--output", "-o", required=True, type=Path, metavar="FILE", help="Output restituted image (.tif)")
    p.add_argument(
        "--work-dir",
        "-w",
        type=Path,
        default=None,
        metavar="DIR",
        help="Working directory for intermediates (default: <output_parent>/_work)",
    )
    p.add_argument("--qc-dir", "-q", type=Path, default=None, metavar="DIR", help="Quality control output directory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.set_defaults(func=_cmd_preproc)

    # --- batch_preproc ---
    bp = subparsers.add_parser("batch_preproc", help="Batch preprocess multiple KH-9 PC scans")
    bp.set_defaults(func=_cmd_batch_preproc)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    args.func(args)
