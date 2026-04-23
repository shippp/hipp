"""CLI entry point for the KH-9 PC pipeline.

Usage
-----
    python -m hipp.kh9pc --input scan.tgz --output /out/images/DZB1215.tif --qc-dir /out/qc
    python -m hipp.kh9pc --input t1.tif t2.tif t3.tif --output /out/DZB1215.tif --qc-dir /out/qc
    python -m hipp.kh9pc --input scan.tgz --output /out/DZB1215.tif --qc-dir /out/qc --config cfg.toml
"""

import argparse
import logging
from pathlib import Path

from hipp.kh9pc.pipeline import KH9Pipeline, PipelineConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hipp.kh9pc",
        description="KH-9 Panoramic Camera end-to-end preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        nargs="+",
        metavar="PATH",
        help="path to a .tgz archive, or an ordered list of .tif tiles",
    )
    p.add_argument("--output", required=True, metavar="FILE", help="path for the final rectified .tif")
    p.add_argument("--qc-dir", required=True, metavar="DIR", help="root directory for QC outputs")
    p.add_argument("--work-dir", metavar="DIR", default=None, help="directory for intermediate files (default: <output parent>/_work)")
    p.add_argument("--config", metavar="TOML", default=None, help="TOML config file; CLI flags override it")
    p.add_argument("--overwrite", action="store_true", default=False, help="re-run steps even when outputs already exist")
    p.add_argument("--steps", nargs="+", metavar="STEP", default=None, help="subset of steps to run (default: all)")
    p.add_argument("--cleanup", action="store_true", default=False, help="delete the work directory after completion")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)

    # Only hipp logs at the requested level; silence noisy third-party loggers.
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.WARNING)
    logging.getLogger("hipp").setLevel(getattr(logging, args.log_level))

    config = PipelineConfig.from_toml(Path(args.config)) if args.config else PipelineConfig()

    # CLI flags override YAML values when explicitly provided
    if args.overwrite:
        config.overwrite = True
    if args.steps:
        config.steps = args.steps
    if args.cleanup:
        config.cleanup = True

    inputs = [Path(p) for p in args.input]
    pipeline_input: Path | list[Path] = inputs[0] if len(inputs) == 1 else inputs

    pipeline = KH9Pipeline(
        input=pipeline_input,
        output=Path(args.output),
        qc_dir=Path(args.qc_dir),
        work_dir=Path(args.work_dir) if args.work_dir else None,
        config=config,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
