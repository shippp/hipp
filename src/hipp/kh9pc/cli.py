# mypy: disable-error-code="misc"
import logging
import sys
from pathlib import Path

import click

from hipp.kh9pc.pipeline import batch_preprocess_kh9pc, preprocess_kh9pc


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stderr
    )
    logging.getLogger("hipp").setLevel(level)


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v INFO, -vv DEBUG)")
@click.pass_context
def main(ctx: click.Context, verbose: int) -> None:
    """KH-9 Panoramic Camera preprocessing tools."""
    ctx.ensure_object(dict)
    _configure_logging(verbose)


@main.command()
@click.option(
    "--input",
    "-i",
    "input_files",
    multiple=True,
    required=True,
    metavar="FILE",
    help="Input archive (.tgz) or tile files (.tif)",
)
@click.option(
    "--output-dir", "-o", required=True, type=Path, metavar="DIR", help="Output directory for restituted images"
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.option("--keep-work", is_flag=True, help="Keep intermediate working files")
def preproc(input_files: tuple[str, ...], output_dir: Path, overwrite: bool, keep_work: bool) -> None:
    """Preprocess a single KH-9 PC scan."""
    preprocess_kh9pc(
        input=list(input_files) if len(input_files) > 1 else input_files[0],
        output_dir=output_dir,
        overwrite=overwrite,
        keep_work=keep_work,
    )


@main.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=Path,
    metavar="DIR",
    help="Directory containing input archives or tile subdirectories",
)
@click.option(
    "--output-dir", "-o", required=True, type=Path, metavar="DIR", help="Output directory for restituted images"
)
@click.option("--n-jobs", "-j", default=1, show_default=True, help="Number of parallel jobs")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.option("--keep-work", is_flag=True, help="Keep intermediate working files")
@click.option("--dry-run", is_flag=True, help="Log what would be processed without running")
def batch_preproc(
    input_dir: Path, output_dir: Path, n_jobs: int, overwrite: bool, keep_work: bool, dry_run: bool
) -> None:
    """Batch preprocess multiple KH-9 PC scans."""
    batch_preprocess_kh9pc(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=overwrite,
        keep_work=keep_work,
        n_jobs=n_jobs,
        dry_run=dry_run,
    )
