"""
Copyright (c) 2026 HIPP developers
Description: Click CLI exposing ``preproc`` and ``batch-preproc`` commands for KH-9 PC preprocessing.
"""

# mypy: disable-error-code="misc"
import logging
import sys
from pathlib import Path
from typing import Literal

import click

from hipp.kh9pc.pipeline import batch_preprocess_kh9pc, preprocess_kh9pc


def _configure_logging(verbosity: int) -> None:
    """Set the ``hipp`` logger level based on a verbosity count (0=WARNING, 1=INFO, 2+=DEBUG)."""
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
    )
    logging.getLogger("hipp").setLevel(level)


@click.group()
def main() -> None:
    """KH-9 Panoramic Camera preprocessing tools."""


@main.command()
@click.option("--entity-id", "-e", required=True, metavar="ID", help="Entity id of the scan to preprocess")
@click.option(
    "--archive-dir", "-a", required=True, type=Path, metavar="DIR", help="Directory containing {entity_id}.tgz"
)
@click.option(
    "--output-dir", "-o", required=True, type=Path, metavar="DIR", help="Output directory for georeferenced images"
)
@click.option("--work-dir", "-w", required=True, type=Path, metavar="DIR", help="Directory for intermediate files")
@click.option(
    "--scenes-metadata", "-m", required=True, type=Path, metavar="FILE", help="USGS scenes metadata file for GCPs"
)
@click.option("--plot-dir", "-p", type=Path, metavar="DIR", help="Directory for QC plots and quickviews")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.option(
    "--clean",
    type=click.Choice(["none", "heavy", "all"]),
    default="heavy",
    show_default=True,
    help="Intermediate files removed after a successful run",
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v INFO, -vv DEBUG)")
def preproc(
    entity_id: str,
    archive_dir: Path,
    output_dir: Path,
    work_dir: Path,
    scenes_metadata: Path,
    plot_dir: Path | None,
    overwrite: bool,
    clean: Literal["none", "heavy", "all"],
    verbose: int,
) -> None:
    """Preprocess a single KH-9 PC scan."""
    _configure_logging(verbose)
    preprocess_kh9pc(
        entity_id=entity_id,
        archive_dir=archive_dir,
        output_dir=output_dir,
        work_dir=work_dir,
        scenes_metadata_file=scenes_metadata,
        plot_dir=plot_dir,
        overwrite=overwrite,
        clean=clean,
    )


@main.command()
@click.option(
    "--archive-dir", "-a", required=True, type=Path, metavar="DIR", help="Directory containing the .tgz archives"
)
@click.option(
    "--output-dir", "-o", required=True, type=Path, metavar="DIR", help="Output directory for georeferenced images"
)
@click.option("--work-dir", "-w", required=True, type=Path, metavar="DIR", help="Directory for intermediate files")
@click.option(
    "--scenes-metadata", "-m", required=True, type=Path, metavar="FILE", help="USGS scenes metadata file for GCPs"
)
@click.option("--plot-dir", "-p", type=Path, metavar="DIR", help="Directory for QC plots and quickviews")
@click.option(
    "--entity-id",
    "-e",
    "entity_ids",
    multiple=True,
    metavar="ID",
    help="Entity id to process (repeatable, default: every archive in --archive-dir)",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.option(
    "--clean",
    type=click.Choice(["none", "heavy", "all"]),
    default="heavy",
    show_default=True,
    help="Intermediate files removed after a successful run",
)
@click.option("--n-process", "-j", default=1, show_default=True, help="Number of parallel processes")
@click.option("--dry-run", is_flag=True, help="Log the batch status without processing")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v INFO, -vv DEBUG)")
def batch_preproc(
    archive_dir: Path,
    output_dir: Path,
    work_dir: Path,
    scenes_metadata: Path,
    plot_dir: Path | None,
    entity_ids: tuple[str, ...],
    overwrite: bool,
    clean: Literal["none", "heavy", "all"],
    n_process: int,
    dry_run: bool,
    verbose: int,
) -> None:
    """Batch preprocess multiple KH-9 PC scans."""
    _configure_logging(verbose)
    batch_preprocess_kh9pc(
        archive_dir=archive_dir,
        output_dir=output_dir,
        work_dir=work_dir,
        scenes_metadata_file=scenes_metadata,
        plot_dir=plot_dir,
        entity_ids=list(entity_ids) or None,
        overwrite=overwrite,
        clean=clean,
        n_process=n_process,
        dry_run=dry_run,
    )
