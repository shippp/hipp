"""
Copyright (c) 2026 HIPP developers
Description: High-level pipeline orchestrating the full KH-9 PC preprocessing workflow:
    archive extraction → image mosaicking → edge and fiducial detection → restitution (warp + georeferencing).
    Exposes ``preprocess_kh9pc`` for single images and ``batch_preprocess_kh9pc`` for
    parallel batch processing.
"""

import glob
import logging
import shutil
from collections import Counter
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from hipp.image import generate_quickview
from hipp.kh9pc import quality_control as qc
from hipp.kh9pc.detection.base import DetectionError, JsonModel
from hipp.kh9pc.detection.detect_fiducials import FiducialDetectionResult, detect_fiducials
from hipp.kh9pc.detection.detect_poly_horizontal_edges import PolyDetectionResult, detect_poly_horizontal_edges
from hipp.kh9pc.detection.detect_vertical_edges import detect_vertical_edges
from hipp.kh9pc.georeferencing import read_corners, set_corner_gcps
from hipp.kh9pc.mosaic import image_mosaic
from hipp.kh9pc.transformations import apply_transformation, compute_fiducial_transformation
from hipp.tools import atomic_output, extract_archive, resolve_wildcards, skip_existing

logger = logging.getLogger(__name__)


Step = Literal["extract", "mosaic", "vertical", "poly", "fiducials", "transformation", "restitution"]

STEP_OUTPUTS: dict[Step, list[str]] = {
    "extract": ["{work_dir}/{entity_id}/extracted"],
    "mosaic": ["{work_dir}/{entity_id}/{entity_id}.tif"],
    "vertical": ["{work_dir}/{entity_id}/vertical_detection_result.json"],
    "poly": ["{work_dir}/{entity_id}/poly_detection_result.json"],
    "fiducials": ["{work_dir}/{entity_id}/fiducial_detection_result.json"],
    "transformation": ["{work_dir}/{entity_id}/transformation.json"],
    "restitution": ["{output_dir}/{entity_id}.tif"],
}

LOG_FILE = "{work_dir}/{entity_id}/preprocess.log"
STATUS_FILE = "{work_dir}/{entity_id}/status.json"
ARCHIVE_FILE = "{archive_dir}/{entity_id}.tgz"

Status = Literal["done", "failed", "incomplete", "todo"]


class RunStatus(JsonModel):
    """State of the last preprocessing run of one entity."""

    state: Literal["running", "failed", "done"]
    error: str | None = None


# step outputs removed after a successful run, per ``clean`` level
CLEAN_STEPS: dict[Literal["none", "heavy", "all"], list[Step]] = {
    "none": [],
    "heavy": ["extract", "mosaic"],
    "all": ["extract", "mosaic", "vertical", "poly", "fiducials", "transformation"],
}

PLOTS_OUTPUTS: dict[Step, list[str]] = {
    "mosaic": ["{plot_dir}/{entity_id}/mosaic_qv.jpg"],
    "vertical": [
        "{plot_dir}/{entity_id}/vertical_edges.png",
        "{plot_dir}/{entity_id}/vertical_ruptures.png",
    ],
    "poly": [
        "{plot_dir}/{entity_id}/poly_edges.png",
        "{plot_dir}/{entity_id}/poly_distortions.png",
    ],
    "fiducials": [
        "{plot_dir}/{entity_id}/fiducial_filtering.png",
        "{plot_dir}/{entity_id}/fiducial_detected_profiles.png",
        "{plot_dir}/{entity_id}/fiducial_distortions.png",
        "{plot_dir}/{entity_id}/fiducial_detected_boxes_top.png",
        "{plot_dir}/{entity_id}/fiducial_detected_boxes_bottom.png",
    ],
    "transformation": [
        "{plot_dir}/{entity_id}/crop_area.png",
        "{plot_dir}/{entity_id}/deformation_grid.png",
    ],
    "restitution": ["{plot_dir}/{entity_id}/final_qv.jpg", "{plot_dir}/{entity_id}/georeferencing.png"],
}


@contextmanager
def _log_to_file(path: Path) -> Generator[None, None, None]:
    """Temporarily attach a FileHandler to the hipp logger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    hipp_logger = logging.getLogger("hipp")
    previous_level = hipp_logger.level
    # without logging configured (plain API use) the hipp logger stays at WARNING and the file would miss INFO lines
    if hipp_logger.getEffectiveLevel() > logging.INFO:
        hipp_logger.setLevel(logging.INFO)
    hipp_logger.addHandler(handler)
    try:
        yield
    finally:
        hipp_logger.removeHandler(handler)
        hipp_logger.setLevel(previous_level)
        handler.close()


def preprocess_kh9pc(
    entity_id: str,
    archive_dir: str | Path,
    output_dir: str | Path,
    work_dir: str | Path,
    scenes_metadata_file: str | Path,
    plot_dir: str | Path | None = None,
    overwrite: bool = False,
    clean: Literal["none", "heavy", "all"] = "heavy",
) -> None:
    """Preprocess the archive of *entity_id* into a restituted GeoTIFF, following the ``STEP_OUTPUTS`` layout.

    The GeoTIFF gets corner GCPs from the USGS *scenes_metadata_file*. QC plots and quickviews are written to
    *plot_dir* if given, the log to ``LOG_FILE``. Skipped if the output already exists and *overwrite* is False.
    Raises DetectionError if the poly or fiducial detection is not trustworthy.
    """
    fields = {"entity_id": entity_id, "archive_dir": archive_dir, "output_dir": output_dir, "work_dir": work_dir}
    # every step has a single output
    outputs = {step: Path(resolve_wildcards(templates, **fields)[0]) for step, templates in STEP_OUTPUTS.items()}

    if skip_existing(outputs["restitution"], overwrite):
        return

    plots: dict[Step, list[Path]] = {}
    if plot_dir is not None:
        plots = {
            step: [Path(p) for p in resolve_wildcards(templates, **fields, plot_dir=plot_dir)]
            for step, templates in PLOTS_OUTPUTS.items()
        }

    with _log_to_file(Path(resolve_wildcards([LOG_FILE], **fields)[0])):
        logger.info("%s - start preprocessing", entity_id)
        status_file = resolve_wildcards([STATUS_FILE], **fields)[0]
        RunStatus(state="running").save(status_file)
        try:
            archive = resolve_wildcards([ARCHIVE_FILE], **fields)[0]
            extract_archive(archive, outputs["extract"], overwrite=overwrite)

            tiles = sorted(
                (p for p in outputs["extract"].rglob("*") if p.suffix.lower() in (".tif", ".tiff")),
                key=lambda p: p.name.lower(),
            )
            if not tiles:
                raise FileNotFoundError(f"no .tif tiles found in {outputs['extract']} (extracted from {archive})")
            image_mosaic(tiles, outputs["mosaic"], overwrite=overwrite)
            if plots:
                (mosaic_qv,) = plots["mosaic"]
                generate_quickview(outputs["mosaic"], mosaic_qv, scale_factor=0.1, jpeg_quality=70, overwrite=overwrite)

            detect_vertical_edges(outputs["mosaic"], outputs["vertical"], overwrite=overwrite)
            if plots:
                edges_png, ruptures_png = plots["vertical"]
                qc.plot_vertical_edges(outputs["vertical"], edges_png, overwrite=overwrite)
                qc.plot_vertical_ruptures(outputs["vertical"], ruptures_png, overwrite=overwrite)

            detect_poly_horizontal_edges(outputs["mosaic"], outputs["poly"], outputs["vertical"], overwrite=overwrite)
            if plots:
                edges_png, distortions_png = plots["poly"]
                qc.plot_poly_edges(outputs["poly"], edges_png, overwrite=overwrite)
                qc.plot_poly_distortions(outputs["poly"], distortions_png, overwrite=overwrite)
            # checked after plotting so a failed fit still gets its QC figures
            poly = PolyDetectionResult.load(outputs["poly"])
            if poly.failure_reason:
                raise DetectionError(f"poly detection failed: {poly.failure_reason}")

            detect_fiducials(outputs["mosaic"], outputs["fiducials"], outputs["poly"], overwrite=overwrite)
            if plots:
                filtering_png, profiles_png, distortions_png, boxes_top_png, boxes_bottom_png = plots["fiducials"]
                qc.plot_fiducial_filtering(outputs["fiducials"], filtering_png, overwrite=overwrite)
                qc.plot_fiducial_detected_profiles(outputs["fiducials"], profiles_png, overwrite=overwrite)
                qc.plot_fiducial_distortions(outputs["fiducials"], distortions_png, overwrite=overwrite)
                qc.plot_fiducial_detected_boxes(outputs["fiducials"], boxes_top_png, "top", overwrite=overwrite)
                qc.plot_fiducial_detected_boxes(outputs["fiducials"], boxes_bottom_png, "bottom", overwrite=overwrite)
            fiducials = FiducialDetectionResult.load(outputs["fiducials"])
            if fiducials.failure_reason:
                raise DetectionError(f"fiducial detection failed: {fiducials.failure_reason}")

            compute_fiducial_transformation(outputs["fiducials"], outputs["transformation"], overwrite=overwrite)
            if plots:
                crop_png, grid_png = plots["transformation"]
                qc.plot_crop_area(outputs["transformation"], crop_png, overwrite=overwrite)
                qc.plot_deformation_grid(outputs["transformation"], grid_png, overwrite=overwrite)

            # read before the warp so a scene missing from the metadata fails in seconds, not after it
            corners = read_corners(entity_id, scenes_metadata_file)
            # one atomic output for warp + GCPs: the full raster is written once and never exists without its GCPs
            with atomic_output(outputs["restitution"]) as tmp:
                apply_transformation(outputs["transformation"], tmp, overwrite=True)
                set_corner_gcps(tmp, corners)
            if plots:
                final_qv, georef_png = plots["restitution"]
                generate_quickview(
                    outputs["restitution"], final_qv, scale_factor=0.1, jpeg_quality=70, overwrite=overwrite
                )
                qc.plot_georeferencing(outputs["restitution"], georef_png, overwrite=overwrite)
        except Exception as error:
            logger.exception("%s - preprocessing failed", entity_id)
            RunStatus(state="failed", error=f"{type(error).__name__}: {error}").save(status_file)
            raise
        RunStatus(state="done").save(status_file)
        logger.info("%s - finish preprocessing", entity_id)

    # only reached on success: a failed entity keeps its work files to resume from
    # remove the listed outputs only, so the cleaning follows the layout whatever the folder structure
    for step in CLEAN_STEPS[clean]:
        if outputs[step].is_dir():
            shutil.rmtree(outputs[step])
        else:
            outputs[step].unlink(missing_ok=True)


def preprocess_status(entity_id: str, output_dir: str | Path, work_dir: str | Path) -> Status:
    """Status of *entity_id*, read from its output and the ``STATUS_FILE`` of its last run.

    ``incomplete`` means a run started but never recorded its end: it was interrupted, or is still running.
    """
    fields = {"entity_id": entity_id, "output_dir": output_dir, "work_dir": work_dir}
    if Path(resolve_wildcards(STEP_OUTPUTS["restitution"], **fields)[0]).exists():
        return "done"
    status_file = Path(resolve_wildcards([STATUS_FILE], **fields)[0])
    if not status_file.exists():
        return "todo"
    return "failed" if RunStatus.load(status_file).state == "failed" else "incomplete"


def batch_preprocess_status(
    archive_dir: str | Path, output_dir: str | Path, work_dir: str | Path, entity_ids: list[str] | None = None
) -> dict[str, Status]:
    """Status of every entity of a batch, *entity_ids* defaulting to the archives found in *archive_dir*."""
    if entity_ids is None:
        entity_ids = find_entity_ids(archive_dir)
    return {entity_id: preprocess_status(entity_id, output_dir, work_dir) for entity_id in entity_ids}


def find_entity_ids(archive_dir: str | Path) -> list[str]:
    """Entity ids of the archives in *archive_dir* matching ``ARCHIVE_FILE``, the id being the archive file stem."""
    pattern = resolve_wildcards([ARCHIVE_FILE], archive_dir=archive_dir, entity_id="*")[0]
    return sorted(Path(path).stem for path in glob.glob(pattern))


def batch_preprocess_kh9pc(
    archive_dir: str | Path,
    output_dir: str | Path,
    work_dir: str | Path,
    scenes_metadata_file: str | Path,
    plot_dir: str | Path | None = None,
    entity_ids: list[str] | None = None,
    overwrite: bool = False,
    clean: Literal["none", "heavy", "all"] = "heavy",
    n_process: int = 1,
    dry_run: bool = False,
) -> None:
    """Run ``preprocess_kh9pc`` in parallel on *entity_ids*, defaulting to the archives found in *archive_dir*.

    Entities already done are skipped unless *overwrite* is set. A failure is logged without aborting the batch; its
    traceback is in the entity log. With *dry_run*, only the status of the batch is logged.
    """
    statuses = batch_preprocess_status(archive_dir, output_dir, work_dir, entity_ids)
    todo = list(statuses) if overwrite else [e for e, status in statuses.items() if status != "done"]

    logger.info("Batch preprocess — %d images", len(statuses))
    logger.info("  output_dir : %s", output_dir)
    logger.info("  n_process  : %d  |  clean : %s  |  overwrite : %s", n_process, clean, overwrite)
    logger.info("  status     : %s", dict(Counter(statuses.values())))
    for entity_id, status in statuses.items():
        if status == "failed":
            status_file = resolve_wildcards([STATUS_FILE], work_dir=work_dir, entity_id=entity_id)[0]
            logger.info("  failed     : %s - %s", entity_id, RunStatus.load(status_file).error)
    logger.info("  to process : %d", len(todo))

    if dry_run:
        return

    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=n_process) as executor:
        futures = {
            executor.submit(
                preprocess_kh9pc,
                entity_id,
                archive_dir,
                output_dir,
                work_dir,
                scenes_metadata_file,
                plot_dir,
                overwrite,
                clean,
            ): entity_id
            for entity_id in todo
        }
        for future in as_completed(futures):
            entity_id = futures[future]
            try:
                future.result()
            # one failed entity must not abort the batch; its traceback is in its own log
            except Exception as error:  # noqa: BLE001
                log_file = resolve_wildcards([LOG_FILE], work_dir=work_dir, entity_id=entity_id)[0]
                logger.error("%s - failed: %s (see %s)", entity_id, error, log_file)
                failed.append(entity_id)

    logger.info("Batch finished — %d succeeded, %d failed %s", len(todo) - len(failed), len(failed), sorted(failed))
