import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

import joblib

from hipp.image import generate_quickview
from hipp.kh9pc.mosaic import image_mosaic
from hipp.kh9pc.quality_control import save_figures
from hipp.kh9pc.restitution.fiducial_strategy import FiducialStrategy
from hipp.tools import extract_archive

logger = logging.getLogger(__name__)


def preprocess_kh9pc(
    input: str | Path | Sequence[str | Path], output_dir: str | Path, overwrite: bool = False, keep_work: bool = False
) -> None:
    # standardize path
    input_paths: Path | list[Path] = Path(input) if isinstance(input, (str, Path)) else [Path(f) for f in input]
    output_dir = Path(output_dir)

    # extract entity id from input
    entity_id = input_paths.stem if isinstance(input_paths, Path) else input_paths[0].stem.split("_")[0]

    # create all path
    output_path = output_dir / "images" / f"{entity_id}.tif"
    qc_dir = output_dir / "qc"
    work_dir = output_dir / "work"

    # overwrite checking
    if output_path.exists() and not overwrite:
        logger.info("Skipping preprocess_kh9pc: %s (already exists, overwrite=False)", str(output_path))
        return

    # START PREPROCESSING
    logger.info("Start preprocessing of %s", entity_id)

    # STEP 1 : EXTRACTION (can be skipped if the input is a list)
    if isinstance(input_paths, Path):
        tiles = extract_archive(input_paths, work_dir / "extracted" / entity_id, overwrite=overwrite)
    else:
        tiles = input_paths

    # STEP 2 : JOIN_IMAGES
    joined_image = work_dir / "joined_images" / f"{entity_id}.tif"
    image_mosaic(tiles, joined_image, overwrite=overwrite)

    # QC STEP : QUICKVIEW
    generate_quickview(
        joined_image,
        qc_dir / "mosaic_qv" / f"{entity_id}.jpg",
        scale_factor=0.1,
        jpeg_quality=70,
        overwrite=overwrite,
    )

    # STEP 3 : RESTITUTION
    strategy = FiducialStrategy().fit(joined_image)
    (work_dir / "joblibs").mkdir(parents=True, exist_ok=True)
    joblib.dump(strategy, work_dir / "joblibs" / f"{entity_id}.joblib")

    # QC STEP : RESTITUTION
    save_figures(strategy, qc_dir / "restitution")

    strategy.transform(output_path)

    # QC STEP : QUICKVIEW (skipped if no qc dir is provideed)
    generate_quickview(
        output_path,
        qc_dir / "final_qv" / f"{entity_id}.jpg",
        scale_factor=0.1,
        jpeg_quality=70,
        overwrite=overwrite,
    )

    # clean the work dir
    if not keep_work:
        shutil.rmtree(work_dir)

    logger.info("Finish preprocessing of %s", entity_id)


def search_input_dir(input_dir: str | Path) -> list[Path | list[Path]]:
    """Scan a directory and return inputs ready for preprocess_kh9pc, one entry per image.

    - .tgz files at root       → one Path per archive
    - subdirectories with .tif → one list[Path] of tiles per subdir
    - .tif files at root       → grouped by entity_id prefix into list[Path]
    Mixed directories are supported.
    """
    from itertools import groupby

    input_dir = Path(input_dir)
    result: list[Path | list[Path]] = []

    result.extend(sorted(input_dir.glob("*.tgz")))

    for subdir in sorted(d for d in input_dir.iterdir() if d.is_dir()):
        tiles = sorted(subdir.glob("*.tif"))
        if tiles:
            result.append(tiles)

    loose = sorted(input_dir.glob("*.tif"))
    if loose:

        def _entity_id(p: Path) -> str:
            return p.stem.split("_")[0]

        for _, group in groupby(loose, key=_entity_id):
            result.append(list(group))

    return result


def batch_preprocess_kh9pc(
    input_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    keep_work: bool = False,
    n_jobs: int = 1,
    dry_run: bool = False,
) -> None:
    """Run preprocess_kh9pc on all images found in input_dir, logging failures without stopping the batch."""
    output_dir = Path(output_dir)

    def entity_id(inp: Path | list[Path]) -> str:
        return inp.stem if isinstance(inp, Path) else inp[0].stem.split("_")[0]

    inputs = search_input_dir(input_dir)
    done = [inp for inp in inputs if (output_dir / "images" / f"{entity_id(inp)}.tif").exists()]
    todo = [inp for inp in inputs if inp not in done]

    logger.info("Batch preprocess — %d images found in %s", len(inputs), input_dir)
    logger.info("  output_dir : %s", output_dir)
    logger.info("  n_jobs     : %d  |  keep_work : %s  |  overwrite : %s", n_jobs, keep_work, overwrite)
    logger.info("  done       : %d  %s", len(done), [entity_id(i) for i in done])
    logger.info("  remaining  : %d  %s", len(todo), [entity_id(i) for i in todo])

    if dry_run:
        return

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(preprocess_kh9pc, inp, output_dir, overwrite, keep_work): inp for inp in inputs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.error("Failed to process %s", entity_id(futures[future]), exc_info=True)
