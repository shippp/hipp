import logging
from collections.abc import Sequence
from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from hipp.image import generate_quickview
from hipp.kh9pc.image_mosaic import image_mosaic
from hipp.kh9pc.quality_control import get_figures
from hipp.kh9pc.restitution_strategy.mixed_strategy import MixedStrategy
from hipp.tools import extract_archive

logger = logging.getLogger(__name__)


def preprocess_kh9pc(
    input: str | Path | Sequence[str | Path],
    output_path: str | Path,
    work_dir: str | Path | None = None,
    qc_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    # standardize path
    input_paths: Path | list[Path] = Path(input) if isinstance(input, (str, Path)) else [Path(f) for f in input]
    output_path = Path(output_path)
    work_dir = Path(work_dir) if work_dir else output_path.parent / "_work"
    qc_dir = Path(qc_dir) if qc_dir else None

    # extract entity id from input
    entity_id = input_paths.stem if isinstance(input_paths, Path) else input_paths[0].stem.split("_")[0]

    # overwrite checking
    if output_path.exists() and not overwrite:
        logger.info("Skipping preprocess_kh9pc: %s (already exists, overwrite=False)", str(output_path))
        return

    # START PREPROCESSING
    logger.info("Start preprocessing of %s", entity_id)

    # STEP 1 : EXTRACTION (can be skipped if the input is a list)
    if isinstance(input_paths, Path):
        tiles = extract_archive(input_paths, work_dir / "tiles", overwrite=overwrite)
    else:
        tiles = input_paths

    # STEP 2 : JOIN_IMAGES
    joined_image = work_dir / "mosaic.tif"
    image_mosaic(tiles, joined_image, overwrite=overwrite)

    # QC STEP : QUICKVIEW (skipped if no qc dir is provideed)
    if qc_dir:
        generate_quickview(
            joined_image,
            qc_dir / "mosaic_qv" / f"{entity_id}.jpg",
            scale_factor=0.1,
            jpeg_quality=70,
            overwrite=overwrite,
        )

    # STEP 3 : RESTITUTION
    strategy = MixedStrategy().fit(joined_image)
    joblib.dump(strategy, work_dir / "strategy.joblib")
    strategy.transform(output_path)

    # QC STEP : RESTITUTION (skipped if no qc dir is provideed)
    if qc_dir:
        qc_restitution_dir = qc_dir / "restitution" / entity_id
        qc_restitution_dir.mkdir(exist_ok=True, parents=True)
        for i, figure in enumerate(get_figures(strategy)):
            figure.savefig(qc_restitution_dir / f"{i}.png")
            plt.close(figure)

    # QC STEP : QUICKVIEW (skipped if no qc dir is provideed)
    if qc_dir:
        generate_quickview(
            output_path,
            qc_dir / "final_qv" / f"{entity_id}.jpg",
            scale_factor=0.1,
            jpeg_quality=70,
            overwrite=overwrite,
        )
