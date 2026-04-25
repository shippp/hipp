# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HIPP (Historical Image Pre-Processing)** is a Python library for preprocessing scanned historical aerial and satellite images for photogrammetric analysis (Structure from Motion). It supports:
- Aerial images with fiducial markers (USGS, NAGAP datasets)
- Declassified US reconnaissance satellite images (KH-9 Hexagon panoramic/mapping camera, KH-4/4A/4B Corona)

## Commands

The project uses **Hatch** as the project manager.

```bash
hatch shell dev         # Enter development environment
hatch run dev:check     # Type check (mypy --strict) + lint (ruff)
hatch run dev:pytest    # Run tests
hatch run dev:lab       # Start Jupyter Lab on port 8333
hatch run dev:kernel    # Install IPython kernel
```

Direct commands (inside `hatch shell dev`):
```bash
pytest                            # All tests
pytest tests/aerial/test_core.py  # Single test file
ruff check .                      # Lint
mypy src/ --strict --ignore-missing-imports --no-warn-unused-ignores --allow-untyped-calls
pre-commit install                # Install git hooks (run once after cloning)
```

Line length is 120 characters. Pre-commit hooks run ruff + mypy on every commit.

## Architecture

### Package Layout (`src/hipp/`)

```
hipp/
├── image.py         # Low-level image I/O, CLAHE, resizing
├── math.py          # Geometric transforms, matrix ops
├── intrinsics.py    # Intrinsics class (camera calibration parameters)
├── tools.py         # GUI point picking, archive extraction, quickviews
├── aerial/          # Fiducial-based aerial image preprocessing
│   ├── core.py      # Main pipeline: template creation → detection → restitution
│   ├── fiducials.py # Fiducial marker detection, matching, transformation
│   └── quality_control.py
├── kh9pc/           # KH-9 panoramic camera preprocessing
│   ├── pipeline.py  # End-to-end orchestration (PipelineStep, KH9Pipeline, PipelineConfig)
│   ├── image_mosaic.py   # ORB keypoint matching, RANSAC, image stitching (ImageAlignment)
│   ├── batch.py          # Batch join_images
│   ├── quality_control.py
│   ├── utils.py
│   └── restitution/      # Image rectification
│       ├── types.py      # StepResult, StrategyAttempt data classes
│       ├── strategy.py   # RectificationStrategy + Collimation/Poly/Flat strategies
│       ├── vertical.py   # VerticalDetector (collimation line detection)
│       └── plotters.py
└── dataquery/       # USGS/NAGAP data download
```

### Data Flow

**Aerial pipeline** (`hipp.aerial.core`):
1. `create_fiducial_templates()` — user picks fiducial locations on reference image
2. `iter_detect_fiducials()` — OpenCV template matching on input images
3. `filter_detected_fiducials()` — removes low-confidence matches
4. `compute_transformations()` — estimates affine/similarity transforms
5. `iter_image_restitution()` — crops, applies CLAHE, outputs standardized images

**KH-9 pipeline** (`hipp.kh9pc.pipeline.KH9Pipeline`):
1. Extract archive → list of TIF scan strips
2. `join_images()` — stitch strips via ORB keypoints + RANSAC affine alignment
3. Restitute (rectify):
   - Detect vertical collimation edges (`VerticalDetector`)
   - Detect horizontal edges with strategy fallback: `CollimationStrategy` → `PolyStrategy` → `FlatStrategy`
   - Apply analytical inverse-map transform (bilinear interpolation between fitted polynomial curves via `build_inverse_map`)
4. Generate QC reports

Valid `PipelineConfig.steps` names (in order): `extract`, `join_images`, `quickview_mosaic`, `restitution`, `quickview_final`, `qc_report`.

**CLI** (`python -m hipp.kh9pc`):
```bash
python -m hipp.kh9pc --input scan.tgz --output /out/images/DZB1215.tif --qc-dir /out/qc
python -m hipp.kh9pc --input t1.tif t2.tif t3.tif --output /out/DZB1215.tif --qc-dir /out/qc
python -m hipp.kh9pc --input scan.tgz --output /out/DZB1215.tif --qc-dir /out/qc --config cfg.toml
```
`PipelineConfig.from_toml()` accepts keys: `overwrite`, `cleanup`, `steps`, and `output_height` (integer, default `22064`).

### Key Patterns

- **`PipelineStep`**: declarative step class with `inputs`/`outputs`/`overwrite` — enables skip-if-done logic
- **Strategy pattern** in `kh9pc/restitution/strategy.py`: multiple fallback strategies for edge detection; all inherit `RectificationStrategy` ABC which chains `_fit()` + `make_inverse_map()` + centering translation into `transform(output_height)`; strategies expose `detected_region() → ((col_off, row_top), (width, height))`
- **Pandas Series for fiducials**: coordinate data stored with named keys like `corner_top_left_x`, `midside_left_x`
- **`Intrinsics` class**: wraps focal length, pixel pitch, true fiducial coordinates in mm, principal point
- **3×3 homogeneous matrices** throughout for image transforms
- **Rasterio** for all geospatial raster I/O; **OpenCV** for image operations; `build_inverse_map` in `utils.py` for the analytical curve-interpolation warp
- **Intermediate files persisted as `.joblib`**: `vertical.joblib`, `horizontal.joblib`, `alignments.joblib` — individual steps can be re-run by loading these directly

### Notebooks

Practical usage examples live in `notebooks/`:
- `aerial_preprocessing.ipynb` — aerial fiducial workflow
- `kh9pc_preprocessing.ipynb` — full KH-9 pipeline
- `kh9pc_collimation_rectification.ipynb` — detailed rectification walkthrough
