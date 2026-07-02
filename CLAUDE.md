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
│   ├── pipeline.py       # preprocess_kh9pc / batch_preprocess_kh9pc (orchestration)
│   ├── mosaic.py         # image_mosaic() — ORB keypoints + RANSAC Euclidean + WarpedVRT compositing
│   ├── fiducial_patterns.py  # Fiducial pattern geometry, TPS control-point computation
│   ├── kh9_image_spec.py     # KH9ImageSpec — per-mission dimensions, fiducial type, collimation flag
│   ├── quality_control.py    # QC figure generation
│   ├── cli.py            # Click CLI (preproc / batch-preproc commands)
│   ├── __main__.py       # Entry point for `python -m hipp.kh9pc`
│   └── restitution/      # Geometric correction strategies
│       ├── base.py       # FittingClass ABC, DetectionError, detect_ruptures, fit_ransac_poly
│       ├── vertical_detector.py   # VerticalDetector — left/right film-edge detection
│       ├── poly_strategy.py       # PolyStrategy — RANSAC polynomial top/bottom edges
│       ├── collimation_strategy.py # CollimationStrategy — refines poly via collimation peaks
│       ├── flat_strategy.py       # FlatStrategy — fallback for missions without collimation lines
│       ├── fiducial_strategy.py   # FiducialStrategy — TPS warp via template-matched fiducials
│       └── mixed_strategy.py      # MixedStrategy — ordered fallback chain across strategies
└── dataquery/       # USGS/NAGAP data download
```

### Data Flow

**Aerial pipeline** (`hipp.aerial.core`):
1. `create_fiducial_templates()` — user picks fiducial locations on reference image
2. `iter_detect_fiducials()` — OpenCV template matching on input images
3. `filter_detected_fiducials()` — removes low-confidence matches
4. `compute_transformations()` — estimates affine/similarity transforms
5. `iter_image_restitution()` — crops, applies CLAHE, outputs standardized images

**KH-9 pipeline** (`hipp.kh9pc.pipeline`):
1. Extract archive (if `.tgz`) → list of TIF scan strips
2. `image_mosaic()` — stitch strips via ORB keypoints + RANSAC Euclidean alignment + WarpedVRT block compositing
3. `FiducialStrategy().fit(joined_image)` — template-match fiducials, compute TPS control points, build warp
4. `strategy.transform(output_path)` — apply TPS inverse warp → restituted GeoTIFF
5. QC quickviews + figures generated at each major step

Output layout under `output_dir/`:
- `images/{entity_id}.tif` — final restituted image
- `qc/mosaic_qv/`, `qc/restitution/`, `qc/final_qv/` — quickviews and QC figures
- `work/joined_images/`, `work/joblibs/` — intermediates (preserved on failure for retry)
- `logs/{entity_id}.log` — per-image log

**CLI** (`python -m hipp.kh9pc` or `hipp-kh9pc`):
```bash
hipp-kh9pc preproc -i scan.tgz -o /out/
hipp-kh9pc preproc -i t1.tif t2.tif t3.tif -o /out/
hipp-kh9pc batch-preproc -i /data/scans/ -o /out/ -j 4 -v
hipp-kh9pc preproc --help
```

### Key Patterns

- **`FittingClass` ABC** (`restitution/base.py`): all strategies inherit this. `fit(raster_filepath)` calls `_fit()` and records `raster_filepath_`. `is_failed` signals whether the result is usable. `transform(output_path)` applies the computed warp.
- **Strategy hierarchy**: `VerticalDetector` → `PolyStrategy` (uses VerticalDetector) → `CollimationStrategy` (uses PolyStrategy) → `FiducialStrategy` (uses PolyStrategy + TPS); `FlatStrategy` is the fallback for missions without collimation lines; `MixedStrategy` chains them with ordered fallback.
- **`KH9ImageSpec`** (`kh9_image_spec.py`): derived from filename entity ID (e.g. `D3C1210-…`), gives expected size, fiducial type (`disk` / `wagon_wheel`), collimation presence, and fiducial pattern names. Covers missions 1201–1219.
- **`Transformation` dataclass**: carries the TPS `deformation`, `crop_offset`, and `output_size`; passed from `FiducialStrategy._compute_transformation()` to `transform()`.
- **Intermediate files preserved on failure**: `work/joined_images/` and `work/joblibs/` are only deleted on successful completion (`keep_work=False`), so a re-run with `overwrite=False` resumes from the last completed step.
- **`Pandas Series` for fiducials** (aerial module): coordinate data stored with named keys like `corner_top_left_x`, `midside_left_x`.
- **`Intrinsics` class**: wraps focal length, pixel pitch, true fiducial coordinates in mm, principal point (aerial module).
- **3×3 homogeneous matrices** throughout for image transforms.
- **Rasterio** for all geospatial raster I/O; **OpenCV** for image operations and template matching.

### Notebooks

Practical usage examples live in `notebooks/`:
- `aerial_preprocessing.ipynb` — aerial fiducial workflow
- `kh9pc_preprocessing.ipynb` — full KH-9 pipeline
- `kh9pc_collimation_rectification.ipynb` — detailed rectification walkthrough
