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

Line length is 120 characters. Pre-commit hooks run ruff (lint + format) on every commit; mypy is not a hook, run `hatch run dev:check` before merging.

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
│   ├── pipeline.py       # preprocess_kh9pc / batch_preprocess_kh9pc, run status, STEP_OUTPUTS layout
│   ├── mosaic.py         # image_mosaic() — ORB keypoints + RANSAC Euclidean + WarpedVRT compositing
│   ├── transformations.py    # Transformation model, compute_*_transformation, apply_transformation
│   ├── georeferencing.py     # read_corners() / set_corner_gcps() — corner GCPs from the USGS metadata
│   ├── fiducial_patterns.py  # Fiducial pattern geometry, TPS control-point computation
│   ├── kh9_image_spec.py     # KH9ImageSpec — per-mission dimensions, fiducial type, collimation flag
│   ├── quality_control.py    # QC plot functions, each reading a step's JSON result
│   ├── cli.py            # Click CLI (preproc / batch-preproc commands)
│   ├── __main__.py       # Entry point for `python -m hipp.kh9pc`
│   └── detection/        # Detection steps: raster in → JSON result out
│       ├── base.py       # DetectionError, JsonModel, DetectionResult, detect_ruptures, fit_ransac_poly
│       ├── detect_vertical_edges.py         # left/right film edges
│       ├── detect_poly_horizontal_edges.py  # RANSAC polynomial top/bottom edges
│       ├── detect_flat_horizontal_edges.py  # flat top/bottom edges
│       ├── detect_collimation_lines.py      # collimation line peaks
│       ├── detect_fiducials.py              # template-matched fiducials (templates/*.png)
│       └── templates/
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
1. `extract_archive()` — `{archive_dir}/{entity_id}.tgz` → TIF scan strips
2. `image_mosaic()` — stitch strips via ORB keypoints + RANSAC Euclidean alignment + WarpedVRT block compositing
3. `detect_vertical_edges()` — left/right film edges
4. `detect_poly_horizontal_edges()` — top/bottom edges; raises `DetectionError` if not trustworthy
5. `detect_fiducials()` — template-match fiducials along the edges; raises `DetectionError` if not trustworthy
6. `compute_fiducial_transformation()` — TPS control points + crop → `transformation.json`
7. Restitution — `read_corners()` first (fails fast on a scene missing from the metadata), then
   `apply_transformation()` + `set_corner_gcps()` inside one `atomic_output` → final GeoTIFF with corner GCPs (EPSG:4326),
   so the full raster is written once
8. QC plots/quickviews after each step, only when `plot_dir` is given

File layout (`STEP_OUTPUTS`, `PLOTS_OUTPUTS` in `pipeline.py`):
- `{output_dir}/{entity_id}.tif` — final georeferenced image
- `{work_dir}/{entity_id}/` — `extracted/`, mosaic `{entity_id}.tif`, step JSON results, `preprocess.log`,
  `status.json`
- `{plot_dir}/{entity_id}/` — QC figures and quickviews

**CLI** (`python -m hipp.kh9pc` or `hipp-kh9pc`):
```bash
hipp-kh9pc preproc -e D3C1210-... -a /data/archives -o /out/images -w /out/work -m scenes.csv -p /out/plots
hipp-kh9pc batch-preproc -a /data/archives -o /out/images -w /out/work -m scenes.csv -j 4 -v
hipp-kh9pc batch-preproc -a /data/archives -o /out/images -w /out/work -m scenes.csv --dry-run -v  # status only
hipp-kh9pc preproc --help
```

### Key Patterns

- **File-in / file-out steps** (KH-9): every step is a function `step(input_file, output_file, ..., overwrite=False)` that is skipped when its output exists (`skip_existing`), so steps chain through files and a re-run resumes where it stopped.
- **`JsonModel` / `DetectionResult`** (`detection/base.py`): pydantic models with `load()` / `save()` JSON round-trip. Each detection writes a `*DetectionResult`; `failure_reason` / `is_failed` say whether it is trustworthy. QC plots read these JSON files.
- **`Transformation`** (`transformations.py`): a `JsonModel` with control points (`kind`: identity / affine / tps), `crop_offset` and `output_size`; `compute_*_transformation()` builds it, `apply_transformation()` warps the raster.
- **`KH9ImageSpec`** (`kh9_image_spec.py`): derived from filename entity ID (e.g. `D3C1210-…`), gives expected size, fiducial type (`disk` / `wagon_wheel`), collimation presence, and fiducial pattern names. Covers missions 1201–1219.
- **Run status**: `status.json` records `running` / `failed` / `done`; `preprocess_status()` maps it to `done` / `failed` / `incomplete` / `todo`. `batch_preprocess_kh9pc` skips `done` entities unless `overwrite`.
- **Intermediate files preserved on failure**: the `clean` level (`none` / `heavy` / `all`, see `CLEAN_STEPS`) only removes step outputs after a successful run, so a failed entity can be resumed.
- **`Pandas Series` for fiducials** (aerial module): coordinate data stored with named keys like `corner_top_left_x`, `midside_left_x`.
- **`Intrinsics` class**: wraps focal length, pixel pitch, true fiducial coordinates in mm, principal point (aerial module).
- **3×3 homogeneous matrices** throughout for image transforms.
- **Rasterio** for all geospatial raster I/O; **OpenCV** for image operations and template matching.

### Notebooks

Practical usage examples live in `notebooks/`:
- `aerial_preprocessing.ipynb` — aerial fiducial workflow
- `kh9pc_preprocessing.ipynb` — full KH-9 pipeline
- `kh9pc_collimation_rectification.ipynb` — detailed rectification walkthrough
