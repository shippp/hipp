"""
Copyright (c) 2025 HIPP developers
Description: End-to-end preprocessing pipeline for KH-9 Panoramic Camera imagery.
"""

import contextvars
import importlib.metadata
import json
import logging
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from hipp.image import generate_quickview
from hipp.tools import extract_archive
from hipp.kh9pc.image_mosaic import ImageAlignment, compute_sequential_alignments, write_mosaic
from hipp.kh9pc.restitution.types import StepResult, StrategyAttempt
from hipp.kh9pc.restitution.vertical import VerticalDetector
from hipp.kh9pc.restitution.output_size import AutoSize, FixedHeightSize, FixedSize, MarginSize, OutputSize, SameSize
from hipp.kh9pc.restitution.strategy import (
    CollimationStrategy,
    FlatStrategy,
    PolyStrategy,
    RectificationStrategy,
)

logger = logging.getLogger(__name__)

_entity_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("entity_id", default="")


def _get_git_hash() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _get_hipp_version() -> str | None:
    try:
        return importlib.metadata.version("hipp")
    except Exception:
        return None


def _validate_tif(path: Path) -> None:
    import rasterio

    try:
        with rasterio.open(path) as ds:
            if ds.width == 0 or ds.height == 0 or ds.count == 0:
                raise ValueError(f"TIF appears empty: {path}")
    except Exception as exc:
        raise ValueError(f"Invalid TIF file {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Base step
# ---------------------------------------------------------------------------


class PipelineStep:
    """A single pipeline step with declared file inputs and outputs.

    Parameters
    ----------
    inputs : list[Path]
        Files that must exist before this step can run.
    outputs : list[Path]
        Files produced by this step. If all exist and *overwrite* is False,
        the step is skipped.
    overwrite : bool
        Force re-execution even when outputs already exist.
    """

    def __init__(
        self,
        inputs: list[Path],
        outputs: list[Path],
        overwrite: bool = False,
        max_retries: int = 0,
        retry_delay: float = 2.0,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.overwrite = overwrite
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._metrics: dict[str, Any] = {}
        self.__result_: StepResult | None = None

    @property
    def result_(self) -> StepResult:
        if self.__result_ is None:
            raise RuntimeError("call run() before result_.")
        return self.__result_

    def is_done(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs)

    def check_inputs(self) -> None:
        missing = [p for p in self.inputs if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{self.__class__.__name__}: missing inputs: {missing}")
        for p in self.inputs:
            if p.suffix == ".tif":
                _validate_tif(p)

    def _validate_outputs(self) -> None:
        for p in self.outputs:
            if p.suffix == ".tif" and p.exists():
                _validate_tif(p)

    def run(self) -> None:
        started_at = datetime.now()
        t0 = time.perf_counter()

        if self.is_done() and not self.overwrite:
            logger.info("%s: already done, skipping", self.__class__.__name__)
            self.__result_ = StepResult(
                name=self.__class__.__name__,
                status="skipped",
                started_at=started_at,
                duration=0.0,
            )
            return

        self.check_inputs()
        for p in self.outputs:
            p.parent.mkdir(parents=True, exist_ok=True)

        try:
            for attempt in range(self.max_retries + 1):
                try:
                    self.execute()
                    break
                except OSError as exc:
                    if attempt == self.max_retries:
                        raise
                    logger.warning(
                        "%s: OSError (%s), retry %d/%d in %.0fs",
                        self.__class__.__name__, exc, attempt + 1, self.max_retries, self.retry_delay,
                    )
                    time.sleep(self.retry_delay)
            self._validate_outputs()
            self.__result_ = StepResult(
                name=self.__class__.__name__,
                status="ran",
                started_at=started_at,
                duration=time.perf_counter() - t0,
                metrics=self._metrics or None,
            )
        except Exception as exc:
            self.__result_ = StepResult(
                name=self.__class__.__name__,
                status="failed",
                started_at=started_at,
                duration=time.perf_counter() - t0,
                error=str(exc),
            )
            raise

    def execute(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete steps
# ---------------------------------------------------------------------------


class ExtractArchiveStep(PipelineStep):
    """Extract a .tgz archive (or symlink pre-extracted tiles) into a tiles/ directory.

    If all inputs are .tif files they are symlinked into the output directory.
    If the single input is a .tgz archive it is extracted directly into the output directory.

    outputs[0] : tiles/ directory containing the .tif files (real or symlinked).
    """

    _SENTINEL = ".extracted"

    def is_done(self) -> bool:
        return (self.outputs[0] / self._SENTINEL).exists()

    def execute(self) -> None:
        tiles_dir = self.outputs[0]
        tiles_dir.mkdir(parents=True, exist_ok=True)

        if all(p.suffix == ".tif" for p in self.inputs):
            for src in sorted(self.inputs):
                dst = tiles_dir / src.name
                if dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            logger.info("ExtractArchiveStep: %d tiles symlinked → %s", len(self.inputs), tiles_dir)
        else:
            archive = self.inputs[0]
            logger.info("ExtractArchiveStep: extracting %s → %s", archive, tiles_dir)
            all_extracted = extract_archive(archive, tiles_dir)
            paths = [p for p in all_extracted if p.suffix == ".tif"]
            if not paths:
                raise RuntimeError(f"No .tif files found after extracting {archive}")
            logger.info("ExtractArchiveStep: %d tiles extracted → %s", len(paths), tiles_dir)

        (tiles_dir / self._SENTINEL).touch()


class ComputeAlignmentsStep(PipelineStep):
    """Compute sequential ORB+RANSAC alignments between image tiles.

    inputs      : ordered list of .tif tile paths
    outputs[0]  : alignments.joblib
    """

    def execute(self) -> None:
        alignments = compute_sequential_alignments([str(t) for t in self.inputs])
        joblib.dump(alignments, self.outputs[0])
        logger.info("ComputeAlignmentsStep: %d alignments → %s", len(alignments), self.outputs[0])


class BuildMosaicStep(PipelineStep):
    """Composite aligned tiles into a single mosaic GeoTIFF.

    inputs[0]  : alignments.joblib
    outputs[0] : mosaic.tif
    """

    def execute(self) -> None:
        alignments: list[ImageAlignment] = joblib.load(self.inputs[0])
        write_mosaic(alignments, str(self.outputs[0]))
        logger.info("BuildMosaicStep: mosaic written to %s", self.outputs[0])


class JoinImagesStep(PipelineStep):
    """Align tiles and build a mosaic — combines alignment and compositing.

    inputs[0]  : tiles/ directory produced by ExtractArchiveStep
    outputs[0] : mosaic.tif

    ``alignments.joblib`` is written to ``outputs[0].parent`` and reused on
    re-runs (sub-steps are skip-if-done individually).
    """

    def execute(self) -> None:
        tiles_dir = self.inputs[0]
        tiles = sorted(tiles_dir.glob("*.tif"))
        if not tiles:
            raise RuntimeError(f"No .tif tiles found in {tiles_dir}")

        alignments_path = self.outputs[0].parent / "alignments.joblib"

        ComputeAlignmentsStep(
            inputs=tiles,
            outputs=[alignments_path],
            overwrite=self.overwrite,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        ).run()

        BuildMosaicStep(
            inputs=[alignments_path],
            outputs=self.outputs,
            overwrite=self.overwrite,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        ).run()


class QuickviewStep(PipelineStep):
    """Generate a downsampled JPEG preview of a raster.

    inputs[0]  : source .tif
    outputs[0] : destination .jpg
    """

    def execute(self) -> None:
        generate_quickview(self.inputs[0], self.outputs[0], scale_factor=0.05)
        logger.info("QuickviewStep: %s → %s", self.inputs[0].name, self.outputs[0])


class DetectVerticalEdgesStep(PipelineStep):
    """Detect left/right vertical frame edges.

    inputs[0]  : mosaic.tif
    outputs[0] : vertical.joblib

    QC figures are saved as PNGs to ``outputs[0].parent / "vertical_qc/"``.
    """

    def execute(self) -> None:
        from hipp.kh9pc.restitution.plotters import vertical_figures

        detector = VerticalDetector().fit(self.inputs[0])
        joblib.dump(detector, self.outputs[0])
        _save_qc_figures(vertical_figures(detector), self.outputs[0].parent / "vertical_qc")
        self._metrics["left_position"] = detector.left.position
        self._metrics["right_position"] = detector.right.position
        logger.info(
            "DetectVerticalEdgesStep: left=%d right=%d",
            detector.left.position,
            detector.right.position,
        )


class DetectHorizontalEdgesStep(PipelineStep):
    """Detect top/bottom film edges using a cascade of strategies.

    Tries CollimationStrategy → PolyStrategy → FlatStrategy in order.

    inputs[0]  : mosaic.tif
    inputs[1]  : vertical.joblib
    outputs[0] : horizontal.joblib (winning strategy)
    outputs[1] : horizontal_attempts.joblib (all StrategyAttempt, including failures)

    QC figures are saved as PNGs to ``outputs[0].parent / "horizontal_qc/"``.
    """

    def execute(self) -> None:
        from hipp.kh9pc.restitution.plotters import plot_strategy_header, plot_strategy_params, strategy_figures

        vertical: VerticalDetector = joblib.load(self.inputs[1])
        edges = vertical.edges

        winner: RectificationStrategy | None = None
        min_inlier_ratio = 0.5
        attempts: list[StrategyAttempt] = []

        for StrategyClass in [CollimationStrategy, PolyStrategy, FlatStrategy]:
            label = StrategyClass.__name__
            try:
                logger.info("DetectHorizontalEdgesStep: trying %s", label)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    strategy = StrategyClass(vertical_edges=edges).fit(self.inputs[0])

                if hasattr(strategy, "top") and hasattr(strategy.top, "inlier_ratio"):
                    ratio = min(strategy.top.inlier_ratio, strategy.bottom.inlier_ratio)
                    if ratio < min_inlier_ratio:
                        reason = f"inlier ratio too low ({ratio:.1%})"
                        logger.warning("DetectHorizontalEdgesStep: %s — %s, trying next", label, reason)
                        attempts.append(StrategyAttempt(strategy=strategy, success=False, failure_reason=reason))
                        continue

                attempts.append(StrategyAttempt(strategy=strategy, success=True, failure_reason=None))
                winner = strategy
                break

            except Exception as exc:
                logger.warning("DetectHorizontalEdgesStep: %s failed (%s), trying next", label, exc)
                attempts.append(StrategyAttempt(strategy=None, success=False, failure_reason=str(exc)))  # type: ignore[arg-type]

        if winner is None:
            raise RuntimeError("All horizontal edge detection strategies failed.")

        joblib.dump(winner, self.outputs[0])
        joblib.dump(attempts, self.outputs[1])

        self._metrics["strategy_winner"] = winner.__class__.__name__
        self._metrics["n_attempts"] = len(attempts)
        top_inlier = getattr(getattr(winner, "top", None), "inlier_ratio", None)
        bottom_inlier = getattr(getattr(winner, "bottom", None), "inlier_ratio", None)
        if top_inlier is not None and bottom_inlier is not None:
            self._metrics["top_inlier_ratio"] = round(float(top_inlier), 4)
            self._metrics["bottom_inlier_ratio"] = round(float(bottom_inlier), 4)

        figures = []
        for attempt in attempts:
            figures.append(plot_strategy_header(attempt))
            if attempt.strategy is not None:
                figures.append(plot_strategy_params(attempt.strategy))
                figures.extend(strategy_figures(attempt.strategy))
        _save_qc_figures(figures, self.outputs[0].parent / "horizontal_qc")

        logger.info(
            "DetectHorizontalEdgesStep: winner=%s, %d attempt(s) → %s",
            winner.__class__.__name__,
            len(attempts),
            self.outputs[0],
        )


class ApplyRestitutionStep(PipelineStep):
    """Apply the geometric restitution transform and write the final image.

    inputs[0]  : mosaic.tif
    inputs[1]  : horizontal.joblib
    outputs[0] : rectified .tif

    Parameters
    ----------
    output_size : OutputSize
        Strategy controlling the canvas size of the rectified image.
    """

    def __init__(
        self,
        inputs: list[Path],
        outputs: list[Path],
        output_size: OutputSize,
        overwrite: bool = False,
        max_retries: int = 0,
        retry_delay: float = 2.0,
    ) -> None:
        super().__init__(inputs, outputs, overwrite, max_retries, retry_delay)
        self.output_size = output_size

    def execute(self) -> None:
        strategy: RectificationStrategy = joblib.load(self.inputs[1])
        strategy.transform(self.inputs[0], self.outputs[0], self.output_size)
        logger.info("ApplyRestitutionStep: written to %s", self.outputs[0])


class RestitutionStep(PipelineStep):
    """Detect vertical/horizontal edges and apply restitution — combines the three restitution steps.

    inputs[0]  : mosaic.tif
    outputs[0] : rectified .tif

    ``vertical.joblib``, ``horizontal.joblib``, and ``horizontal_attempts.joblib``
    are written to ``inputs[0].parent`` and reused on re-runs (sub-steps are
    skip-if-done individually).

    Parameters
    ----------
    output_size : OutputSize
        Canvas-sizing strategy for the rectified image.
    """

    def __init__(
        self,
        inputs: list[Path],
        outputs: list[Path],
        output_size: OutputSize,
        overwrite: bool = False,
        max_retries: int = 0,
        retry_delay: float = 2.0,
    ) -> None:
        super().__init__(inputs, outputs, overwrite, max_retries, retry_delay)
        self.output_size = output_size

    @property
    def vertical_qc_dir(self) -> Path:
        return self.inputs[0].parent / "vertical_qc"

    @property
    def horizontal_qc_dir(self) -> Path:
        return self.inputs[0].parent / "horizontal_qc"

    def execute(self) -> None:
        work_dir = self.inputs[0].parent

        DetectVerticalEdgesStep(
            inputs=[self.inputs[0]],
            outputs=[work_dir / "vertical.joblib"],
            overwrite=self.overwrite,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        ).run()

        DetectHorizontalEdgesStep(
            inputs=[self.inputs[0], work_dir / "vertical.joblib"],
            outputs=[work_dir / "horizontal.joblib", work_dir / "horizontal_attempts.joblib"],
            overwrite=self.overwrite,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        ).run()

        ApplyRestitutionStep(
            inputs=[self.inputs[0], work_dir / "horizontal.joblib"],
            outputs=self.outputs,
            output_size=self.output_size,
            overwrite=self.overwrite,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        ).run()


class GenerateQCReportStep(PipelineStep):
    """Assemble QC figures into a PDF report.

    inputs[0] : summary.json — pipeline step results written by KH9Pipeline
    inputs[1] : vertical_qc/ — directory of PNG figures from DetectVerticalEdgesStep
    inputs[2] : horizontal_qc/ — directory of PNG figures from DetectHorizontalEdgesStep
    outputs[0] : report PDF
    """

    def execute(self) -> None:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        from hipp.kh9pc.restitution.plotters import plot_pipeline_summary
        from hipp.kh9pc.utils import generate_qc_report

        raw: Any = json.loads(self.inputs[0].read_text())
        if isinstance(raw, dict) and "steps" in raw:
            step_results: list[dict[str, Any]] = raw["steps"]
            meta: dict[str, Any] | None = raw.get("meta")
        else:
            step_results = raw
            meta = None
        figures = [plot_pipeline_summary(step_results, meta=meta)]

        for qc_dir in self.inputs[1:]:
            for png in sorted(qc_dir.glob("*.png")):
                img = mpimg.imread(str(png))
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout(pad=0)
                figures.append(fig)

        generate_qc_report(self.outputs[0], figures)
        logger.info("GenerateQCReportStep: %s", self.outputs[0])


class CleanupWorkDirStep(PipelineStep):
    """Delete the per-scene work directory.

    Parameters
    ----------
    work_dir : Path
        Directory to remove (recursively).
    """

    def __init__(self, work_dir: Path) -> None:
        super().__init__(inputs=[], outputs=[])
        self.work_dir = work_dir

    def run(self) -> None:
        self.execute()

    def execute(self) -> None:
        import shutil

        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
            logger.info("CleanupWorkDirStep: removed %s", self.work_dir)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PipelineConfig:
    """Runtime options for :class:`KH9Pipeline`.

    Parameters
    ----------
    overwrite : bool
        Re-run a step even when its output already exists on disk.
    output_size : OutputSize | None
        Strategy controlling the canvas size of the rectified image.
        Defaults to :class:`FixedHeightSize` with ``height=22064``.
    steps : list[str] | None
        Subset of step names to execute. ``None`` runs every step in order.
    """

    def __init__(
        self,
        overwrite: bool = False,
        output_size: OutputSize | None = None,
        steps: list[str] | None = None,
        cleanup: bool = False,
        dry_run: bool = False,
        max_retries: int = 0,
        retry_delay: float = 2.0,
    ) -> None:
        self.overwrite = overwrite
        self.output_size: OutputSize = output_size or FixedHeightSize(height=22064)
        self.steps = steps
        self.cleanup = cleanup
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @classmethod
    def from_toml(cls, path: Path) -> "PipelineConfig":
        """Load a :class:`PipelineConfig` from a TOML file.

        CLI flags take precedence — callers should override individual attributes
        after construction when command-line arguments are provided.

        Expected TOML keys (all optional):

        .. code-block:: toml

            overwrite = false
            cleanup = false
            steps = ["extract", "join_images", "quickview_mosaic", "restitution", "quickview_final", "qc_report"]

            [output_size]
            type = "fixed_height"   # auto | fixed_height | fixed_size | same_size | margin
            height = 22064
        """
        import tomllib

        with path.open("rb") as f:
            raw: dict[str, Any] = tomllib.load(f)
        return cls(
            overwrite=raw.get("overwrite", False),
            output_size=_parse_output_size(raw.get("output_size")),
            steps=raw.get("steps"),
            cleanup=raw.get("cleanup", False),
            dry_run=raw.get("dry_run", False),
            max_retries=raw.get("max_retries", 0),
            retry_delay=raw.get("retry_delay", 2.0),
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class KH9Pipeline:
    """Orchestrates the KH-9 PC preprocessing pipeline for a single scene.

    Each step is an independent :class:`PipelineStep` that reads its inputs
    from disk and writes its outputs to disk.  The pipeline wires them together
    by mapping output paths of one step to input paths of the next.

    Directory layout
    ----------------
    ::

        <output>                              ← final rectified image (exact path)

        qc_dir/
            mosaic_qv/{entity_id}.jpg
            final_qv/{entity_id}.jpg
            report/{entity_id}.pdf

        work_dir/                             ← intermediate files, cleaned at step cleanup
            {entity_id}/
                files.json
                alignments.joblib
                mosaic.tif
                vertical.joblib
                horizontal.joblib

    Parameters
    ----------
    input : Path | list[Path]
        Either a ``.tgz`` archive or an explicit ordered list of ``.tif`` tiles.
    output : Path
        Exact path for the final rectified ``.tif``. ``entity_id`` is derived from its stem.
    qc_dir : Path
        Root directory for QC outputs.
    work_dir : Path | None
        Root directory for intermediate files. Defaults to ``output.parent / "_work"``.
    config : PipelineConfig | None
        Runtime options. Defaults to :class:`PipelineConfig` with all defaults.

    Examples
    --------
    Run the full pipeline::

        pipeline = KH9Pipeline(
            input=Path("DZB1215-500587L002001.tgz"),
            output=Path("outputs/images/DZB1215-500587L002001.tif"),
            qc_dir=Path("outputs/qc/"),
        )
        pipeline.run()

    Re-run only the restitution steps with a custom input mosaic::

        step = ApplyRestitutionStep(
            inputs=[Path("my_mosaic.tif"), Path("my_horizontal.joblib")],
            outputs=[Path("my_rectified.tif")],
            output_size=FixedHeightSize(height=22064),
        )
        step.run()
    """

    def __init__(
        self,
        input: Path | list[Path],
        output: Path,
        qc_dir: Path,
        work_dir: Path | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.input = input
        self.output = Path(output)
        self.entity_id = self.output.stem
        self.qc_dir = Path(qc_dir)
        self._work_base = Path(work_dir) if work_dir is not None else self.output.parent / "_work"
        self.config = config or PipelineConfig()
        self.results_: list[StepResult] = []

        _entity_id_ctx.set(self.entity_id)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _work_dir(self) -> Path:
        return self._work_base / self.entity_id

    def _qc(self, subdir: str, ext: str) -> Path:
        return self.qc_dir / subdir / f"{self.entity_id}.{ext}"

    def _tmp(self, filename: str) -> Path:
        return self._work_dir / filename

    def _build_steps(self) -> dict[str, PipelineStep]:
        ow = self.config.overwrite
        mr = self.config.max_retries
        rd = self.config.retry_delay

        extract_inputs: list[Path] = (
            [Path(p) for p in self.input] if isinstance(self.input, list) else [Path(self.input)]
        )

        restitution = RestitutionStep(
            inputs=[self._tmp("mosaic.tif")],
            outputs=[self.output],
            output_size=self.config.output_size,
            overwrite=ow,
            max_retries=mr,
            retry_delay=rd,
        )

        return {
            "extract": ExtractArchiveStep(
                inputs=extract_inputs,
                outputs=[self._tmp("tiles")],
                overwrite=ow,
                max_retries=mr,
                retry_delay=rd,
            ),
            "join_images": JoinImagesStep(
                inputs=[self._tmp("tiles")],
                outputs=[self._tmp("mosaic.tif")],
                overwrite=ow,
                max_retries=mr,
                retry_delay=rd,
            ),
            "quickview_mosaic": QuickviewStep(
                inputs=[self._tmp("mosaic.tif")],
                outputs=[self._qc("mosaic_qv", "jpg")],
                overwrite=ow,
                max_retries=mr,
                retry_delay=rd,
            ),
            "restitution": restitution,
            "quickview_final": QuickviewStep(
                inputs=[self.output],
                outputs=[self._qc("final_qv", "jpg")],
                overwrite=ow,
                max_retries=mr,
                retry_delay=rd,
            ),
            "qc_report": GenerateQCReportStep(
                inputs=[
                    self._tmp("summary.json"),
                    restitution.vertical_qc_dir,
                    restitution.horizontal_qc_dir,
                ],
                outputs=[self._qc("report", "pdf")],
                overwrite=ow,
                max_retries=mr,
                retry_delay=rd,
            ),
        }

    def _write_summary(self) -> None:
        data: dict[str, Any] = {
            "meta": {
                "entity_id": self.entity_id,
                "git_hash": _get_git_hash(),
                "hipp_version": _get_hipp_version(),
            },
            "steps": [
                {
                    "name": r.name,
                    "status": r.status,
                    "started_at": r.started_at.strftime("%H:%M:%S"),
                    "duration": r.duration,
                    "error": r.error,
                    "metrics": r.metrics,
                }
                for r in self.results_
            ],
        }
        self._tmp("summary.json").write_text(json.dumps(data, indent=2))

    def run(self) -> None:
        """Execute all pipeline steps (or the subset defined in ``config.steps``)."""
        steps = self._build_steps()
        names = self.config.steps or list(steps.keys())

        for name in names:
            if name not in steps:
                raise ValueError(f"Unknown step '{name}'. Valid steps: {list(steps.keys())}")

        if self.config.dry_run:
            for name in names:
                step = steps[name]
                outcome = "skip (already done)" if (step.is_done() and not self.config.overwrite) else "would execute"
                logger.info("[%s] [DRY-RUN] %-30s → %s", self.entity_id, name, outcome)
            return

        for name in names:
            logger.info("[%s] Running %s", self.entity_id, name)
            steps[name].run()
            self.results_.append(steps[name].result_)
            self._write_summary()

        if self.config.cleanup:
            CleanupWorkDirStep(work_dir=self._work_dir).run()


def _parse_output_size(cfg: dict[str, Any] | None) -> OutputSize | None:
    if cfg is None:
        return None
    type_ = cfg.get("type", "fixed_height")
    if type_ == "auto":
        return AutoSize()
    if type_ == "fixed_height":
        return FixedHeightSize(height=int(cfg["height"]))
    if type_ == "fixed_size":
        return FixedSize(width=int(cfg["width"]), height=int(cfg["height"]))
    if type_ == "same_size":
        return SameSize(width=int(cfg["width"]), height=int(cfg["height"]))
    if type_ == "margin":
        return MarginSize(
            top=int(cfg.get("top", 0)),
            right=int(cfg.get("right", 0)),
            bottom=int(cfg.get("bottom", 0)),
            left=int(cfg.get("left", 0)),
        )
    raise ValueError(f"Unknown output_size type: {type_!r}. Valid: auto, fixed_height, fixed_size, same_size, margin")


def _save_qc_figures(figures: list[Any], directory: Path) -> None:
    """Save a list of matplotlib figures as numbered PNGs and close them."""
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for i, fig in enumerate(figures):
        fig.savefig(directory / f"{i:02d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
