"""
Copyright (c) 2025 HIPP developers
Description: End-to-end preprocessing pipeline for KH-9 Panoramic Camera imagery.
"""

import json
import logging
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

    def __init__(self, inputs: list[Path], outputs: list[Path], overwrite: bool = False) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.overwrite = overwrite
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
            self.execute()
            self.__result_ = StepResult(
                name=self.__class__.__name__,
                status="ran",
                started_at=started_at,
                duration=time.perf_counter() - t0,
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
    """Extract a .tgz archive (or register pre-extracted tiles) and write files.json.

    If all inputs are .tif files they are treated as pre-extracted tiles.
    If the single input is a .tgz archive it is extracted to a tiles/ subdirectory.

    outputs[0] : files.json — JSON array of tile paths.
    """

    def execute(self) -> None:
        output = self.outputs[0]

        if all(p.suffix == ".tif" for p in self.inputs):
            paths = sorted(self.inputs)
            _save_json([str(p) for p in paths], output)
            logger.info("ExtractArchiveStep: %d tiles registered → %s", len(paths), output)
            return

        archive = self.inputs[0]
        tiles_dir = output.parent / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ExtractArchiveStep: extracting %s → %s", archive, tiles_dir)
        all_extracted = extract_archive(archive, tiles_dir)
        paths = sorted(p for p in all_extracted if p.suffix == ".tif")
        if not paths:
            raise RuntimeError(f"No .tif files found after extracting {archive}")
        _save_json([str(p) for p in paths], output)
        logger.info("ExtractArchiveStep: %d tiles extracted → %s", len(paths), output)


class ComputeAlignmentsStep(PipelineStep):
    """Compute sequential ORB+RANSAC alignments between image tiles.

    inputs[0]  : files.json
    outputs[0] : alignments.joblib
    """

    def execute(self) -> None:
        tiles = [Path(p) for p in _load_json(self.inputs[0])]  # type: ignore[attr-defined]
        alignments = compute_sequential_alignments([str(t) for t in tiles])
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
    """

    def execute(self) -> None:
        detector = VerticalDetector().fit(self.inputs[0])
        joblib.dump(detector, self.outputs[0])
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
    """

    def execute(self) -> None:
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

                if hasattr(strategy, "top") and hasattr(strategy, "bottom"):
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
    ) -> None:
        super().__init__(inputs, outputs, overwrite)
        self.output_size = output_size

    def execute(self) -> None:
        strategy: RectificationStrategy = joblib.load(self.inputs[1])
        strategy.transform(self.inputs[0], self.outputs[0], self.output_size)
        logger.info("ApplyRestitutionStep: written to %s", self.outputs[0])


class GenerateQCReportStep(PipelineStep):
    """Generate the full QC PDF report for a single scene.

    Combines a pipeline summary page, vertical edge QC figures, and horizontal
    strategy QC figures (all attempts) into a single PDF.

    inputs[0]    : vertical.joblib
    inputs[1]    : horizontal_attempts.joblib
    outputs[0]   : report PDF
    step_results : pipeline step results used to build the summary page (passed by reference)
    """

    def __init__(
        self,
        inputs: list[Path],
        outputs: list[Path],
        step_results: list[StepResult],
        overwrite: bool = False,
    ) -> None:
        super().__init__(inputs, outputs, overwrite)
        self.step_results = step_results

    def execute(self) -> None:
        from hipp.kh9pc.restitution.plotters import (
            plot_pipeline_summary,
            plot_strategy_header,
            plot_strategy_params,
            strategy_figures,
            vertical_figures,
        )
        from hipp.kh9pc.utils import generate_qc_report

        figures = []

        figures.append(plot_pipeline_summary(self.step_results))

        detector: VerticalDetector = joblib.load(self.inputs[0])
        figures.extend(vertical_figures(detector))

        attempts: list[StrategyAttempt] = joblib.load(self.inputs[1])
        for attempt in attempts:
            figures.append(plot_strategy_header(attempt))
            if attempt.strategy is not None:
                figures.append(plot_strategy_params(attempt.strategy))
                figures.extend(strategy_figures(attempt.strategy))

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
    ) -> None:
        self.overwrite = overwrite
        self.output_size: OutputSize = output_size or FixedHeightSize(height=22064)
        self.steps = steps
        self.cleanup = cleanup

    @classmethod
    def from_toml(cls, path: Path) -> "PipelineConfig":
        """Load a :class:`PipelineConfig` from a TOML file.

        CLI flags take precedence — callers should override individual attributes
        after construction when command-line arguments are provided.

        Expected TOML keys (all optional):

        .. code-block:: toml

            overwrite = false
            cleanup = false
            steps = ["extract", "align", "mosaic", "vertical", "horizontal", "transform", "quickview_final", "qc_report"]

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

    @staticmethod
    def _derive_entity_id(input: Path | list[Path]) -> str:
        if isinstance(input, list):
            return Path(input[0]).stem.rsplit("_", 1)[0]
        return Path(input).stem

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

        extract_inputs: list[Path] = (
            [Path(p) for p in self.input] if isinstance(self.input, list) else [Path(self.input)]
        )

        return {
            "extract": ExtractArchiveStep(
                inputs=extract_inputs,
                outputs=[self._tmp("files.json")],
                overwrite=ow,
            ),
            "align": ComputeAlignmentsStep(
                inputs=[self._tmp("files.json")],
                outputs=[self._tmp("alignments.joblib")],
                overwrite=ow,
            ),
            "mosaic": BuildMosaicStep(
                inputs=[self._tmp("alignments.joblib")],
                outputs=[self._tmp("mosaic.tif")],
                overwrite=ow,
            ),
            "quickview_mosaic": QuickviewStep(
                inputs=[self._tmp("mosaic.tif")],
                outputs=[self._qc("mosaic_qv", "jpg")],
                overwrite=ow,
            ),
            "vertical": DetectVerticalEdgesStep(
                inputs=[self._tmp("mosaic.tif")],
                outputs=[self._tmp("vertical.joblib")],
                overwrite=ow,
            ),
            "horizontal": DetectHorizontalEdgesStep(
                inputs=[self._tmp("mosaic.tif"), self._tmp("vertical.joblib")],
                outputs=[self._tmp("horizontal.joblib"), self._tmp("horizontal_attempts.joblib")],
                overwrite=ow,
            ),
            "transform": ApplyRestitutionStep(
                inputs=[self._tmp("mosaic.tif"), self._tmp("horizontal.joblib")],
                outputs=[self.output],
                output_size=self.config.output_size,
                overwrite=ow,
            ),
            "quickview_final": QuickviewStep(
                inputs=[self.output],
                outputs=[self._qc("final_qv", "jpg")],
                overwrite=ow,
            ),
            "qc_report": GenerateQCReportStep(
                inputs=[self._tmp("vertical.joblib"), self._tmp("horizontal_attempts.joblib")],
                outputs=[self._qc("report", "pdf")],
                step_results=self.results_,
                overwrite=ow,
            ),
            # "cleanup": CleanupWorkDirStep(work_dir=self._work_dir),
        }

    def run(self) -> None:
        """Execute all pipeline steps (or the subset defined in ``config.steps``)."""
        steps = self._build_steps()
        names = self.config.steps or list(steps.keys())
        for name in names:
            if name not in steps:
                raise ValueError(f"Unknown step '{name}'. Valid steps: {list(steps.keys())}")
            logger.info("[%s] Running %s", self.entity_id, name)
            steps[name].run()
            self.results_.append(steps[name].result_)

        if self.config.cleanup:
            CleanupWorkDirStep(work_dir=self._work_dir).run()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _save_json(data: object, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


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
