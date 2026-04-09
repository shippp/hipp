from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from hipp.kh9pc.restitution.base import RectificationStrategy
from hipp.kh9pc.restitution.collimation_rectification_strategy import CollimationRectificationStrategy
from hipp.kh9pc.restitution.flat_rectification_strategy import FlatRectificationStrategy
from hipp.kh9pc.restitution.output_size import FixedHeightSize
from hipp.kh9pc.restitution.poly_rectification_strategy import PolyRectificationStrategy
from hipp.kh9pc.restitution.transformer import ImageTransformerAffine, ImageTransformerTps
from hipp.kh9pc.restitution.vertical_edges_estimator import VerticalEdgesEstimator
from hipp.kh9pc.utils import generate_qc_report


class Pipeline(ABC):
    """Abstract base class for rectification pipelines."""

    @abstractmethod
    def run(self, input_file: str | Path, output_file: str | Path, qc_output: str | Path | None = None) -> None:
        """Run the pipeline on a single image.

        Parameters
        ----------
        input_file:
            Path to the input raster (GeoTIFF).
        output_file:
            Destination path for the rectified raster.
        qc_output:
            Optional path for the PDF quality-control report.
            If ``None``, no report is generated.
        """
        ...


@dataclass
class AdaptivePipeline(Pipeline):
    """Rectification pipeline with automatic strategy fallback.

    Strategies are tried in order of decreasing complexity:

    1. ``CollimationRectificationStrategy`` — fits polynomial models on the
       physical collimation lines; most accurate when the lines are visible.
    2. ``PolyRectificationStrategy`` — fits polynomial models on the image
       content edges; used when collimation lines are not detectable.
    3. ``FlatRectificationStrategy`` — simple flat crop with no distortion
       correction; last resort when both polynomial fits fail.

    A strategy is accepted when both its top and bottom RANSAC inlier ratios
    are at or above ``min_inlier_ratio``.  The output height is always fixed to
    ``collimation_line_dist + margin_top + margin_bottom`` regardless of the
    strategy used, so downstream SfM tools always receive a canonical image size.

    Parameters
    ----------
    min_inlier_ratio:
        Minimum RANSAC inlier ratio required to accept a strategy.
    collimation_line_dist:
        Expected pixel distance between the two collimation lines.
        Also used to compute the fixed output height.
    margin_top:
        Pixels added above the content region in the output raster.
    margin_bottom:
        Pixels added below the content region in the output raster.
    verbose:
        Print progress and diagnostic messages to stdout.
    dry_run:
        If ``True``, run all fitting steps but skip the image transform.
        Useful for inspecting strategy selection and QC without writing output.
    """

    min_inlier_ratio: float = 0.5
    collimation_line_dist: int = 21770
    margin_top: int = 147
    margin_bottom: int = 147
    verbose: bool = False
    dry_run: bool = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _is_reliable(self, strat: CollimationRectificationStrategy | PolyRectificationStrategy) -> bool:
        """Return True if both top and bottom inlier ratios meet the threshold."""
        return strat.top.inlier_ratio >= self.min_inlier_ratio and strat.bottom.inlier_ratio >= self.min_inlier_ratio

    def run(self, input_file: str | Path, output_file: str | Path, qc_output: str | Path | None = None) -> None:
        """Run the adaptive pipeline on a single image."""
        figures = []
        # Fixed output height regardless of which strategy is selected
        target_height = self.collimation_line_dist + self.margin_top + self.margin_bottom

        self._log(f"[pipeline] input  : {input_file}")
        self._log(f"[pipeline] output : {output_file}")
        self._log(f"[pipeline] target height : {target_height} px  (dry_run={self.dry_run})")

        try:
            strategy: RectificationStrategy

            # Step 1 — detect left/right content boundaries
            self._log("[1/4] fitting vertical edges...")
            vest = VerticalEdgesEstimator().fit(input_file)
            self._log(f"      edges: left={vest.edges[0]} px, right={vest.edges[1]} px  ({vest.fitting_time_:.1f}s)")
            if qc_output is not None:
                figures.extend(vest.get_qc_figures())

            # Step 2 — try collimation-based strategy (most accurate)
            self._log("[2/4] fitting collimation rectification strategy...")
            c_strat = CollimationRectificationStrategy(
                vest.edges, collimation_line_dist=self.collimation_line_dist
            ).fit(input_file)
            self._log(
                f"      top inlier ratio: {c_strat.top.inlier_ratio:.1%}  bottom: {c_strat.bottom.inlier_ratio:.1%}  ({c_strat.fitting_time_:.1f}s)"
            )
            if qc_output is not None:
                figures.extend(c_strat.get_qc_figures())

            if self._is_reliable(c_strat):
                self._log("      -> using CollimationRectificationStrategy")
                strategy = c_strat
            else:
                # Step 3 — fall back to polynomial edge detection
                self._log(f"      -> inlier ratio below {self.min_inlier_ratio:.0%}, falling back to poly...")
                self._log("[3/4] fitting poly rectification strategy...")
                p_strat = PolyRectificationStrategy(vest.edges).fit(input_file)
                self._log(
                    f"      top inlier ratio: {p_strat.top.inlier_ratio:.1%}  bottom: {p_strat.bottom.inlier_ratio:.1%}  ({p_strat.fitting_time_:.1f}s)"
                )
                if qc_output is not None:
                    figures.extend(p_strat.get_qc_figures())

                if self._is_reliable(p_strat):
                    self._log("      -> using PolyRectificationStrategy")
                    strategy = p_strat
                else:
                    # Step 4 — last resort: flat crop, no distortion correction
                    self._log(f"      -> inlier ratio below {self.min_inlier_ratio:.0%}, falling back to flat...")
                    f_strat = FlatRectificationStrategy(vest.edges).fit(input_file)
                    self._log(
                        f"      top={f_strat.top.position} px  bottom={f_strat.bottom.position} px  ({f_strat.fitting_time_:.1f}s)"
                    )
                    self._log("      -> using FlatRectificationStrategy")
                    strategy = f_strat
                    if qc_output is not None:
                        figures.extend(strategy.get_qc_figures())

            # Build TPS control-point grids and enforce the fixed output height
            src, dst, detected = strategy.compute_grid()
            src, dst, final = FixedHeightSize(target_height).apply(src, dst, detected)
            self._log(f"[4/4] output size: {final[0]} x {final[1]} px")

            # FlatRectificationStrategy produces only 4 corner points → affine is sufficient and TPS would be ill-conditioned
            transformer = (
                ImageTransformerAffine() if isinstance(strategy, FlatRectificationStrategy) else ImageTransformerTps()
            )
            self._log(f"      transformer: {transformer.__class__.__name__}")

            if self.dry_run:
                self._log("      -> dry_run=True, skipping transform")
            else:
                transformer.fit(src, dst, final)
                transformer.transform(input_file, output_file)
                self._log("      -> done")

        finally:
            if qc_output is not None:
                generate_qc_report(qc_output, figures)
                self._log(f"[qc]  report written to {qc_output}")
