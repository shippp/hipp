from matplotlib.figure import Figure

from hipp.kh9pc.qc.collimation import plot_collimation_distortions, plot_collimation_edges
from hipp.kh9pc.qc.fiducial import (
    mean_patch_from_centers,
    plot_fiducial_detected_boxes,
    plot_fiducial_detected_profiles,
    plot_fiducial_distortions,
    plot_fiducial_filtering,
)
from hipp.kh9pc.qc.flat import plot_flat_edges, plot_flat_ruptures
from hipp.kh9pc.qc.poly import plot_poly_distortions, plot_poly_edges
from hipp.kh9pc.qc.transform import plot_crop_area, plot_deformation_grid
from hipp.kh9pc.qc.vertical import plot_vertical_edges, plot_vertical_ruptures
from hipp.kh9pc.restitution.base import FittingClass
from hipp.kh9pc.restitution.collimation import CollimationStrategy
from hipp.kh9pc.restitution.fiducial import FiducialStrategy
from hipp.kh9pc.restitution.flat import FlatStrategy
from hipp.kh9pc.restitution.mixed import MixedStrategy
from hipp.kh9pc.restitution.poly import PolyStrategy
from hipp.kh9pc.restitution.vertical import VerticalDetector

__all__ = [
    "get_figures",
    "mean_patch_from_centers",
    "plot_collimation_distortions",
    "plot_collimation_edges",
    "plot_crop_area",
    "plot_deformation_grid",
    "plot_fiducial_detected_boxes",
    "plot_fiducial_detected_profiles",
    "plot_fiducial_distortions",
    "plot_fiducial_filtering",
    "plot_flat_edges",
    "plot_flat_ruptures",
    "plot_poly_distortions",
    "plot_poly_edges",
    "plot_vertical_edges",
    "plot_vertical_ruptures",
]


def get_figures(fitting_class: FittingClass, plot_transformation: bool = True) -> list[Figure]:
    """Return all QC figures for a fitted FittingClass instance."""
    if isinstance(fitting_class, VerticalDetector):
        return [plot_vertical_edges(fitting_class), plot_vertical_ruptures(fitting_class)]
    if isinstance(fitting_class, FlatStrategy):
        return [
            *get_figures(fitting_class.vertical_detector, plot_transformation=False),
            plot_flat_edges(fitting_class),
            plot_flat_ruptures(fitting_class),
            *([plot_crop_area(fitting_class.transformation_)] if plot_transformation else []),
        ]
    if isinstance(fitting_class, PolyStrategy):
        return [
            *get_figures(fitting_class.vertical_detector, plot_transformation=False),
            plot_poly_edges(fitting_class),
            plot_poly_distortions(fitting_class),
            *(
                [plot_deformation_grid(fitting_class.transformation_), plot_crop_area(fitting_class.transformation_)]
                if plot_transformation
                else []
            ),
        ]
    if isinstance(fitting_class, CollimationStrategy):
        return [
            *get_figures(fitting_class.poly_strategy, plot_transformation=False),
            plot_collimation_edges(fitting_class),
            plot_collimation_distortions(fitting_class),
            *(
                [plot_deformation_grid(fitting_class.transformation_), plot_crop_area(fitting_class.transformation_)]
                if plot_transformation
                else []
            ),
        ]
    if isinstance(fitting_class, FiducialStrategy):
        return [
            *get_figures(fitting_class.poly_strategy, plot_transformation=False),
            plot_fiducial_filtering(fitting_class),
            plot_fiducial_distortions(fitting_class),
            plot_fiducial_detected_profiles(fitting_class),
            *plot_fiducial_detected_boxes(fitting_class),
            *(
                [plot_deformation_grid(fitting_class.transformation_), plot_crop_area(fitting_class.transformation_)]
                if plot_transformation
                else []
            ),
        ]
    if isinstance(fitting_class, MixedStrategy):
        return get_figures(fitting_class.selected_strategy_, plot_transformation=plot_transformation)

    return []
