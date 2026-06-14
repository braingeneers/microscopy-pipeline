"""Operation modules.

Each submodule exposes:

* a pure-array core function (e.g. ``crop``) that operates on numpy arrays;
* file/folder convenience wrappers (``crop_file`` / ``crop_folder``);
* a ``cli()`` callable used by the corresponding script in
  ``preprocessing/`` / ``processing/`` / ``postprocessing/``.

Importing from this package never executes a CLI, so it is safe to compose
operations programmatically.
"""

from .crop import crop, crop_stack, crop_file, crop_folder
from .mask import mask, mask_file, mask_folder, parse_color
from .remove_lone_pixels import (
    find_lone_pixels,
    remove_lone_pixels,
    remove_lone_pixels_batch,
    remove_lone_pixels_folder,
)
from .scale_colors import scale_colors, scale_colors_file, scale_colors_folder
from .pngs_to_tiff import pngs_to_tiff_stacks, group_pngs_by_x
from .extract_times import extract_times, parse_time_offset
from .align import align_pair, align_stack, align_session
from .complex_edf import (
    complex_edf,
    complex_edf_image,
    complex_edf_file,
    complex_edf_folder,
)
from .fuse_tiffs import fuse_project, fuse_average, fuse_average_file, fuse_average_folder
from .clahe import clahe, clahe_file, clahe_folder
from .grey_to_color import grey_to_color, grey_to_color_file, grey_to_color_folder
from .scale_bar import add_scale_bar, add_scale_bar_file, add_scale_bar_folder
from .tiff_to_gif import tiff_to_gif, frames_to_gif, frames_to_video
from .scale_brightness import (
    scale_brightness,
    scale_brightness_file,
    scale_brightness_folder,
)
from .find_max_brightness import find_max_brightness, find_max_brightness_file, MaxBrightness
from .sum_fluorescence import sum_brightness, sum_fluorescence_folder
from .plot_summed_fluorescence import plot_fluorescence
from .identify_core import identify_core, identify_core_file, identify_core_folder
from .graph_contour_data import graph_contour_data
from .brightfield_organoid_tracker import (
    find_organoid_contour,
    track_organoid,
    track_organoid_file,
    track_organoid_folder,
)

__all__ = [
    # geometry / pixels
    "crop", "crop_stack", "crop_file", "crop_folder",
    "mask", "mask_file", "mask_folder", "parse_color",
    "find_lone_pixels", "remove_lone_pixels", "remove_lone_pixels_batch",
    "remove_lone_pixels_folder",
    # brightness / colour
    "scale_colors", "scale_colors_file", "scale_colors_folder",
    "scale_brightness", "scale_brightness_file", "scale_brightness_folder",
    "find_max_brightness", "find_max_brightness_file", "MaxBrightness",
    "clahe", "clahe_file", "clahe_folder",
    "grey_to_color", "grey_to_color_file", "grey_to_color_folder",
    # alignment
    "align_pair", "align_stack", "align_session",
    # stack ops
    "pngs_to_tiff_stacks", "group_pngs_by_x",
    "complex_edf", "complex_edf_image", "complex_edf_file", "complex_edf_folder",
    "fuse_project", "fuse_average", "fuse_average_file", "fuse_average_folder",
    "tiff_to_gif", "frames_to_gif", "frames_to_video",
    # annotation
    "add_scale_bar", "add_scale_bar_file", "add_scale_bar_folder",
    # analysis
    "extract_times", "parse_time_offset",
    "sum_brightness", "sum_fluorescence_folder",
    "plot_fluorescence",
    "identify_core", "identify_core_file", "identify_core_folder",
    "find_organoid_contour", "track_organoid",
    "track_organoid_file", "track_organoid_folder",
    "graph_contour_data",
]
