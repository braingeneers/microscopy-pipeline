"""Unified microscopy image-processing pipeline.

All command-line scripts in this repository are thin wrappers around the
operations exposed here.  Each operation is available in three forms:

* ``op(array, **params) -> array`` -- pure numpy core, composable for piping
  multiple operations together without touching the disk.
* ``op_file(input_path, output_path, **params)`` -- file in / file out.
* ``op_folder(input_dir, output_dir, **params)`` -- batch over a folder.

Example: chain operations on a single in-memory stack::

    from microscopy_pipeline import io, ops

    stack = io.load_stack("raw.tif")
    fused = ops.complex_edf(stack, wavelet="db3", levels=3)
    fused = ops.clahe(fused, clip_limit=2.0)
    fused = ops.add_scale_bar(fused, scale_um=100, pixel_size_um=0.5)
    io.save_image(fused, "out.tif")

CLI conventions
---------------

Every ``op`` script accepts the same core flags:

* ``-i`` / ``--input``  : input file or directory (auto-detected).
* ``-o`` / ``--output`` : output file or directory.

Op-specific parameters use ``--kebab-case`` flag names matching the keyword
argument of the array-level function (e.g. ``--clip-limit`` ``clip_limit``).
"""

from . import io, ops, workflows  # noqa: F401

# Re-export every operation directly under the package namespace so users can
# write ``from microscopy_pipeline import crop, clahe`` etc.
from .ops import *  # noqa: F401,F403
from .ops import __all__ as _ops_all

__all__ = ["io", "ops", "workflows", *list(_ops_all)]
