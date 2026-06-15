"""Convert images / TIFF stacks to OME-TIFF (with physical pixel-size metadata).

OME-TIFF embeds an OME-XML block carrying the dimension order and the physical
pixel size, so downstream tools (Fiji/Bio-Formats, napari, QuPath) read the
calibration directly.  Requires the optional ``tifffile`` package
(``pip install 'microscopy-pipeline[ome]'``).
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Optional

from .. import io
from ..cli import build_io_parser, configure_logging, add_folder_flags

logger = logging.getLogger(__name__)


def _ome_output_path(src: Path, output_dir: Path) -> Path:
    """``foo.tif`` -> ``foo.ome.tif`` (strip an existing .ome to avoid doubling)."""
    stem = src.stem
    if stem.lower().endswith(".ome"):
        stem = stem[:-4]
    return output_dir / f"{stem}.ome.tif"


def to_ome_tiff_file(input_path, output_path, *, pixel_size_um: Optional[float] = None):
    """Convert a single image or a multi-frame TIFF stack to OME-TIFF.

    ``pixel_size_um`` defaults to the source file's metadata (resolution tags or
    OME-XML) when not given.
    """
    if io.is_tiff_stack(input_path):
        frames = io.load_stack(input_path)
    else:
        frames = [io.load_image(input_path)]
    if pixel_size_um is None:
        pixel_size_um = io.read_pixel_size_um(input_path)
    io.save_ome_tiff(frames, output_path, pixel_size_um=pixel_size_um)
    logger.info("wrote %s (%d frame(s), pixel_size_um=%s)",
                output_path, len(frames), pixel_size_um)


def to_ome_tiff_folder(input_dir, output_dir, *, pixel_size_um: Optional[float] = None,
                       jobs=1, skip_existing=False):
    """Convert every image/stack in ``input_dir`` to ``*.ome.tif`` in ``output_dir``."""
    output_dir = io.ensure_dir(output_dir)
    pairs = [(src, _ome_output_path(src, Path(output_dir))) for src in io.list_images(input_dir)]
    io.map_folder(pairs, partial(to_ome_tiff_file, pixel_size_um=pixel_size_um),
                  jobs=jobs, skip_existing=skip_existing, desc="ome-tiff")


def cli(argv=None):
    parser = build_io_parser(
        "Convert images / TIFF stacks to OME-TIFF, embedding physical pixel-size metadata."
    )
    parser.add_argument("--pixel-size-um", type=float, default=None,
                        help="Physical pixel size in micrometres (default: copy from source metadata).")
    add_folder_flags(parser)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    src = Path(args.input)
    if src.is_dir():
        to_ome_tiff_folder(args.input, args.output, pixel_size_um=args.pixel_size_um,
                           jobs=args.jobs, skip_existing=args.skip_existing)
    else:
        to_ome_tiff_file(args.input, args.output, pixel_size_um=args.pixel_size_um)


if __name__ == "__main__":
    cli()
