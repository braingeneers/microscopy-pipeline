"""Average the frames of a TIFF stack into a single image (optionally rescaled)."""

from __future__ import annotations

import logging
import re
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import io
from ..cli import build_io_parser, configure_logging, add_folder_flags

logger = logging.getLogger(__name__)

_STACK_PATTERN = re.compile(r"stack_(\d+)\.tiff?$", re.IGNORECASE)


PROJECTIONS = ("mean", "max", "median", "std", "percentile")


def fuse_project(
    frames: Sequence[np.ndarray],
    *,
    projection: str = "mean",
    percentile: float = 90.0,
    brightfield: bool = False,
    bit_depth: int | None = None,
) -> np.ndarray:
    """Collapse a stack to one image via a projection statistic.

    ``projection`` is one of mean / max / median / std / percentile (the last
    uses ``percentile``).  ``brightfield`` range-stretches the result to fill the
    bit depth.
    """
    if not frames:
        raise ValueError("no frames provided")
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}")
    if bit_depth is None:
        bit_depth = 16 if frames[0].dtype == np.uint16 else 8
    max_val = 65535 if bit_depth == 16 else 255

    stack = np.stack([f.astype(np.float64) for f in frames], axis=0)
    if projection == "mean":
        out = stack.mean(axis=0)
    elif projection == "max":
        out = stack.max(axis=0)
    elif projection == "median":
        out = np.median(stack, axis=0)
    elif projection == "std":
        out = stack.std(axis=0)
    else:  # percentile
        out = np.percentile(stack, percentile, axis=0)

    if brightfield:
        lo, hi = float(out.min()), float(out.max())
        if hi > lo:
            out = (out - lo) / (hi - lo) * max_val
    out = np.clip(out, 0, max_val)
    return out.astype(np.uint16 if bit_depth == 16 else np.uint8)


def fuse_average(
    frames: Sequence[np.ndarray],
    *,
    brightfield: bool = False,
    bit_depth: int | None = None,
) -> np.ndarray:
    """Mean projection (back-compat alias for ``fuse_project(projection='mean')``)."""
    return fuse_project(frames, projection="mean", brightfield=brightfield, bit_depth=bit_depth)


def fuse_average_file(input_path, output_path, *, brightfield: bool = False,
                      projection: str = "mean", percentile: float = 90.0):
    frames = io.load_stack(input_path, grayscale=True)
    if len(frames) < 2:
        raise ValueError("input must be a multi-frame TIFF stack")
    out = fuse_project(frames, projection=projection, percentile=percentile,
                       brightfield=brightfield)
    io.save_image(out, output_path)


def fuse_average_folder(input_dir, output_dir, *, brightfield: bool = False,
                        projection: str = "mean", percentile: float = 90.0,
                        jobs=1, skip_existing=False):
    """Process every ``stack_X.tiff`` file in numerical order of X."""
    output_dir = io.ensure_dir(output_dir)
    matches = []
    for p in Path(input_dir).iterdir():
        m = _STACK_PATTERN.match(p.name)
        if m:
            matches.append((int(m.group(1)), p))
    matches.sort()
    proj_suffix = "_superimposed" if projection == "mean" else f"_{projection}"
    suffix = ("_brightfield" + proj_suffix) if brightfield else proj_suffix
    pairs = [(path, output_dir / f"{path.stem}{suffix}{path.suffix}") for _x, path in matches]
    io.map_folder(pairs, partial(fuse_average_file, brightfield=brightfield,
                                 projection=projection, percentile=percentile),
                  jobs=jobs, skip_existing=skip_existing, desc="fuse")
    logger.info("fused %d stack(s) -> %s", len(pairs), output_dir)


def cli(argv=None):
    parser = build_io_parser(
        "Average all frames of a TIFF stack into a single image. "
        "Folder input processes stack_X.tiff files in numerical order."
    )
    parser.add_argument("--brightfield", action="store_true",
                        help="Stretch the result to fill the full bit-depth range.")
    parser.add_argument("--projection", choices=PROJECTIONS, default="mean",
                        help="Projection statistic across frames (default: mean).")
    parser.add_argument("--percentile", type=float, default=90.0,
                        help="Percentile for --projection percentile (default: 90).")
    add_folder_flags(parser)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    src = Path(args.input)
    if src.is_dir():
        fuse_average_folder(args.input, args.output, brightfield=args.brightfield,
                            projection=args.projection, percentile=args.percentile,
                            jobs=args.jobs, skip_existing=args.skip_existing)
    else:
        fuse_average_file(args.input, args.output, brightfield=args.brightfield,
                          projection=args.projection, percentile=args.percentile)


if __name__ == "__main__":
    cli()
