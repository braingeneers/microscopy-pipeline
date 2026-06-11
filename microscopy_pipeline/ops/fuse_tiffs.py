"""Average the frames of a TIFF stack into a single image (optionally rescaled)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import io
from ..cli import build_io_parser

_STACK_PATTERN = re.compile(r"stack_(\d+)\.tiff?$", re.IGNORECASE)


def fuse_average(
    frames: Sequence[np.ndarray],
    *,
    brightfield: bool = False,
    bit_depth: int | None = None,
) -> np.ndarray:
    """Average ``frames`` into a single image; optional brightfield range stretch."""
    if not frames:
        raise ValueError("no frames provided")
    if bit_depth is None:
        bit_depth = 16 if frames[0].dtype == np.uint16 else 8
    max_val = 65535 if bit_depth == 16 else 255

    acc_dtype = np.uint64 if bit_depth == 16 else np.uint32
    summed = np.zeros_like(frames[0], dtype=acc_dtype)
    for f in frames:
        summed += f.astype(acc_dtype)
    averaged = summed / len(frames)

    if brightfield:
        lo, hi = float(averaged.min()), float(averaged.max())
        if hi > lo:
            averaged = (averaged - lo) / (hi - lo) * max_val
    averaged = np.clip(averaged, 0, max_val)
    return averaged.astype(np.uint16 if bit_depth == 16 else np.uint8)


def fuse_average_file(input_path, output_path, *, brightfield: bool = False):
    frames = io.load_stack(input_path, grayscale=True)
    if len(frames) < 2:
        raise ValueError("input must be a multi-frame TIFF stack")
    out = fuse_average(frames, brightfield=brightfield)
    io.save_image(out, output_path)


def fuse_average_folder(input_dir, output_dir, *, brightfield: bool = False):
    """Process every ``stack_X.tiff`` file in numerical order of X."""
    output_dir = io.ensure_dir(output_dir)
    matches = []
    for p in Path(input_dir).iterdir():
        m = _STACK_PATTERN.match(p.name)
        if m:
            matches.append((int(m.group(1)), p))
    matches.sort()
    suffix = "_brightfield_superimposed" if brightfield else "_superimposed"
    for x, path in matches:
        out = output_dir / f"{path.stem}{suffix}{path.suffix}"
        fuse_average_file(str(path), str(out), brightfield=brightfield)
        print(f"  fused {path.name} -> {out.name}")


def cli(argv=None):
    parser = build_io_parser(
        "Average all frames of a TIFF stack into a single image. "
        "Folder input processes stack_X.tiff files in numerical order."
    )
    parser.add_argument("--brightfield", action="store_true",
                        help="Stretch the result to fill the full bit-depth range.")
    args = parser.parse_args(argv)
    src = Path(args.input)
    if src.is_dir():
        fuse_average_folder(args.input, args.output, brightfield=args.brightfield)
    else:
        fuse_average_file(args.input, args.output, brightfield=args.brightfield)


if __name__ == "__main__":
    cli()
