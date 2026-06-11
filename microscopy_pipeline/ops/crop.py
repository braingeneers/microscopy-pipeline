"""Crop pixels off the edges of an image (or every frame of a stack)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import io
from ..cli import build_io_parser


def crop(image: np.ndarray, *, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> np.ndarray:
    """Crop ``left/top/right/bottom`` pixels off a 2-D or RGB image array."""
    if any(v < 0 for v in (left, top, right, bottom)):
        raise ValueError("crop amounts must be non-negative")
    h, w = image.shape[:2]
    return image[top : h - bottom, left : w - right]


def crop_stack(frames, **kwargs):
    """Apply :func:`crop` to every frame of a stack (list of arrays)."""
    return [crop(f, **kwargs) for f in frames]


def crop_file(input_path, output_path, *, left=0, top=0, right=0, bottom=0):
    """Crop a single image or every frame of a TIFF stack."""
    if io.is_tiff_stack(input_path):
        frames = io.load_stack(input_path)
        cropped = crop_stack(frames, left=left, top=top, right=right, bottom=bottom)
        io.save_stack(cropped, output_path)
    else:
        arr = io.load_image(input_path)
        io.save_image(crop(arr, left=left, top=top, right=right, bottom=bottom), output_path)


def crop_folder(input_dir, output_dir, *, left=0, top=0, right=0, bottom=0):
    """Crop every image (or stack) in ``input_dir`` into ``output_dir``."""
    output_dir = io.ensure_dir(output_dir)
    for src in io.list_images(input_dir):
        crop_file(str(src), str(output_dir / src.name), left=left, top=top, right=right, bottom=bottom)


def cli(argv=None):
    parser = build_io_parser("Crop pixels from image edges. Supports single files, folders, and TIFF stacks.")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--right", type=int, default=0)
    parser.add_argument("--bottom", type=int, default=0)
    args = parser.parse_args(argv)
    src = Path(args.input)
    if src.is_dir():
        crop_folder(args.input, args.output, left=args.left, top=args.top, right=args.right, bottom=args.bottom)
    else:
        crop_file(args.input, args.output, left=args.left, top=args.top, right=args.right, bottom=args.bottom)


if __name__ == "__main__":
    cli()
