"""Multiply image brightness by a scalar factor (cv2.convertScaleAbs)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .. import io
from ..cli import build_io_parser


def scale_brightness(image: np.ndarray, *, factor: float = 1.0) -> np.ndarray:
    """Multiply pixel values by ``factor`` and clip to the original dtype range."""
    if image.dtype == np.uint8:
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    # Generic path -- preserve dtype.
    out = image.astype(np.float32) * factor
    if image.dtype == np.uint16:
        return np.clip(out, 0, 65535).astype(np.uint16)
    return np.clip(out, 0, 255).astype(np.uint8)


def scale_brightness_file(input_path, output_path, *, factor: float):
    arr = io.load_image(input_path)
    io.save_image(scale_brightness(arr, factor=factor), output_path)


def scale_brightness_folder(input_dir, output_dir, *, factor: float):
    output_dir = io.ensure_dir(output_dir)
    for src in io.list_images(input_dir):
        scale_brightness_file(str(src), str(output_dir / src.name), factor=factor)


def cli(argv=None):
    parser = build_io_parser("Multiply image brightness by a scalar factor.")
    parser.add_argument("--factor", type=float, required=True, help="Brightness multiplier.")
    args = parser.parse_args(argv)
    src = Path(args.input)
    if src.is_dir():
        scale_brightness_folder(args.input, args.output, factor=args.factor)
    else:
        scale_brightness_file(args.input, args.output, factor=args.factor)


if __name__ == "__main__":
    cli()
