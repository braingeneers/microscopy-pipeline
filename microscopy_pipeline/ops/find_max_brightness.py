"""Report the maximum pixel value of an image (after a small Gaussian blur)."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from .. import io
from ..cli import build_io_parser


class MaxBrightness(NamedTuple):
    max_value: float
    abs_max: float
    ratio: float


def find_max_brightness(image: np.ndarray, *, blur_kernel: int = 5) -> MaxBrightness:
    """Return the max pixel value after a Gaussian blur, plus the abs-max ratio."""
    if blur_kernel and blur_kernel >= 3:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    abs_max = 65535.0 if image.dtype == np.uint16 else 255.0
    max_v = float(np.max(image))
    return MaxBrightness(max_v, abs_max, max_v / abs_max)


def find_max_brightness_file(input_path, *, blur_kernel: int = 5) -> MaxBrightness:
    arr = io.load_image(input_path, grayscale=True)
    return find_max_brightness(arr, blur_kernel=blur_kernel)


def cli(argv=None):
    parser = build_io_parser(
        "Report the maximum pixel brightness in an image (after a small Gaussian blur).",
        output_required=False,
    )
    parser.add_argument("--blur-kernel", type=int, default=5)
    args = parser.parse_args(argv)
    res = find_max_brightness_file(args.input, blur_kernel=args.blur_kernel)
    print(f"max={res.max_value}  abs_max={res.abs_max}  ratio={res.ratio:.4f}")


if __name__ == "__main__":
    cli()
