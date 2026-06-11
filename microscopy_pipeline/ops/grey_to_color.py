"""Convert grayscale images to colorized RGB by setting one channel to the gray value."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .. import io
from ..cli import build_io_parser

CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2}


def _gamma_correct(arr: np.ndarray, gamma: float, max_val: float) -> np.ndarray:
    if gamma == 1.0:
        return arr
    norm = arr.astype(np.float64) / max_val
    corrected = np.power(norm, gamma) * max_val
    return corrected.astype(arr.dtype)


def grey_to_color(image: np.ndarray, *, channel: str = "red", gamma: float = 1.0) -> np.ndarray:
    """Map a grayscale array onto one RGB channel; returns 8-bit RGB."""
    if channel not in CHANNEL_INDEX:
        raise ValueError(f"channel must be one of {list(CHANNEL_INDEX)}")
    if image.dtype == np.uint16:
        max_val = 65535
        gray = image
    else:
        max_val = 255
        gray = image if image.dtype == np.uint8 else image.astype(np.uint8)

    rgb = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=gray.dtype)
    rgb[..., CHANNEL_INDEX[channel]] = gray
    rgb = _gamma_correct(rgb, gamma, max_val)
    if rgb.dtype == np.uint16:
        rgb = (rgb.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
    return rgb


def grey_to_color_file(input_path, output_path, *, channel: str = "red", gamma: float = 1.0):
    arr = io.load_image(input_path)
    if arr.ndim == 3:
        arr = np.asarray(Image.fromarray(arr).convert("L"))
    out = grey_to_color(arr, channel=channel, gamma=gamma)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGB").save(output_path)


def grey_to_color_folder(input_dir, output_dir, *, channel: str = "red", gamma: float = 1.0):
    output_dir = io.ensure_dir(output_dir)
    for src in io.list_images(input_dir, exts={".png", ".tif", ".tiff"}):
        grey_to_color_file(str(src), str(output_dir / src.name), channel=channel, gamma=gamma)


def cli(argv=None):
    parser = build_io_parser("Colorise grayscale images by routing the gray value into one RGB channel.")
    parser.add_argument("--channel", choices=list(CHANNEL_INDEX), required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args(argv)
    src = Path(args.input)
    if src.is_dir():
        grey_to_color_folder(args.input, args.output, channel=args.channel, gamma=args.gamma)
    else:
        grey_to_color_file(args.input, args.output, channel=args.channel, gamma=args.gamma)


if __name__ == "__main__":
    cli()
