"""Linearly remap an 8-bit grayscale band [bottom, top] to the full 0..65535 range."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np

from .. import io
from ..cli import build_io_parser, add_folder_flags


def scale_colors(image: np.ndarray, *, top: int = 255, bottom: int = 0) -> np.ndarray:
    """Map ``[bottom, top]`` (8-bit) linearly onto ``[0, 65535]`` (uint16)."""
    if not (0 <= bottom < top):
        raise ValueError("require 0 <= bottom < top")
    img_f = image.astype(np.float32)
    if img_f.max() > 255:
        img_f = img_f * (255.0 / img_f.max())
    scaled = np.clip((img_f - bottom) * (65535.0 / (top - bottom)), 0, 65535)
    return scaled.astype(np.uint16)


def scale_colors_file(input_path, output_path, *, top=255, bottom=0):
    arr = io.load_image(input_path, grayscale=True)
    io.save_image(scale_colors(arr, top=top, bottom=bottom), output_path)


def scale_colors_folder(input_dir, output_dir, *, top=255, bottom=0, jobs=1, skip_existing=False):
    output_dir = io.ensure_dir(output_dir)
    pairs = [(src, output_dir / (src.stem + ".png")) for src in io.list_images(input_dir)]
    io.map_folder(pairs, partial(scale_colors_file, top=top, bottom=bottom),
                  jobs=jobs, skip_existing=skip_existing, desc="scale-colors")


def cli(argv=None):
    parser = build_io_parser(
        "Scale a grayscale image's brightness band [bottom, top] to fill the 16-bit range 0..65535."
    )
    parser.add_argument("--top", type=int, required=True, help="Upper 8-bit threshold (-> 65535).")
    parser.add_argument("--bottom", type=int, required=True, help="Lower 8-bit threshold (-> 0).")
    add_folder_flags(parser)
    args = parser.parse_args(argv)
    src = Path(args.input)
    if src.is_dir():
        scale_colors_folder(args.input, args.output, top=args.top, bottom=args.bottom,
                            jobs=args.jobs, skip_existing=args.skip_existing)
    else:
        scale_colors_file(args.input, args.output, top=args.top, bottom=args.bottom)


if __name__ == "__main__":
    cli()
