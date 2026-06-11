"""CLAHE: Contrast-Limited Adaptive Histogram Equalization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .. import io
from ..cli import build_io_parser


def clahe(
    image: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    channel: Optional[str] = None,
) -> np.ndarray:
    """Apply CLAHE to a grayscale image, or to one channel of an RGB image.

    Parameters
    ----------
    channel : "red" | "green" | "blue" | None
        If given, the image is treated as RGB and CLAHE is applied only to
        the given channel.  ``None`` means grayscale processing.
    """
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if channel is None:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return op.apply(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("channel-specific CLAHE requires an RGB image")
    idx = {"red": 0, "green": 1, "blue": 2}[channel]
    out = image.copy()
    out[:, :, idx] = op.apply(image[:, :, idx])
    return out


def clahe_file(input_path, output_path, *, clip_limit=2.0, tile_grid_size=(8, 8), channel=None):
    if channel is None:
        arr = io.load_image(input_path, grayscale=True)
    else:
        # cv2 reads BGR; we want RGB internally
        bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(input_path)
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = clahe(arr, clip_limit=clip_limit, tile_grid_size=tile_grid_size, channel=channel)
    if channel is not None:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), out)
    else:
        io.save_image(out, output_path)


def clahe_folder(input_dir, output_dir, *, clip_limit=2.0, tile_grid_size=(8, 8), channel=None):
    output_dir = io.ensure_dir(output_dir)
    for src in io.list_images(input_dir):
        clahe_file(str(src), str(output_dir / src.name),
                   clip_limit=clip_limit, tile_grid_size=tile_grid_size, channel=channel)


def cli(argv=None):
    parser = build_io_parser(
        "Apply CLAHE (contrast-limited adaptive histogram equalisation) to an image or folder."
    )
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid", type=int, default=8, help="Tile grid size (creates NxN grid).")
    parser.add_argument("--channel", choices=["red", "green", "blue"], default=None,
                        help="Apply CLAHE to this RGB channel only (default: grayscale).")
    args = parser.parse_args(argv)
    grid = (args.tile_grid, args.tile_grid)
    src = Path(args.input)
    if src.is_dir():
        clahe_folder(args.input, args.output, clip_limit=args.clip_limit,
                     tile_grid_size=grid, channel=args.channel)
    else:
        clahe_file(args.input, args.output, clip_limit=args.clip_limit,
                   tile_grid_size=grid, channel=args.channel)


if __name__ == "__main__":
    cli()
