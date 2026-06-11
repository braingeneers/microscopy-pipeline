"""Add a scale bar (rectangle + label) to an image."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .. import io
from ..cli import build_io_parser
from .mask import parse_color


def add_scale_bar(
    image: np.ndarray,
    *,
    scale_um: float,
    pixel_size_um: float,
    bar_height: int = 10,
    margin: int = 30,
    color=(255, 255, 255),
    label: bool = True,
) -> np.ndarray:
    """Draw a scale bar in the lower-left corner.  ``color`` is RGB."""
    if scale_um <= 0 or pixel_size_um <= 0:
        raise ValueError("scale_um and pixel_size_um must be positive")
    if isinstance(color, str):
        color = parse_color(color)
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))

    # Work in BGR (cv2 convention)
    if image.ndim == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.dtype == np.uint8 else image[..., ::-1].copy()

    h, w = img.shape[:2]
    bar_px = int(round(scale_um / pixel_size_um))
    if bar_px + 2 * margin > w:
        raise ValueError(f"scale bar ({bar_px}px) too long for image width ({w}px)")
    x0, y0 = margin, h - margin - bar_height
    cv2.rectangle(img, (x0, y0), (x0 + bar_px, y0 + bar_height), bgr_color, -1)
    if label:
        text = f"{scale_um:g} um"
        cv2.putText(img, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_color, 2, cv2.LINE_AA)

    # Convert back to RGB for downstream consistency.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_scale_bar_file(input_path, output_path, *, scale_um, pixel_size_um,
                       bar_height=10, margin=30, color="white", label=True):
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(input_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = add_scale_bar(rgb, scale_um=scale_um, pixel_size_um=pixel_size_um,
                        bar_height=bar_height, margin=margin, color=color, label=label)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), bgr):
        raise IOError(f"failed to save {output_path}")


def add_scale_bar_folder(input_dir, output_dir, *, scale_um, pixel_size_um,
                         bar_height=10, margin=30, color="white", label=True):
    output_dir = io.ensure_dir(output_dir)
    for src in io.list_images(input_dir):
        add_scale_bar_file(str(src), str(output_dir / src.name),
                           scale_um=scale_um, pixel_size_um=pixel_size_um,
                           bar_height=bar_height, margin=margin, color=color, label=label)


def cli(argv=None):
    parser = build_io_parser("Add a scale bar (and optional label) to microscopy images.")
    parser.add_argument("--scale-um", type=float, required=True, help="Scale bar length in micrometres.")
    parser.add_argument("--pixel-size-um", type=float, required=True, help="Pixel size in micrometres.")
    parser.add_argument("--bar-height", type=int, default=10)
    parser.add_argument("--margin", type=int, default=30)
    parser.add_argument("--color", default="white")
    parser.add_argument("--no-label", action="store_true", help="Suppress the text label.")
    args = parser.parse_args(argv)
    src = Path(args.input)
    fn = add_scale_bar_folder if src.is_dir() else add_scale_bar_file
    fn(args.input, args.output, scale_um=args.scale_um, pixel_size_um=args.pixel_size_um,
       bar_height=args.bar_height, margin=args.margin, color=args.color, label=not args.no_label)


if __name__ == "__main__":
    cli()
