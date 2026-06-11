"""Detect and patch single-pixel hot spots that persist across many images.

The original detection algorithm flags a pixel as "lone" when it is
``threshold`` (e.g. 20%) brighter than every 8-connected neighbour, and remains
so in at least ``min_fraction`` of the input images.  Lone pixels are replaced
with the mean of their 8 neighbours.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .. import io
from ..cli import build_io_parser


def find_lone_pixels(images: Sequence[np.ndarray], *, threshold: float = 0.2,
                     min_fraction: float = 0.2) -> np.ndarray:
    """Return a 2-D boolean mask of lone-bright pixels."""
    if not images:
        return np.zeros((0, 0), dtype=bool)
    stack = np.stack([img.astype(np.float32) for img in images], axis=0)
    n, h, w = stack.shape
    padded = np.pad(stack, ((0, 0), (1, 1), (1, 1)), mode="edge")
    centre = padded[:, 1:-1, 1:-1]
    neighbours = np.stack([
        padded[:, 0:-2, 1:-1], padded[:, 2:, 1:-1],
        padded[:, 1:-1, 0:-2], padded[:, 1:-1, 2:],
        padded[:, 0:-2, 0:-2], padded[:, 0:-2, 2:],
        padded[:, 2:, 0:-2], padded[:, 2:, 2:],
    ], axis=0)
    is_lone = np.all(centre[None] > neighbours * (1 + threshold), axis=0)  # (n, h, w)
    counts = np.sum(is_lone, axis=0)
    return counts >= int(n * min_fraction)


def remove_lone_pixels(image: np.ndarray, lone_mask: np.ndarray) -> np.ndarray:
    """Replace pixels indicated by ``lone_mask`` with the mean of their 8 neighbours."""
    if lone_mask.shape != image.shape[:2]:
        raise ValueError("mask shape must match image")
    out = image.copy()
    padded = np.pad(image.astype(np.float32), 1, mode="edge")
    neighbours = np.stack([
        padded[0:-2, 1:-1], padded[2:, 1:-1],
        padded[1:-1, 0:-2], padded[1:-1, 2:],
        padded[0:-2, 0:-2], padded[0:-2, 2:],
        padded[2:, 0:-2], padded[2:, 2:],
    ], axis=0)
    mean_n = neighbours.mean(axis=0)
    out[lone_mask] = mean_n[lone_mask].astype(image.dtype)
    return out


def remove_lone_pixels_batch(
    images: Sequence[np.ndarray], *, threshold: float = 0.2, min_fraction: float = 0.2,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return (corrected_images, lone_mask) for a single group of images."""
    mask = find_lone_pixels(images, threshold=threshold, min_fraction=min_fraction)
    return [remove_lone_pixels(img, mask) for img in images], mask


def _group_files_by_prefix(folder: Path) -> dict:
    """Group files by their prefix (numeric part before the first underscore)."""
    groups: dict = defaultdict(list)
    for p in io.list_images(folder):
        name = p.name
        prefix = name.split("_", 1)[0] if "_" in name else p.stem
        groups[prefix].append(p)
    for k in groups:
        groups[k].sort()
    return groups


def remove_lone_pixels_folder(
    input_dir, output_dir, *, threshold: float = 0.2, min_fraction: float = 0.2,
    start_prefix: str | None = None,
):
    """Group images by numeric prefix, detect lone pixels per group, and write corrections."""
    output_dir = io.ensure_dir(output_dir)
    groups = _group_files_by_prefix(Path(input_dir))

    def sort_key(prefix: str):
        try:
            return (0, int(prefix))
        except ValueError:
            return (1, prefix)

    started = start_prefix is None
    for prefix in sorted(groups, key=sort_key):
        if not started:
            if prefix == start_prefix:
                started = True
            else:
                continue
        files = groups[prefix]
        images = [io.load_image(str(f), grayscale=True) for f in files]
        corrected, _ = remove_lone_pixels_batch(images, threshold=threshold, min_fraction=min_fraction)
        for src, arr in zip(files, corrected):
            io.save_image(arr, str(output_dir / src.name))


def cli(argv=None):
    parser = build_io_parser(
        "Remove single-pixel hot spots common across a group of images. Files are grouped "
        "by the numeric prefix before the first underscore."
    )
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Min fractional brightness over neighbours (default 0.2 = 20%%).")
    parser.add_argument("--min-fraction", type=float, default=0.2,
                        help="Min fraction of images in which a pixel must be lone (default 0.2).")
    parser.add_argument("--start-prefix", default=None, help="Resume from this group prefix.")
    args = parser.parse_args(argv)
    if not 0 < args.threshold <= 1 or not 0 < args.min_fraction <= 1:
        parser.error("--threshold and --min-fraction must be in (0, 1]")
    remove_lone_pixels_folder(args.input, args.output,
                              threshold=args.threshold, min_fraction=args.min_fraction,
                              start_prefix=args.start_prefix)


if __name__ == "__main__":
    cli()
