"""Bundle ``{X}_{Y}.png`` files into one TIFF stack per X value (ImageJ-compatible)."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .. import io
from ..cli import build_io_parser

_PNG_PATTERN = re.compile(r"(\d+)_(-?\d+)\.png$")


def group_pngs_by_x(folder) -> Dict[int, List[Tuple[int, Path]]]:
    """Group files matching ``X_Y.png`` by X; values are ``[(Y, path), ...]`` sorted by Y."""
    folder = Path(folder)
    groups: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)
    for p in folder.iterdir():
        m = _PNG_PATTERN.match(p.name)
        if m:
            groups[int(m.group(1))].append((int(m.group(2)), p))
    for k in groups:
        groups[k].sort(key=lambda t: t[0])
    return groups


def _convert(img: Image.Image, bit_depth: int) -> Image.Image:
    if bit_depth == 16:
        if img.mode in ("I;16", "I;16B", "I;16L"):
            return img
        if img.mode == "I":
            return img.point(lambda v: min(v, 65535)).convert("I;16")
        if img.mode == "L":
            return img.point(lambda v: v * 257).convert("I;16")
        return img.convert("L").point(lambda v: v * 257).convert("I;16")
    # 8-bit target
    if img.mode in ("I;16", "I;16B", "I;16L"):
        return img.point(lambda v: v // 257).convert("L")
    if img.mode == "I":
        return img.point(lambda v: min(v // 257, 255)).convert("L")
    if img.mode != "L":
        return img.convert("L")
    return img


def pngs_to_tiff_stacks(input_dir, output_dir, *, bit_depth: int | None = None) -> List[Path]:
    """Create one ``stack_{X}.tiff`` per X value found in ``input_dir``.

    ``bit_depth`` defaults to the bit depth of the first image in each group.
    Returns the list of output paths.
    """
    output_dir = io.ensure_dir(output_dir)
    groups = group_pngs_by_x(input_dir)
    outputs: List[Path] = []
    for x in sorted(groups):
        files = [p for _, p in groups[x]]
        if not files:
            continue
        first = Image.open(files[0])
        depth = bit_depth or io.detect_bit_depth(first)
        first.close()
        frames = [np.asarray(_convert(Image.open(p), depth)) for p in files]
        out = output_dir / f"stack_{x}.tiff"
        # 16-bit stacks: avoid LZW issues automatically inside save_stack
        compression = "tiff_lzw" if depth == 8 else None
        io.save_stack(frames, out, imagej=True, compression=compression)
        outputs.append(out)
        print(f"wrote {out} ({len(frames)} slices, {depth}-bit)")
    return outputs


def cli(argv=None):
    parser = build_io_parser(
        "Bundle {X}_{Y}.png files into one ImageJ-compatible TIFF stack per X (sorted by Y)."
    )
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=None,
                        help="Force output bit depth (default: derive from input).")
    args = parser.parse_args(argv)
    pngs_to_tiff_stacks(args.input, args.output, bit_depth=args.bit_depth)


if __name__ == "__main__":
    cli()
