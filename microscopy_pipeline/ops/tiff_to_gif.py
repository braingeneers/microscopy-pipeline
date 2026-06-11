"""Build animated GIFs from TIFF stacks or folders of TIFF frames."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

from .. import io
from ..cli import build_io_parser

_DEFAULT_PATTERN = re.compile(r"stack_(\d+)_edf\.tif", re.IGNORECASE)


def _normalize_to_8bit(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    if a.max() > a.min():
        a = (a - a.min()) / (a.max() - a.min()) * 255.0
    return a.astype(np.uint8)


def _get_sorted_tiffs(folder: Path, pattern: str | None) -> List[Path]:
    folder = Path(folder)
    if pattern:
        rx = re.compile(pattern)
        matches = [(int(m.group(1)) if m.lastindex else 0, p)
                   for p in folder.iterdir()
                   if (m := rx.search(p.name))]
        return [p for _, p in sorted(matches)]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))


def frames_to_gif(
    frames: Sequence[np.ndarray],
    output_path,
    *,
    duration_ms: int = 10,
    loop: int = 0,
    optimize: bool = True,
) -> None:
    """Write a sequence of arrays as an animated GIF."""
    if not frames:
        raise ValueError("no frames provided")
    pil_frames = [Image.fromarray(_normalize_to_8bit(np.asarray(f))).convert("RGB") for f in frames]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=optimize,
    )


def tiff_to_gif(
    input_path,
    output_path,
    *,
    pattern: str | None = r"stack_(\d+)_edf\.tif",
    duration_ms: int = 10,
    loop: int = 0,
):
    """Convert a TIFF stack file or a folder of TIFFs into an animated GIF."""
    input_path = Path(input_path)
    if input_path.is_dir():
        files = _get_sorted_tiffs(input_path, pattern)
        if not files:
            raise FileNotFoundError(f"no matching TIFFs in {input_path}")
        frames = [io.load_image(f) for f in files]
    else:
        frames = io.load_stack(input_path)
    frames_to_gif(frames, output_path, duration_ms=duration_ms, loop=loop)


def cli(argv=None):
    parser = build_io_parser("Convert a folder of TIFFs (or a TIFF stack) into an animated GIF.")
    parser.add_argument("--pattern", default=r"stack_(\d+)_edf\.tif",
                        help="Regex with one numeric capture group to sort folder frames (default matches stack_N_edf.tif).")
    parser.add_argument("--duration-ms", type=int, default=10)
    parser.add_argument("--loop", type=int, default=0, help="0 = infinite.")
    args = parser.parse_args(argv)
    tiff_to_gif(args.input, args.output, pattern=args.pattern,
                duration_ms=args.duration_ms, loop=args.loop)


if __name__ == "__main__":
    cli()
