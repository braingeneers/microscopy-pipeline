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


def _normalize_to_8bit(arr: np.ndarray, *, vmin: float | None = None,
                       vmax: float | None = None) -> np.ndarray:
    """Scale ``arr`` to uint8.  With ``vmin``/``vmax`` a shared (global) window is
    used; otherwise the array's own min/max (per-frame).  uint8 input with no
    explicit window passes through unchanged."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8 and vmin is None and vmax is None:
        return arr
    a = arr.astype(np.float32)
    lo = float(a.min()) if vmin is None else float(vmin)
    hi = float(a.max()) if vmax is None else float(vmax)
    if hi > lo:
        a = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255)
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)


def _normalize_frames(frames: Sequence[np.ndarray], mode: str = "global") -> List[np.ndarray]:
    """Return uint8 frames normalized per the chosen ``mode``.

    * ``"global"`` (default): one min/max window computed across all non-uint8
      frames, so absolute brightness differences between frames are preserved
      (no time-lapse flicker).  uint8 frames pass through unchanged.
    * ``"per-frame"``: each frame stretched to its own min/max (legacy behaviour;
      causes flicker when frames differ in absolute intensity).
    * ``"none"``: no rescaling; values are cast/clipped straight to uint8.
    """
    if mode not in ("global", "per-frame", "none"):
        raise ValueError(f"normalize must be 'global', 'per-frame' or 'none', got {mode!r}")
    arrs = [np.asarray(f) for f in frames]
    if mode == "none":
        return [a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8) for a in arrs]
    if mode == "per-frame":
        return [_normalize_to_8bit(a) for a in arrs]
    # global: shared window across the frames that actually need scaling
    scalable = [a for a in arrs if a.dtype != np.uint8]
    if not scalable:
        return arrs
    lo = min(float(a.min()) for a in scalable)
    hi = max(float(a.max()) for a in scalable)
    return [a if a.dtype == np.uint8 else _normalize_to_8bit(a, vmin=lo, vmax=hi) for a in arrs]


def _get_sorted_tiffs(folder: Path, pattern: str | None) -> List[Path]:
    folder = Path(folder)
    if pattern:
        rx = re.compile(pattern)
        matches = [(int(m.group(1)) if m.lastindex else 0, p)
                   for p in folder.iterdir()
                   if (m := rx.search(p.name))]
        return [p for _, p in sorted(matches)]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))


_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _apply_labels(rgb_frames, labels):
    """Draw a text label (top-left) on each RGB uint8 frame; returns new arrays."""
    if not labels:
        return rgb_frames
    import cv2
    out = []
    for i, fr in enumerate(rgb_frames):
        img = np.ascontiguousarray(fr).copy()  # writable buffer for cv2.putText
        text = labels[i] if i < len(labels) else None
        if text:
            cv2.putText(img, str(text), (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(img)
    return out


def frames_to_gif(
    frames: Sequence[np.ndarray],
    output_path,
    *,
    duration_ms: int = 10,
    loop: int = 0,
    optimize: bool = True,
    normalize: str = "global",
    labels=None,
) -> None:
    """Write a sequence of arrays as an animated GIF.

    ``normalize`` controls intensity scaling across the sequence: ``"global"``
    (default) uses one shared window so brightness stays consistent frame to
    frame; ``"per-frame"`` is the legacy per-frame stretch; ``"none"`` skips
    rescaling.  See :func:`_normalize_frames`.  ``labels`` (one string per frame)
    overlays text on each frame.
    """
    if not frames:
        raise ValueError("no frames provided")
    rgb = _apply_labels(
        [np.asarray(Image.fromarray(a).convert("RGB")) for a in _normalize_frames(frames, normalize)],
        labels,
    )
    pil_frames = [Image.fromarray(a) for a in rgb]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=optimize,
    )


def frames_to_video(frames: Sequence[np.ndarray], output_path, *, fps: int = 10,
                    normalize: str = "global", labels=None) -> None:
    """Write frames as a video (mp4/mov/avi). Requires the optional ``imageio`` package."""
    if not frames:
        raise ValueError("no frames provided")
    try:
        import imageio.v2 as _imageio
    except Exception:
        try:
            import imageio as _imageio
        except Exception as exc:
            raise RuntimeError(
                "video output requires the optional 'imageio' package "
                "(pip install 'imageio[ffmpeg]')") from exc
    rgb = _apply_labels(
        [np.asarray(Image.fromarray(a).convert("RGB")) for a in _normalize_frames(frames, normalize)],
        labels,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        _imageio.mimsave(str(output_path), rgb, fps=fps)
    except TypeError:
        # Writers without the imageio-ffmpeg backend (e.g. the pillow fallback)
        # reject the ``fps`` keyword; write at the default frame rate instead.
        _imageio.mimsave(str(output_path), rgb)


def tiff_to_gif(
    input_path,
    output_path,
    *,
    pattern: str | None = r"stack_(\d+)_edf\.tif",
    duration_ms: int = 10,
    loop: int = 0,
    normalize: str = "global",
    labels=None,
):
    """Convert a TIFF stack file or folder of TIFFs into a GIF (or MP4/MOV/AVI).

    The output format is chosen from the ``output_path`` extension; video
    extensions require the optional ``imageio`` package.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        files = _get_sorted_tiffs(input_path, pattern)
        if not files:
            raise FileNotFoundError(f"no matching TIFFs in {input_path}")
        frames = [io.load_image(f) for f in files]
    else:
        frames = io.load_stack(input_path)
    if Path(output_path).suffix.lower() in _VIDEO_EXTS:
        fps = max(1, round(1000.0 / duration_ms)) if duration_ms else 10
        frames_to_video(frames, output_path, fps=fps, normalize=normalize, labels=labels)
    else:
        frames_to_gif(frames, output_path, duration_ms=duration_ms, loop=loop,
                      normalize=normalize, labels=labels)


def cli(argv=None):
    parser = build_io_parser("Convert a folder of TIFFs (or a TIFF stack) into an animated GIF.")
    parser.add_argument("--pattern", default=r"stack_(\d+)_edf\.tif",
                        help="Regex with one numeric capture group to sort folder frames (default matches stack_N_edf.tif).")
    parser.add_argument("--duration-ms", type=int, default=10)
    parser.add_argument("--loop", type=int, default=0, help="0 = infinite.")
    parser.add_argument("--normalize", choices=("global", "per-frame", "none"), default="global",
                        help="Intensity scaling across frames (default: global, avoids flicker).")
    args = parser.parse_args(argv)
    tiff_to_gif(args.input, args.output, pattern=args.pattern,
                duration_ms=args.duration_ms, loop=args.loop, normalize=args.normalize)


if __name__ == "__main__":
    cli()
