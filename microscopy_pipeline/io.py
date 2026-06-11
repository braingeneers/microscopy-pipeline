"""Image and stack I/O helpers used by every operation.

The goal is to keep the format-handling logic in one place so the operation
modules can focus on pure-numpy transforms.

Canonical in-memory representation (used uniformly across the package):

* **Single image** -- one ``np.ndarray``: 2-D ``(H, W)`` grayscale (uint8/uint16)
  or 3-D ``(H, W, 3)`` RGB (uint8, RGB channel order).  ``load_image`` returns
  this and ``save_image`` accepts it.
* **Stack / multi-frame TIFF** -- ``List[np.ndarray]`` (a list of 2-D frames),
  *not* a 3-D ``(frames, H, W)`` array.  ``load_stack`` returns a list and
  ``save_stack`` accepts any iterable of frames.  A 3-D array is permitted only
  as a private, transient vectorization buffer inside a single function; it must
  never cross a public function boundary.

Animated GIFs, CSVs and matplotlib figures are deliberately outside this layer
(see ``tiff_to_gif``, ``sum_fluorescence``, ``plot_summed_fluorescence``,
``graph_contour_data``); those ops handle their own specialized I/O.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from PIL import Image

PathLike = Union[str, os.PathLike]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".jp2"}
TIFF_EXTS = {".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Format / mode introspection
# ---------------------------------------------------------------------------

def is_tiff_path(path: PathLike) -> bool:
    return Path(path).suffix.lower() in TIFF_EXTS


def is_tiff_stack(path: PathLike) -> bool:
    """Return True if ``path`` points to a multi-frame TIFF."""
    if not is_tiff_path(path):
        return False
    try:
        with Image.open(path) as img:
            img.seek(1)
            return True
    except (EOFError, AttributeError, FileNotFoundError):
        return False
    except Exception:
        return False


def detect_bit_depth(img_or_array) -> int:
    """Return 8 or 16 for the given PIL image, numpy array, or path."""
    if isinstance(img_or_array, (str, os.PathLike)):
        with Image.open(img_or_array) as img:
            return detect_bit_depth(img)
    if isinstance(img_or_array, np.ndarray):
        if img_or_array.dtype == np.uint16:
            return 16
        if img_or_array.dtype == np.uint8:
            return 8
        return 16 if img_or_array.max() > 255 else 8
    # PIL image
    mode = img_or_array.mode
    if mode in ("I;16", "I;16B", "I;16L"):
        return 16
    if mode == "I":
        try:
            arr = np.asarray(img_or_array)
            return 16 if arr.max() <= 65535 else 32
        except Exception:
            return 16
    return 8


# ---------------------------------------------------------------------------
# PIL <-> ndarray
# ---------------------------------------------------------------------------

def _pil_to_array(img: Image.Image, *, grayscale: bool | None = None) -> np.ndarray:
    """Convert PIL image to numpy array, preserving bit depth where possible."""
    mode = img.mode
    if mode in ("I;16", "I;16B", "I;16L"):
        return np.asarray(img, dtype=np.uint16)
    if mode == "I":
        arr = np.asarray(img)
        if arr.max() <= 65535:
            return arr.astype(np.uint16)
        return arr.astype(np.uint32)
    if mode == "L":
        return np.asarray(img, dtype=np.uint8)
    if grayscale:
        return np.asarray(img.convert("L"), dtype=np.uint8)
    if mode in ("RGB", "RGBA", "BGR"):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """Wrap a numpy array in a PIL image with the appropriate mode."""
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA" if arr.shape[2] == 4 else "RGB")
    if arr.dtype == np.uint16:
        return Image.fromarray(arr, mode="I;16")
    if arr.dtype == np.uint8:
        return Image.fromarray(arr, mode="L")
    if np.issubdtype(arr.dtype, np.integer):
        if arr.max() <= 255:
            return Image.fromarray(arr.astype(np.uint8), mode="L")
        return Image.fromarray(arr.astype(np.uint16), mode="I;16")
    # floating point -> normalize? safer: clip to known range based on max.
    if arr.max() <= 1.0:
        return Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    if arr.max() <= 255:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    return Image.fromarray(np.clip(arr, 0, 65535).astype(np.uint16), mode="I;16")


# ---------------------------------------------------------------------------
# Single-image load/save
# ---------------------------------------------------------------------------

def load_image(path: PathLike, *, grayscale: bool | None = None) -> np.ndarray:
    """Load a single image (or first frame of a stack) as a numpy array.

    Parameters
    ----------
    path : str or Path
    grayscale : bool, optional
        If True, force conversion to single-channel grayscale.  If None
        (default) the image is loaded with its native channels preserved.
    """
    with Image.open(path) as img:
        return _pil_to_array(img, grayscale=grayscale)


def save_image(arr: np.ndarray, path: PathLike, **save_kwargs) -> None:
    """Save a 2-D (or RGB) array as an image file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = array_to_pil(arr)
    img.save(path, **save_kwargs)


# ---------------------------------------------------------------------------
# Stack load/save
# ---------------------------------------------------------------------------

def load_stack(path: PathLike, *, grayscale: bool | None = None) -> List[np.ndarray]:
    """Load every frame of a multi-frame TIFF (or single image) as a list of arrays."""
    frames: List[np.ndarray] = []
    with Image.open(path) as img:
        try:
            while True:
                frames.append(_pil_to_array(img.copy(), grayscale=grayscale))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    return frames


def save_stack(
    frames: Iterable[np.ndarray],
    path: PathLike,
    *,
    imagej: bool = True,
    compression: str | None = "tiff_lzw",
    extra_tags: dict | None = None,
) -> None:
    """Save a list/iterable of 2-D arrays as a multi-frame TIFF stack.

    ``imagej=True`` adds the ImageJ metadata tag (50839) so the result opens
    as a hyperstack in Fiji/ImageJ.  16-bit stacks default to no compression
    (LZW + 16-bit is fragile in PIL).
    """
    frames = [np.asarray(f) for f in frames]
    if not frames:
        raise ValueError("Cannot save empty stack")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pils = [array_to_pil(f) for f in frames]
    bit_depth = 16 if frames[0].dtype == np.uint16 else 8

    save_kwargs: dict = {"save_all": True, "append_images": pils[1:]}
    if compression == "tiff_lzw" and bit_depth == 16:
        compression = None  # avoid PIL 16-bit LZW issues
    if compression:
        save_kwargs["compression"] = compression

    tiff_tags: dict = {}
    if imagej:
        h, w = frames[0].shape[:2]
        n = len(frames)
        max_val = 65535 if bit_depth == 16 else 255
        info = (
            f"ImageJ=1.53t\nimages={n}\nchannels=1\nslices={n}\nframes=1\n"
            f"hyperstack=false\nmode=grayscale\nloop=false\n"
            f"min=0.0\nmax={float(max_val)}\n"
        )
        tiff_tags = {
            50839: info.encode("utf-8"),
            270: info.encode("utf-8"),
            305: "ImageJ",
            269: f"Stack ({n} slices)",
        }
    if extra_tags:
        tiff_tags.update(extra_tags)
    if tiff_tags:
        save_kwargs["tiffinfo"] = tiff_tags

    pils[0].save(path, **save_kwargs)


# ---------------------------------------------------------------------------
# Folder iteration
# ---------------------------------------------------------------------------

def list_images(folder: PathLike, *, exts: Iterable[str] = IMAGE_EXTS) -> List[Path]:
    """Return sorted list of image files in a folder (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    exts_lower = {e.lower() for e in exts}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts_lower)


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Generic batch helper used by op CLIs.
# ---------------------------------------------------------------------------

def batch_apply(
    input_dir: PathLike,
    output_dir: PathLike,
    func,
    *,
    exts: Iterable[str] = IMAGE_EXTS,
    keep_name: bool = True,
) -> Tuple[int, int]:
    """Apply ``func(in_path, out_path)`` to every image in ``input_dir``.

    Returns ``(n_success, n_failed)``.
    """
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir)
    files = list_images(input_dir, exts=exts)
    ok = fail = 0
    for src in files:
        dst = output_dir / src.name if keep_name else output_dir / src.name
        try:
            func(str(src), str(dst))
            ok += 1
        except Exception as exc:  # pragma: no cover -- diagnostic only
            print(f"  failed: {src.name}: {exc}")
            fail += 1
    return ok, fail
