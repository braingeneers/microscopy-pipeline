"""Apply a binary (or feathered) mask to an image, replacing masked pixels."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

from .. import io
from ..cli import build_io_parser, add_folder_flags

ColorLike = Union[str, int, Tuple[int, int, int]]

_NAMED_COLORS = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "black": (0, 0, 0), "white": (255, 255, 255),
    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
    "orange": (255, 165, 0), "purple": (128, 0, 128), "pink": (255, 192, 203),
    "gray": (128, 128, 128), "grey": (128, 128, 128),
    "lightgray": (211, 211, 211), "lightgrey": (211, 211, 211),
    "darkgray": (64, 64, 64), "darkgrey": (64, 64, 64),
    "white16": (65535, 65535, 65535), "lightgray16": (49151, 49151, 49151),
    "gray16": (32767, 32767, 32767), "darkgray16": (16383, 16383, 16383),
    "black16": (0, 0, 0),
}


def parse_color(color: ColorLike) -> Tuple[int, int, int]:
    """Convert a name / hex code / ``"R,G,B"`` / scalar gray value to an RGB tuple."""
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError("color tuple must have 3 components")
        return tuple(int(c) for c in color)  # type: ignore[return-value]
    if isinstance(color, int):
        return (color, color, color)
    s = str(color).strip()
    low = s.lower()
    if low in _NAMED_COLORS:
        return _NAMED_COLORS[low]
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6:
            raise ValueError(f"invalid hex color: {s}")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    if "," in s:
        parts = [int(p) for p in s.split(",")]
        if len(parts) != 3 or not all(0 <= p <= 65535 for p in parts):
            raise ValueError(f"invalid R,G,B color: {s}")
        return tuple(parts)  # type: ignore[return-value]
    try:
        v = int(s)
        return (v, v, v)
    except ValueError as e:
        raise ValueError(f"unrecognised color: {s}") from e


def _load_mask_array(mask: Union[np.ndarray, str, Path], shape, feather_radius: float) -> np.ndarray:
    """Return a float32 mask in [0, 1] with the same H/W as ``shape``."""
    if isinstance(mask, np.ndarray):
        m = mask
    else:
        m = io.load_image(mask, grayscale=True)
    if m.shape[:2] != shape[:2]:
        raise ValueError(f"mask shape {m.shape[:2]} does not match image {shape[:2]}")
    if feather_radius > 0:
        pil = Image.fromarray(m.astype(np.uint8) if m.dtype != np.uint8 else m, mode="L")
        pil = pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        m = np.asarray(pil, dtype=np.float32) / 255.0
    else:
        m = m.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
    return m


def mask(
    image: np.ndarray,
    mask_image: Union[np.ndarray, str, Path],
    *,
    replace_color: ColorLike = "red",
    feather_radius: float = 0.0,
) -> np.ndarray:
    """Replace pixels where ``mask_image`` is dark with ``replace_color``.

    A pixel value of 0 (black) in the mask means *replace*; 255 (white) means
    *keep*.  ``feather_radius > 0`` enables alpha blending across mask edges.
    Works with 8-bit or 16-bit grayscale and 8-bit RGB images.
    """
    color = parse_color(replace_color)
    is_rgb = image.ndim == 3 and image.shape[2] >= 3
    is_16bit = image.dtype == np.uint16

    m = _load_mask_array(mask_image, image.shape, feather_radius)

    if is_rgb:
        replace_value = np.array(color, dtype=np.float32)
        img_f = image.astype(np.float32)
        if feather_radius > 0:
            alpha = (1.0 - m)[..., None]
            blended = img_f[..., :3] * (1 - alpha) + replace_value * alpha
        else:
            sel = m < 0.5
            blended = img_f[..., :3].copy()
            blended[sel] = replace_value
        out = np.clip(blended, 0, 255).astype(np.uint8)
        if image.shape[2] == 4:
            out = np.dstack([out, image[..., 3]])
        return out

    # grayscale path -- collapse colour to a single luminance value
    r, g, b = color
    if r == g == b:
        gray_val = float(r)
    else:
        gray_val = 0.299 * r + 0.587 * g + 0.114 * b
    if is_16bit and gray_val <= 255:
        gray_val *= 257.0  # scale 8-bit replacement up to 16-bit
    elif not is_16bit and gray_val > 255:
        gray_val = gray_val / 257.0

    img_f = image.astype(np.float32)
    if feather_radius > 0:
        alpha = 1.0 - m
        blended = img_f * (1 - alpha) + gray_val * alpha
    else:
        sel = m < 0.5
        blended = img_f.copy()
        blended[sel] = gray_val

    if is_16bit:
        return np.clip(blended, 0, 65535).astype(np.uint16)
    return np.clip(blended, 0, 255).astype(np.uint8)


def mask_file(input_path, output_path, mask_path, *, replace_color="red", feather_radius=0.0):
    """Apply a mask to a single image or every frame of a TIFF stack."""
    if io.is_tiff_stack(input_path):
        frames = io.load_stack(input_path)
        masked = [mask(f, mask_path, replace_color=replace_color, feather_radius=feather_radius) for f in frames]
        io.save_stack(masked, output_path)
    else:
        arr = io.load_image(input_path)
        io.save_image(mask(arr, mask_path, replace_color=replace_color, feather_radius=feather_radius), output_path)


def mask_folder(input_dir, output_dir, mask_path, *, replace_color="red", feather_radius=0.0,
                jobs=1, skip_existing=False):
    output_dir = io.ensure_dir(output_dir)
    pairs = [(src, output_dir / src.name) for src in io.list_images(input_dir)]
    io.map_folder(pairs, partial(mask_file, mask_path=mask_path,
                                 replace_color=replace_color, feather_radius=feather_radius),
                  jobs=jobs, skip_existing=skip_existing, desc="mask")


def cli(argv=None):
    parser = build_io_parser(
        "Apply a binary or feathered mask to an image (or stack); replace masked pixels with a given color."
    )
    parser.add_argument("--mask", required=True, help="Path to mask image (black = replace, white = keep).")
    parser.add_argument("--replace-color", default="red", help="Replacement color (name, #hex, R,G,B, or scalar).")
    parser.add_argument("--feather-radius", type=float, default=0.0, help="Gaussian feather radius in pixels.")
    add_folder_flags(parser)
    args = parser.parse_args(argv)
    src = Path(args.input)
    if src.is_dir():
        mask_folder(args.input, args.output, args.mask,
                    replace_color=args.replace_color, feather_radius=args.feather_radius,
                    jobs=args.jobs, skip_existing=args.skip_existing)
    else:
        mask_file(args.input, args.output, args.mask,
                  replace_color=args.replace_color, feather_radius=args.feather_radius)


if __name__ == "__main__":
    cli()
