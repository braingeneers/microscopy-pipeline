"""Complex-wavelet Extended Depth of Field fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pywt

from .. import io
from ..cli import build_io_parser


def complex_edf(
    images: Sequence[np.ndarray],
    *,
    wavelet: str = "db3",
    levels: int = 3,
    bit_depth: int = 16,
    top_n: int = 1,
    invert: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """Extended depth of field via complex wavelets.

    ``invert=True`` selects the *least* sharp top-N coefficients (matches the
    behaviour of the legacy ``invertedcomplexedf.py`` script).  Returns
    ``(fused, min_z_used, max_z_used)``.
    """
    max_val = 65535 if bit_depth == 16 else 255
    floats = [img.astype(np.float64) for img in images]

    def decompose(img):
        coeffs = pywt.wavedec2(img, wavelet, level=levels)
        out = []
        for c in coeffs:
            if isinstance(c, tuple):
                cH, cV, cD = c
                out.append((cH + 1j * cV, cD))
            else:
                out.append(c)
        return out

    def reconstruct(coeffs):
        real = []
        for c in coeffs:
            if isinstance(c, tuple):
                complex_c, cD = c
                real.append((np.real(complex_c), np.imag(complex_c), cD))
            else:
                real.append(c)
        return pywt.waverec2(real, wavelet)

    decomposed = [decompose(img) for img in floats]
    z_indices: list = []
    fused_coeffs = []

    for level_idx, level_coeffs in enumerate(zip(*decomposed)):
        if isinstance(level_coeffs[0], tuple):
            complex_list, cD_list = zip(*level_coeffs)
            mags = np.stack([np.abs(c) for c in complex_list], axis=0)
            n = min(top_n, len(complex_list))
            sign = 1 if invert else -1
            top_indices = np.argpartition(sign * mags, n - 1, axis=0)[:n]
            if level_idx == 0:
                z_indices.extend(top_indices.flatten())
            cstack = np.stack(complex_list, axis=0)
            dstack = np.stack(cD_list, axis=0)
            fc = np.zeros_like(complex_list[0])
            fd = np.zeros_like(cD_list[0])
            for i in range(n):
                fc += np.choose(top_indices[i], cstack)
                fd += np.choose(top_indices[i], dstack)
            fused_coeffs.append((fc / n, fd / n))
        else:
            mags = np.stack([np.abs(c) for c in level_coeffs], axis=0)
            cstack = np.stack(level_coeffs, axis=0)
            n = min(top_n, len(level_coeffs))
            sign = 1 if invert else -1
            top_indices = np.argpartition(sign * mags, n - 1, axis=0)[:n]
            if level_idx == 0:
                z_indices.extend(top_indices.flatten())
            fa = np.zeros_like(level_coeffs[0])
            for i in range(n):
                fa += np.choose(top_indices[i], cstack)
            fused_coeffs.append(fa / n)

    fused = reconstruct(fused_coeffs)
    fused = np.clip(fused, 0, max_val)
    if z_indices:
        return fused, int(np.min(z_indices)), int(np.max(z_indices))
    return fused, 0, len(images) - 1


def _scale_to_target(arr: np.ndarray, target_bit: int) -> np.ndarray:
    if target_bit == 16:
        if arr.dtype == np.uint8:
            return arr.astype(np.uint16) * 257
        return arr.astype(np.uint16)
    if arr.dtype == np.uint16 or (np.issubdtype(arr.dtype, np.integer) and arr.max() > 255):
        return (arr.astype(np.float32) / 257).astype(np.uint8)
    return arr.astype(np.uint8)


def complex_edf_image(images, *, wavelet="db3", levels=3, bit_depth=16,
                      top_n=1, invert=False) -> np.ndarray:
    """Fuse a focal stack and return ONLY the image, cast to its target dtype.

    Convenience wrapper around :func:`complex_edf` for task-to-task chaining.
    :func:`complex_edf` returns the 3-tuple ``(fused_float64, min_z, max_z)`` and
    leaves casting to the caller; this returns a single ready-to-use
    ``np.ndarray`` (uint16 or uint8), so it composes exactly like
    :func:`microscopy_pipeline.ops.fuse_average`.  Input frames are first scaled
    to the target bit depth (mirroring :func:`complex_edf_file`).
    """
    frames = [_scale_to_target(np.asarray(f), bit_depth) for f in images]
    fused, _min_z, _max_z = complex_edf(frames, wavelet=wavelet, levels=levels,
                                        bit_depth=bit_depth, top_n=top_n, invert=invert)
    return fused.astype(np.uint16) if bit_depth == 16 else fused.astype(np.uint8)


def complex_edf_file(input_path, output_path, *, wavelet="db3", levels=3,
                     bit_depth=16, top_n=1, invert=False):
    frames = io.load_stack(input_path, grayscale=True)
    frames = [_scale_to_target(f, bit_depth) for f in frames]
    fused, min_z, max_z = complex_edf(frames, wavelet=wavelet, levels=levels,
                                       bit_depth=bit_depth, top_n=top_n, invert=invert)
    out = fused.astype(np.uint16) if bit_depth == 16 else fused.astype(np.uint8)
    io.save_image(out, output_path)
    print(f"  z range used: {min_z}..{max_z} ({len(frames)} frames)")
    return min_z, max_z


def complex_edf_folder(input_dir, output_dir, *, wavelet="db3", levels=3,
                       bit_depth=16, top_n=1, invert=False):
    output_dir = io.ensure_dir(output_dir)
    for src in sorted(Path(input_dir).iterdir()):
        if src.suffix.lower() not in (".tif", ".tiff"):
            continue
        out = output_dir / f"{src.stem}_edf.tif"
        try:
            complex_edf_file(str(src), str(out), wavelet=wavelet, levels=levels,
                             bit_depth=bit_depth, top_n=top_n, invert=invert)
        except Exception as exc:  # pragma: no cover
            print(f"  failed: {src.name}: {exc}")


def cli(argv=None):
    parser = build_io_parser(
        "Complex-wavelet Extended Depth-of-Field fusion of a TIFF stack (or folder of stacks)."
    )
    parser.add_argument("--wavelet", default="db3")
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=16)
    parser.add_argument("--top-n", type=int, default=1, help="Average the top N sharpest layers per pixel.")
    parser.add_argument("--invert", action="store_true",
                        help="Select least-sharp coefficients (legacy invertedcomplexedf behaviour).")
    args = parser.parse_args(argv)
    src = Path(args.input)
    if src.is_dir():
        complex_edf_folder(args.input, args.output, wavelet=args.wavelet, levels=args.levels,
                           bit_depth=args.bit_depth, top_n=args.top_n, invert=args.invert)
    else:
        complex_edf_file(args.input, args.output, wavelet=args.wavelet, levels=args.levels,
                         bit_depth=args.bit_depth, top_n=args.top_n, invert=args.invert)


if __name__ == "__main__":
    cli()
