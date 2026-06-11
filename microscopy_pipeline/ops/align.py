"""ECC-based image alignment for z-stack/timepoint datasets.

Filenames follow ``{timepoint}_zs+{z}.png`` and outputs are written as
``{timepoint}_{z}.png``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .. import io
from ..cli import build_io_parser


def align_pair(reference: np.ndarray, moving: np.ndarray, *,
               warp_mode: int = cv2.MOTION_TRANSLATION,
               max_iterations: int = 5000,
               eps: float = 1e-10) -> np.ndarray:
    """Align ``moving`` to ``reference`` using the ECC algorithm; returns warped image."""
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    mov = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY) if moving.ndim == 3 else moving
    warp = np.eye(3, 3, dtype=np.float32) if warp_mode == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, eps)
    _, warp = cv2.findTransformECC(ref, mov, warp, warp_mode, criteria)
    h, w = reference.shape[:2]
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(moving, warp, (w, h), flags=flags)
    return cv2.warpAffine(moving, warp, (w, h), flags=flags)


def align_stack(frames, *, reference: Optional[np.ndarray] = None,
                warp_mode: int = cv2.MOTION_TRANSLATION,
                max_iterations: int = 5000, eps: float = 1e-10):
    """Register every frame of a stack to a reference; returns ``List[np.ndarray]``.

    In-memory counterpart to :func:`align_session` (which is directory-only and
    locked to the ``{timepoint}_zs+{z}.png`` naming convention).  ``reference``
    defaults to the first frame.  Frames should be 8-bit (uint8) grayscale or RGB
    -- OpenCV's ECC solver only accepts 8-bit or float32 single-channel input.
    A frame whose alignment fails to converge is returned unchanged.
    """
    frames = list(frames)
    if not frames:
        return []
    ref = frames[0] if reference is None else reference
    aligned = []
    for f in frames:
        try:
            aligned.append(align_pair(ref, f, warp_mode=warp_mode,
                                      max_iterations=max_iterations, eps=eps))
        except cv2.error:
            aligned.append(f)
    return aligned


def align_session(
    *,
    input_dir,
    output_dir,
    min_timepoint: int,
    max_timepoint: int,
    min_z: int,
    max_z: int,
    z_increment: int,
    base_z: int,
    reference_image: Optional[str] = None,
    temporal: bool = True,
):
    """Align an entire timepoint x z-stack acquisition session."""
    input_dir = Path(input_dir)
    output_dir = io.ensure_dir(output_dir)

    ref_img = None
    if reference_image:
        ref_img = cv2.imread(str(reference_image))
        if ref_img is None:
            raise FileNotFoundError(reference_image)

    for tp in range(min_timepoint, max_timepoint + 1):
        base_src = input_dir / f"{tp}_zs+{base_z}.png"
        base_out = output_dir / f"{tp}_{base_z}.png"

        # Reference / temporal alignment of base image
        curr_base = cv2.imread(str(base_src))
        if curr_base is None:
            print(f"  missing base image for tp={tp}: {base_src}")
            continue

        wrote_base = False
        if ref_img is not None and (not temporal or tp == min_timepoint):
            aligned = align_pair(ref_img, curr_base)
            cv2.imwrite(str(base_out), aligned)
            wrote_base = True
        if temporal and tp > min_timepoint:
            prev = output_dir / f"{tp-1}_{base_z}.png"
            prev_img = cv2.imread(str(prev)) if prev.exists() else cv2.imread(str(input_dir / f"{tp-1}_zs+{base_z}.png"))
            if prev_img is not None:
                aligned = align_pair(prev_img, curr_base)
                cv2.imwrite(str(base_out), aligned)
                wrote_base = True
        if not wrote_base and not base_out.exists():
            shutil.copy(base_src, base_out)

        # Align other z-slices to the (now aligned) base
        ref_for_z = cv2.imread(str(base_out))
        for z in range(min_z, max_z + 1, z_increment):
            if z == base_z:
                continue
            src = input_dir / f"{tp}_zs+{z}.png"
            mov = cv2.imread(str(src))
            if mov is None:
                continue
            aligned = align_pair(ref_for_z, mov)
            cv2.imwrite(str(output_dir / f"{tp}_{z}.png"), aligned)


def cli(argv=None):
    parser = build_io_parser(
        "Align z-stack / timepoint photo sets using ECC. Input is the working folder of "
        "{timepoint}_zs+{z}.png files; output is the destination folder."
    )
    parser.add_argument("--min-timepoint", type=int, required=True)
    parser.add_argument("--max-timepoint", type=int, required=True)
    parser.add_argument("--min-z", type=int, required=True)
    parser.add_argument("--max-z", type=int, required=True)
    parser.add_argument("--z-increment", type=int, required=True)
    parser.add_argument("--base-z", type=int, required=True, help="Base z-stack value.")
    parser.add_argument("--reference-image", default=None,
                        help="Optional reference image to align the (first) timepoint to.")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable temporal alignment between timepoints.")
    args = parser.parse_args(argv)
    align_session(
        input_dir=args.input,
        output_dir=args.output,
        min_timepoint=args.min_timepoint,
        max_timepoint=args.max_timepoint,
        min_z=args.min_z,
        max_z=args.max_z,
        z_increment=args.z_increment,
        base_z=args.base_z,
        reference_image=args.reference_image,
        temporal=not args.no_temporal,
    )


if __name__ == "__main__":
    cli()
