"""Sum pixel intensities (or inverted intensities for brightfield) across images."""

from __future__ import annotations

import csv
import logging
import re
from glob import glob
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from .. import io
from ..cli import build_io_parser, configure_logging, add_folder_flags

logger = logging.getLogger(__name__)

_INDEX_PATTERN = re.compile(r".*_(\d+)_.*\.tiff?$", re.IGNORECASE)


def sum_brightness(image: np.ndarray, *, brightfield: bool = False,
                   max_value: Optional[float] = None,
                   mask: Optional[np.ndarray] = None,
                   background: Optional[float] = None) -> float:
    """Sum pixel values.

    In brightfield mode pixels are inverted by ``max_value`` first.  ``background``
    (if given) is subtracted from every pixel (clipped at 0) before summing.  A
    boolean ``mask`` (same H/W as the image) restricts the sum to its True pixels.
    """
    arr = image.astype(np.float64)
    if brightfield:
        if max_value is None:
            if image.dtype == np.uint16:
                max_value = 65535
            elif image.dtype == np.uint8:
                max_value = 255
            else:
                max_value = float(arr.max())
        arr = max_value - arr
    if background:
        arr = np.clip(arr - float(background), 0, None)
    if mask is not None:
        m = np.asarray(mask)
        if m.shape[:2] != arr.shape[:2]:
            raise ValueError(f"mask shape {m.shape[:2]} != image {arr.shape[:2]}")
        return float(np.sum(arr[m.astype(bool)]))
    return float(np.sum(arr))


def _organoid_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """Boolean mask of the detected organoid (via the brightfield tracker), or None."""
    import cv2
    from .brightfield_organoid_tracker import track_organoid
    res = track_organoid(image)
    contour = res.get("contour")
    if contour is None:
        return None
    m = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [contour], 1)
    return m.astype(bool)


def _load_timestamps(path) -> dict:
    out: dict = {}
    if not path:
        return out
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if "N" not in reader.fieldnames or "actual_time" not in reader.fieldnames:
            raise ValueError("timestamps CSV must have 'N' and 'actual_time' columns")
        for row in reader:
            try:
                out[int(row["N"])] = row["actual_time"]
            except (ValueError, KeyError):
                continue
    return out


def sum_fluorescence_folder(
    input_dir,
    output_csv,
    *,
    brightfield: bool = False,
    timestamps: Optional[str] = None,
    skip_existing: bool = False,
    mask: Optional[str] = None,
    region: str = "whole",
    background: Optional[float] = None,
):
    """Sum brightness for every ``*_N_*.tif(f)`` file and write a CSV.

    ``region='organoid'`` derives a per-frame mask from the brightfield tracker;
    ``mask`` restricts every frame to a single binary mask image (white = keep).
    """
    if skip_existing and Path(output_csv).exists():
        logger.info("skip (exists): %s", output_csv)
        return []
    files = sorted(
        glob(str(Path(input_dir) / "*_*_*.tif")) + glob(str(Path(input_dir) / "*_*_*.tiff"))
    )
    matches = []
    for f in files:
        m = _INDEX_PATTERN.match(Path(f).name)
        if m:
            matches.append((int(m.group(1)), Path(f)))
    matches.sort()

    timestamp_map = _load_timestamps(timestamps)
    static_mask = (io.load_image(mask, grayscale=True) > 127) if mask else None
    use_mask = static_mask is not None or region == "organoid"
    rows = []
    column = "sum_optical_density" if brightfield else "sum_brightness"
    for index, path in matches:
        with Image.open(path) as img:
            mode = img.mode
            if mode in ("I;16", "I;16B", "I;16L"):
                max_v = 65535
            elif mode == "I":
                arr_tmp = np.asarray(img)
                max_v = 65535 if arr_tmp.max() <= 65535 else (2**32 - 1)
            elif mode == "F":
                max_v = 1.0
            else:
                max_v = 255
            arr = np.asarray(img)
        frame_mask = _organoid_mask(arr) if region == "organoid" else static_mask
        value = sum_brightness(arr, brightfield=brightfield, max_value=max_v,
                               mask=frame_mask, background=background)
        row = {"index": index, "filename": path.name, column: value}
        if use_mask:
            row["masked_area"] = int(np.count_nonzero(frame_mask)) if frame_mask is not None else 0
        if timestamp_map:
            row["actual_time"] = timestamp_map.get(index, "")
        rows.append(row)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    logger.info("wrote %d rows to %s", len(rows), output_csv)
    return rows


def cli(argv=None):
    parser = build_io_parser(
        "Sum pixel brightness (or inverted brightness for brightfield) for *_N_*.tif files."
    )
    parser.add_argument("--brightfield", action="store_true",
                        help="Invert pixel values before summing (optical density mode).")
    parser.add_argument("--timestamps", default=None,
                        help="Optional CSV with N and actual_time columns to merge into output.")
    parser.add_argument("--mask", default=None,
                        help="Path to a binary mask; sum only within white (>127) pixels.")
    parser.add_argument("--region", choices=("whole", "organoid"), default="whole",
                        help="Restrict summation: whole image or the detected organoid region.")
    parser.add_argument("--background", type=float, default=None,
                        help="Subtract this constant from each pixel before summing (clipped at 0).")
    add_folder_flags(parser, jobs=False)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if args.mask and args.region != "whole":
        parser.error("use either --mask or --region organoid, not both")
    sum_fluorescence_folder(args.input, args.output,
                            brightfield=args.brightfield, timestamps=args.timestamps,
                            skip_existing=args.skip_existing, mask=args.mask,
                            region=args.region, background=args.background)


if __name__ == "__main__":
    cli()
