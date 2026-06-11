"""Sum pixel intensities (or inverted intensities for brightfield) across images."""

from __future__ import annotations

import csv
import re
from glob import glob
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from .. import io
from ..cli import build_io_parser

_INDEX_PATTERN = re.compile(r".*_(\d+)_.*\.tiff?$", re.IGNORECASE)


def sum_brightness(image: np.ndarray, *, brightfield: bool = False,
                   max_value: Optional[float] = None) -> float:
    """Sum pixel values; in brightfield mode invert by ``max_value`` first."""
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
    return float(np.sum(arr))


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
):
    """Sum brightness for every ``*_N_*.tif(f)`` file and write a CSV."""
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
        value = sum_brightness(arr, brightfield=brightfield, max_value=max_v)
        row = {"index": index, "filename": path.name, column: value}
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
    print(f"wrote {len(rows)} rows to {output_csv}")
    return rows


def cli(argv=None):
    parser = build_io_parser(
        "Sum pixel brightness (or inverted brightness for brightfield) for *_N_*.tif files."
    )
    parser.add_argument("--brightfield", action="store_true",
                        help="Invert pixel values before summing (optical density mode).")
    parser.add_argument("--timestamps", default=None,
                        help="Optional CSV with N and actual_time columns to merge into output.")
    args = parser.parse_args(argv)
    sum_fluorescence_folder(args.input, args.output,
                            brightfield=args.brightfield, timestamps=args.timestamps)


if __name__ == "__main__":
    cli()
