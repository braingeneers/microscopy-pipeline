"""Extract creation timestamps from numbered image files into a CSV."""

from __future__ import annotations

import csv
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from .. import io
from ..cli import build_io_parser, configure_logging

logger = logging.getLogger(__name__)

_FILENAME_PATTERN = re.compile(r"^(\d+)_zs.*")


def parse_time_offset(value: str | float) -> float:
    """Parse seconds, ``MM:SS``, ``HH:MM:SS`` or ``HH:MM:SS.sss`` into seconds (signed)."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    sign = 1
    if s.startswith("-"):
        sign = -1
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    try:
        return sign * float(s)
    except ValueError:
        pass
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return sign * (int(m) * 60 + float(sec))
    if len(parts) == 3:
        h, m, sec = parts
        return sign * (int(h) * 3600 + int(m) * 60 + float(sec))
    raise ValueError(f"invalid time offset: {value!r}")


def _format_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_times(
    directory,
    *,
    time_offset: float = 0.0,
    output_csv: Optional[str] = None,
    pattern: re.Pattern = _FILENAME_PATTERN,
) -> List[dict]:
    """Walk a directory of ``{N}_zs*`` files, recording per-N creation times.

    The CSV is written next to the images by default (``file_timestamps.csv``).
    Returns the list of row dicts.
    """
    directory = Path(directory)
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    n_to_files: dict = {}
    for f in files:
        m = pattern.match(f)
        if m:
            n_to_files.setdefault(int(m.group(1)), []).append(f)

    rows: List[dict] = []
    first_time = None
    for n in sorted(n_to_files):
        first_file = sorted(n_to_files[n])[0]
        path = directory / first_file
        ctime = os.path.getctime(path)
        if first_time is None:
            first_time = ctime
        elapsed = ctime - first_time
        adjusted = elapsed + time_offset
        rows.append({
            "N": n,
            "filename": first_file,
            "actual_time": datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": _format_elapsed(adjusted),
            "elapsed_seconds": round(adjusted, 3),
            "raw_elapsed_seconds": round(elapsed, 3),
            "time_offset": time_offset,
            "raw_timestamp": ctime,
        })

    csv_path = Path(output_csv) if output_csv else directory / "file_timestamps.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info("wrote %d rows to %s", len(rows), csv_path)
    return rows


def cli(argv=None):
    parser = build_io_parser(
        "Extract file creation times from {N}_zs*.png images and write a CSV summary.",
        output_required=False,
    )
    parser.add_argument("--time-offset", default="0",
                        help="Offset added to elapsed times (seconds, MM:SS, or HH:MM:SS[.sss]; may be negative).")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if not Path(args.input).is_dir():
        parser.error(f"input must be a directory: {args.input}")
    offset = parse_time_offset(args.time_offset)
    extract_times(args.input, time_offset=offset, output_csv=args.output)


if __name__ == "__main__":
    cli()
