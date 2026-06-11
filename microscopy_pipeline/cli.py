"""Shared CLI helpers.

Every operation script accepts the same core flags so that pipelines feel
consistent::

    -i, --input   PATH     Input file or directory.
    -o, --output  PATH     Output file or directory.

Op-specific flags use ``--kebab-case`` names matching the keyword argument of
the array-level function.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional


def build_io_parser(
    description: str,
    *,
    input_required: bool = True,
    output_required: bool = True,
    epilog: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Create an ``ArgumentParser`` pre-populated with ``-i`` / ``-o``."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=input_required,
        help="Input file or directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=output_required,
        help="Output file or directory.",
    )
    return parser


def dispatch_file_or_folder(
    input_path: str,
    output_path: str,
    *,
    file_func: Callable[[str, str], None],
    folder_func: Callable[[str, str], None],
) -> None:
    """Call ``file_func`` if input is a file, ``folder_func`` if it's a folder."""
    p = Path(input_path)
    if p.is_dir():
        folder_func(input_path, output_path)
    elif p.is_file():
        file_func(input_path, output_path)
    else:
        raise FileNotFoundError(f"Input does not exist: {input_path}")
