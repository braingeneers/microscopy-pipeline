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
import logging
from typing import Optional


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
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (repeat -v for debug-level output).",
    )
    return parser


_LOGGER_NAME = "microscopy_pipeline"


def configure_logging(verbosity: int = 0) -> logging.Logger:
    """Configure the package logger for CLI use.

    No ``-v`` shows INFO-level progress messages; ``-v`` enables DEBUG.  When the
    package is imported as a library this is never called, so the logger stays
    unconfigured and quiet (standard logging practice).
    """
    level = logging.DEBUG if verbosity and verbosity >= 1 else logging.INFO
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def add_folder_flags(parser: argparse.ArgumentParser, *, jobs: bool = True) -> argparse.ArgumentParser:
    """Add ``--skip-existing`` (and, unless ``jobs=False``, ``--jobs``) for folder ops.

    Aggregate ops that write a single ordered CSV pass ``jobs=False`` (they stay
    serial to keep deterministic row order).
    """
    if jobs:
        parser.add_argument(
            "--jobs", type=int, default=1,
            help="Parallel worker processes for folder mode (default 1 = serial).",
        )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip inputs whose output already exists (resume an interrupted run).",
    )
    return parser
