"""
pipeline_log.py — shared logging setup for all pipeline scripts.

Usage in any script:
    from pipeline_log import get_logger
    log = get_logger("scalecolors")
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that sends INFO+ to stdout and WARNING+ to stderr,
    both with a consistent timestamp/level prefix.

    Having INFO on stdout and warnings/errors on stderr means a pipeline
    or cloud agent can read progress on one stream and detect failures on
    the other (e.g. shell redirection: 2>errors.log).
    """
    log = logging.getLogger(name)

    # Don't add handlers more than once if get_logger is called repeatedly
    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)  # handlers filter; logger itself passes everything

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(fmt)
    log.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)
    log.addHandler(stderr_handler)

    return log
