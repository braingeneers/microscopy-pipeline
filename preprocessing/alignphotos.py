"""Thin CLI shim. Implementation lives in microscopy_pipeline.ops.align."""
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here.parent, *_here.parents):
    if (_p / "microscopy_pipeline").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from microscopy_pipeline.ops.align import cli

if __name__ == "__main__":
    cli()