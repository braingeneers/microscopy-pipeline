"""Guard tests for the canonical data representation and method exposure.

These pin the decisions made when unifying the pipeline so a future change that
breaks the contract (stacks as lists, all ops exported, complex_edf chainable)
fails loudly.
"""
import importlib

import numpy as np
import pytest

import microscopy_pipeline as mp
from microscopy_pipeline import io, ops, workflows


# --- Canonical representation -----------------------------------------------

def test_load_stack_returns_list_of_2d_arrays(tmp_path):
    frames = [np.full((8, 8), v, dtype=np.uint16) for v in (10, 20, 30)]
    path = tmp_path / "stack.tif"
    io.save_stack(frames, str(path))
    loaded = io.load_stack(str(path))
    assert isinstance(loaded, list)
    assert len(loaded) == 3
    assert all(isinstance(f, np.ndarray) and f.ndim == 2 for f in loaded)


def test_crop_stack_preserves_list_contract():
    frames = [np.zeros((10, 10), dtype=np.uint8) for _ in range(3)]
    out = ops.crop_stack(frames, left=1, top=1, right=1, bottom=1)
    assert isinstance(out, list) and len(out) == 3
    assert all(f.shape == (8, 8) for f in out)


def test_complex_edf_image_returns_single_array_not_tuple():
    stack = [(np.random.rand(64, 64) * 65535).astype(np.uint16) for _ in range(4)]
    out = ops.complex_edf_image(stack, levels=3, bit_depth=16)
    assert isinstance(out, np.ndarray)        # NOT a (fused, min_z, max_z) tuple
    assert out.ndim == 2 and out.dtype == np.uint16
    # the lower-level core still returns the informative tuple
    fused, lo, hi = ops.complex_edf(stack, levels=3, bit_depth=16)
    assert 0 <= lo <= hi <= len(stack) - 1


def test_align_stack_returns_list_same_length():
    frames = [np.full((32, 32), 30, np.uint8) for _ in range(3)]
    for f in frames:
        f[12:20, 12:20] = 200
    out = ops.align_stack(frames)
    assert isinstance(out, list) and len(out) == 3
    assert all(f.shape == (32, 32) for f in out)


# --- Method exposure / packaging --------------------------------------------

NEWLY_EXPOSED = [
    "crop_stack",
    "remove_lone_pixels_batch",
    "MaxBrightness",
    "complex_edf_image",
    "align_stack",
]


@pytest.mark.parametrize("name", NEWLY_EXPOSED)
def test_newly_exposed_symbols_importable(name):
    assert hasattr(mp, name), f"{name} not re-exported from microscopy_pipeline"
    assert name in ops.__all__, f"{name} missing from ops.__all__"


def test_all_ops_exports_resolve():
    """Every name in ops.__all__ must actually exist (no broken exports)."""
    missing = [n for n in ops.__all__ if not hasattr(ops, n)]
    assert not missing, f"broken exports: {missing}"


def test_workflows_exposed_and_callable():
    assert hasattr(mp, "workflows")
    for name in workflows.__all__:
        assert callable(getattr(workflows, name)), name


def test_every_op_module_has_cli():
    """Each op module exposes a cli() so its legacy shim keeps working."""
    import microscopy_pipeline.ops as opspkg
    import pkgutil
    no_cli = []
    for info in pkgutil.iter_modules(opspkg.__path__):
        m = importlib.import_module(f"microscopy_pipeline.ops.{info.name}")
        if not hasattr(m, "cli"):
            no_cli.append(info.name)
    assert not no_cli, f"op modules missing cli(): {no_cli}"
