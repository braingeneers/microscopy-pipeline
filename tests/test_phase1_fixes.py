"""Regression tests for the Phase 1 correctness fixes.

* complex_edf must handle focal stacks with >32 z-slices (np.choose -> take_along_axis).
* tiff_to_gif must support global / per-frame / none normalization.
* the dead helpers (io.batch_apply, cli.dispatch_file_or_folder) must be gone.
"""
import importlib

import numpy as np
import pytest

from microscopy_pipeline import ops

# the package re-exports the `tiff_to_gif` function, shadowing the submodule, so
# import the module unambiguously to reach its private helpers.
t = importlib.import_module("microscopy_pipeline.ops.tiff_to_gif")


# --- complex_edf: 32-frame ceiling lifted -----------------------------------

def test_take_along_axis_matches_choose_for_small_stacks():
    """Mechanism equivalence: the new gather equals np.choose for N<=32."""
    rng = np.random.RandomState(2)
    stack = rng.rand(5, 16, 16)
    idx = rng.randint(0, 5, (16, 16))
    assert np.array_equal(np.choose(idx, stack),
                          np.take_along_axis(stack, idx[None], axis=0)[0])


def test_complex_edf_handles_more_than_32_frames():
    rng = np.random.RandomState(0)
    frames = [rng.randint(0, 65535, (64, 64)).astype(np.uint16) for _ in range(40)]
    fused, lo, hi = ops.complex_edf(frames, levels=3, bit_depth=16)
    assert fused.shape == (64, 64)
    assert fused.dtype == np.float64
    assert 0 <= lo <= hi < 40
    img = ops.complex_edf_image(frames, levels=3, bit_depth=16)
    assert img.shape == (64, 64) and img.dtype == np.uint16
    # top_n>1 and invert exercise the same take_along_axis lines
    f2, _, _ = ops.complex_edf(frames, levels=3, top_n=3, invert=True)
    assert f2.shape == (64, 64)


def test_complex_edf_deterministic():
    rng = np.random.RandomState(1)
    frames = [rng.randint(0, 65535, (32, 32)).astype(np.uint16) for _ in range(5)]
    a, _, _ = ops.complex_edf(frames, levels=2)
    b, _, _ = ops.complex_edf(frames, levels=2)
    assert np.array_equal(a, b)


# --- tiff_to_gif normalization ----------------------------------------------

def test_gif_global_normalization_preserves_relative_brightness():
    base = np.zeros((8, 8), np.uint16)
    base[2:6, 2:6] = 40000
    dim = (base.astype(np.float32) * 0.25).astype(np.uint16)  # same pattern, dimmer
    g = t._normalize_frames([base, dim], "global")
    assert g[1].max() < g[0].max()           # dim stays dimmer under a shared window
    p = t._normalize_frames([base, dim], "per-frame")
    assert p[0].max() == 255 and p[1].max() == 255   # each stretched to full range (flicker)


def test_gif_normalize_none_passes_uint8_through():
    a = np.array([[0, 17, 255]], np.uint8)
    assert np.array_equal(t._normalize_frames([a], "none")[0], a)


def test_gif_invalid_normalize_raises():
    with pytest.raises(ValueError):
        t._normalize_frames([np.zeros((2, 2), np.uint16)], "bogus")


def test_frames_to_gif_writes_file(tmp_path):
    frames = [np.full((8, 8), v, np.uint16) for v in (1000, 50000)]
    out = tmp_path / "movie.gif"
    t.frames_to_gif(frames, str(out))
    assert out.exists() and out.stat().st_size > 0


# --- dead code is gone -------------------------------------------------------

def test_dead_helpers_removed():
    from microscopy_pipeline import io, cli
    assert not hasattr(io, "batch_apply")
    assert not hasattr(cli, "dispatch_file_or_folder")
