"""Phase 3 tests: TIFF pixel-size (resolution tag) read/preserve + consumer fallback."""
import numpy as np
import pytest

from microscopy_pipeline import io, ops


def test_pixel_size_roundtrip(tmp_path):
    frames = [np.full((8, 8), 1000, np.uint16), np.full((8, 8), 2000, np.uint16)]
    p = tmp_path / "s.tif"
    io.save_stack(frames, str(p), pixel_size_um=0.5)
    got = io.read_pixel_size_um(str(p))
    assert got is not None and abs(got - 0.5) < 1e-2


def test_pixel_size_absent_returns_none(tmp_path):
    frames = [np.full((8, 8), 1000, np.uint16), np.full((8, 8), 1000, np.uint16)]
    p = tmp_path / "s2.tif"
    io.save_stack(frames, str(p))  # no pixel size written
    assert io.read_pixel_size_um(str(p)) is None


def test_scale_bar_falls_back_to_metadata(tmp_path):
    frame = np.zeros((40, 200, 3), np.uint8)
    p = tmp_path / "img.tif"
    io.save_stack([frame], str(p), pixel_size_um=0.5)
    out = tmp_path / "out.png"
    # no pixel_size_um passed -> must read 0.5 from metadata instead of raising
    ops.add_scale_bar_file(str(p), str(out), scale_um=20)
    assert out.exists()


def test_scale_bar_without_metadata_raises(tmp_path):
    frame = np.zeros((40, 200, 3), np.uint8)
    p = tmp_path / "img2.tif"
    io.save_stack([frame], str(p))  # no pixel size
    out = tmp_path / "out2.png"
    with pytest.raises(ValueError):
        ops.add_scale_bar_file(str(p), str(out), scale_um=20)
