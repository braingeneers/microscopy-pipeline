"""OME-TIFF I/O: pixel-size round-trip, save_stack auto-routing, the to_ome_tiff op."""
import numpy as np
import pytest

from microscopy_pipeline import io, ops

tifffile = pytest.importorskip("tifffile")


def test_save_ome_tiff_roundtrip_pixel_size(tmp_path):
    frames = [np.full((8, 8), v, np.uint16) for v in (100, 200, 300)]
    p = tmp_path / "stack.ome.tif"
    io.save_ome_tiff(frames, str(p), pixel_size_um=0.65)
    meta = io.read_ome_metadata(str(p))
    assert meta["pixel_size_um"] is not None and abs(meta["pixel_size_um"] - 0.65) < 1e-3
    assert meta["shape"] == (3, 8, 8)
    # frames read back via the normal PIL-based loader
    loaded = io.load_stack(str(p))
    assert len(loaded) == 3 and loaded[0].shape == (8, 8)
    assert int(loaded[2].max()) == 300


def test_read_pixel_size_um_from_ome(tmp_path):
    p = tmp_path / "img.ome.tif"
    io.save_ome_tiff([np.zeros((4, 4), np.uint16)], str(p), pixel_size_um=1.25)
    assert abs(io.read_pixel_size_um(str(p)) - 1.25) < 1e-3


def test_save_stack_auto_routes_to_ome(tmp_path):
    p = tmp_path / "s.ome.tif"
    io.save_stack([np.full((4, 4), 7, np.uint16)] * 2, str(p), pixel_size_um=0.5)
    with tifffile.TiffFile(str(p)) as tif:
        assert tif.is_ome
    assert abs(io.read_pixel_size_um(str(p)) - 0.5) < 1e-3


def test_to_ome_tiff_file_copies_source_metadata(tmp_path):
    src = tmp_path / "src.tif"  # plain TIFF with resolution-tag pixel size
    io.save_stack([np.full((6, 6), 10, np.uint16), np.full((6, 6), 20, np.uint16)],
                  str(src), pixel_size_um=0.8)
    out = tmp_path / "out.ome.tif"
    ops.to_ome_tiff_file(str(src), str(out))  # no pixel size -> copied from source
    assert abs(io.read_ome_metadata(str(out))["pixel_size_um"] - 0.8) < 1e-2


def test_save_ome_tiff_rgb_preserved(tmp_path):
    # an RGB image must stay RGB, not become a 3-slice grayscale Z-stack
    rgb = np.zeros((8, 12, 3), np.uint8)
    rgb[..., 1] = 200
    p = tmp_path / "rgb.ome.tif"
    io.save_ome_tiff(rgb, str(p), pixel_size_um=0.5)
    with tifffile.TiffFile(str(p)) as tif:
        assert tif.series[0].shape == (8, 12, 3)  # NOT (3, 8, 12)
    back = io.load_image(str(p))
    assert back.shape == (8, 12, 3)
    assert int(back[..., 1].max()) == 200 and int(back[..., 0].max()) == 0


def test_to_ome_tiff_folder(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    for i in range(3):
        io.save_image(np.full((5, 5), 100 * (i + 1), np.uint16), str(d / f"f{i}.tif"))
    outdir = tmp_path / "out"
    ops.to_ome_tiff_folder(str(d), str(outdir), pixel_size_um=0.5)
    outs = sorted(outdir.glob("*.ome.tif"))
    assert len(outs) == 3
    assert all(io.read_ome_metadata(str(p))["pixel_size_um"] for p in outs)
