"""Phase 4 tests: warp modes, fuse projections, region summation, mp4/overlay."""
import importlib

import numpy as np
import pytest

from microscopy_pipeline import io, ops

t_mod = importlib.import_module("microscopy_pipeline.ops.tiff_to_gif")


# --- P4.1 alignment warp mode ----------------------------------------------

def test_align_stack_euclidean(drift_frames):
    import cv2
    out = ops.align_stack(drift_frames, warp_mode=cv2.MOTION_EUCLIDEAN)
    assert isinstance(out, list) and len(out) == len(drift_frames)
    assert all(f.shape == drift_frames[0].shape for f in out)


# --- P4.2 fuse projection ---------------------------------------------------

def test_fuse_projection_max_exceeds_mean():
    frames = [np.full((8, 8), 10, np.uint16), np.full((8, 8), 50, np.uint16)]
    mx = ops.fuse_project(frames, projection="max")
    mn = ops.fuse_project(frames, projection="mean")
    assert mx.mean() > mn.mean()
    assert np.array_equal(ops.fuse_average(frames), mn)  # alias == mean projection


def test_fuse_projection_invalid():
    with pytest.raises(ValueError):
        ops.fuse_project([np.zeros((4, 4), np.uint16)], projection="bogus")


# --- P4.3 region-restricted summation --------------------------------------

def test_sum_brightness_mask_and_background():
    img = np.full((10, 10), 100, np.uint16)
    mask = np.zeros((10, 10), bool)
    mask[2:5, 2:5] = True  # 9 pixels
    assert ops.sum_brightness(img, mask=mask) == 100 * 9
    assert ops.sum_brightness(img, mask=mask) < ops.sum_brightness(img)
    assert ops.sum_brightness(img, background=100) == 0.0


def test_sum_fluorescence_folder_with_mask(tmp_path):
    fdir = tmp_path / "f"
    fdir.mkdir()
    for n in range(1, 4):
        io.save_image(np.full((16, 16), 1000 * n, np.uint16), str(fdir / f"img_{n}_z.tif"))
    mpath = tmp_path / "mask.png"
    m = np.zeros((16, 16), np.uint8)
    m[4:8, 4:8] = 255  # 4x4 = 16 pixels
    io.save_image(m, str(mpath))
    rows = ops.sum_fluorescence_folder(str(fdir), str(tmp_path / "out.csv"), mask=str(mpath))
    assert len(rows) == 3
    assert all("masked_area" in r for r in rows)
    assert rows[0]["masked_area"] == 16


# --- P4.4 mp4 export + label overlay ---------------------------------------

def test_gif_label_overlay(tmp_path):
    frames = [np.full((20, 60), v, np.uint16) for v in (1000, 50000)]
    out = tmp_path / "m.gif"
    t_mod.frames_to_gif(frames, str(out), labels=["A", "B"])
    assert out.exists() and out.stat().st_size > 0


def test_frames_to_video_writes_or_guards(tmp_path):
    frames = [np.full((16, 16), v, np.uint16) for v in (1000, 50000)]
    out = tmp_path / "m.mp4"
    try:
        import imageio  # noqa: F401
    except Exception:
        # No imageio at all -> clear dependency-guard error.
        with pytest.raises(RuntimeError):
            t_mod.frames_to_video(frames, str(out))
        return
    # imageio present: the fps fallback must produce a file even without ffmpeg.
    # (Do NOT swallow exceptions here -- that previously masked a real fps= bug.)
    t_mod.frames_to_video(frames, str(out), fps=5)
    assert out.exists() and out.stat().st_size > 0
