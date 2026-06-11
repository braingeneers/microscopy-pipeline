"""Shared synthetic fixtures for the smoke tests.

Every fixture builds tiny in-memory or on-disk data so the workflow tests run in
a few seconds with no external files.  A headless matplotlib backend is forced
here (before any ``pyplot`` import) so the plotting ops work in CI.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import cv2
import pytest

from microscopy_pipeline import io


def make_disk(h, w, cy, cx, r, val, bg=0, dtype=np.uint16):
    """A filled disk of value ``val`` on a ``bg`` background."""
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    arr = np.full((h, w), bg, dtype=dtype)
    arr[mask] = val
    return arr


@pytest.fixture
def focal_stacks():
    """Two timepoints x three z-slices (64x64 uint16); disk sharpest at z=1."""
    stacks = []
    for tp in range(2):
        base = make_disk(64, 64, 32, 32 + tp * 2, 12, val=50000, bg=2000, dtype=np.uint16)
        frames = []
        for z in range(3):
            blur = (5, 1, 5)[z]
            f = cv2.GaussianBlur(base, (blur, blur), 0) if blur > 1 else base.copy()
            frames.append(f.astype(np.uint16))
        stacks.append(frames)
    return stacks


@pytest.fixture
def focal_pngs(tmp_path, focal_stacks):
    """Write ``focal_stacks`` to ``{X}_{Y}.png`` files; return the folder."""
    pdir = tmp_path / "pngs"
    pdir.mkdir()
    for x, frames in enumerate(focal_stacks):
        for y, frame in enumerate(frames):
            io.save_image(frame, str(pdir / f"{x}_{y}.png"))
    return pdir


@pytest.fixture
def drift_frames():
    """Four 64x64 uint8 frames of a disk drifting in x (simulated stage drift)."""
    return [make_disk(64, 64, 32, 30 + i * 2, 12, val=200, bg=30, dtype=np.uint8)
            for i in range(4)]


@pytest.fixture
def brightfield_dir(tmp_path):
    """Four brightfield frames (dark disk on bright bg) as ``frame_N.tif``."""
    bdir = tmp_path / "brightfield"
    bdir.mkdir()
    for tp, r in enumerate((20, 22, 24, 26)):
        img = make_disk(80, 80, 40, 40, r, val=60, bg=220, dtype=np.uint8)
        io.save_image(img, str(bdir / f"frame_{tp}.tif"))
    return bdir


@pytest.fixture
def fluorescence_dir(tmp_path):
    """Eight single-frame TIFFs named ``img_N_z.tif`` with rising intensity."""
    fdir = tmp_path / "fluor"
    fdir.mkdir()
    for n in range(1, 9):
        img = make_disk(32, 32, 16, 16, 6 + n, val=1000 * n + 5000, bg=100, dtype=np.uint16)
        io.save_image(img, str(fdir / f"img_{n}_z.tif"))
    return fdir
