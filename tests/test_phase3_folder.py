"""Phase 3 tests: parallel --jobs equivalence and --skip-existing resume."""
import numpy as np

from microscopy_pipeline import io, ops


def _make_pngs(d, n=4):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        io.save_image(np.full((16, 16), 1000 * (i + 1), np.uint16), str(d / f"{i}_0.png"))
    return d


def test_crop_folder_jobs_matches_serial(tmp_path):
    """jobs=2 (ProcessPoolExecutor) must produce byte-identical output to serial."""
    src = _make_pngs(tmp_path / "in")
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    ops.crop_folder(str(src), str(o1), left=2, top=2, right=2, bottom=2, jobs=1)
    ops.crop_folder(str(src), str(o2), left=2, top=2, right=2, bottom=2, jobs=2)
    names = sorted(p.name for p in o1.iterdir())
    assert names == sorted(p.name for p in o2.iterdir())
    for n in names:
        assert np.array_equal(io.load_image(str(o1 / n)), io.load_image(str(o2 / n)))


def test_skip_existing_does_not_rewrite(tmp_path):
    src = _make_pngs(tmp_path / "in")
    out = tmp_path / "o"
    ops.scale_brightness_folder(str(src), str(out), factor=1.0)
    first = {p.name: io.load_image(str(p)).copy() for p in out.iterdir()}
    # A second run at factor=2.0 would double brightness, but --skip-existing
    # must leave the already-written files untouched.
    ops.scale_brightness_folder(str(src), str(out), factor=2.0, skip_existing=True)
    for name, arr in first.items():
        assert np.array_equal(io.load_image(str(out / name)), arr)
    # sanity: without the flag, factor=2.0 does change at least one output
    ops.scale_brightness_folder(str(src), str(out), factor=2.0)
    assert any(not np.array_equal(io.load_image(str(out / n)), a) for n, a in first.items())
