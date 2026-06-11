"""End-to-end smoke tests for the task-to-task workflows.

Each test drives one pipeline from synthetic input to its final artifact and
asserts the artifact exists / has the expected shape, dtype or row count.
"""
import numpy as np

from microscopy_pipeline import io, ops, workflows


# --- Workflow 1: focal stack -> annotated GIF -------------------------------

def test_focal_stack_to_gif_edf_in_memory(tmp_path, focal_stacks):
    out = tmp_path / "movie.gif"
    result = workflows.focal_stack_to_gif(
        focal_stacks, out, method="edf", levels=3, channel="green",
        scale_um=20, pixel_size_um=1.0,
    )
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_focal_stack_to_gif_average_from_pngs(tmp_path, focal_pngs):
    out = tmp_path / "movie.gif"
    workflows.focal_stack_to_gif(focal_pngs, out, method="average")
    assert out.exists() and out.stat().st_size > 0


# --- Workflow 2: PNG slices -> stacks -> fused stills -----------------------

def test_pngs_to_fused_images_average(tmp_path, focal_pngs):
    odir = tmp_path / "fused"
    outs = workflows.pngs_to_fused_images(focal_pngs, odir, method="average")
    assert len(outs) == 2                       # one per timepoint X
    assert all(p.exists() for p in outs)
    # intermediate stacks cleaned up by default
    assert not (odir / "_stacks").exists()


def test_pngs_to_fused_images_edf_keep_stacks(tmp_path, focal_pngs):
    odir = tmp_path / "fused_edf"
    outs = workflows.pngs_to_fused_images(focal_pngs, odir, method="edf", keep_stacks=True)
    assert len(outs) == 2 and all(p.exists() for p in outs)
    assert (odir / "_stacks").exists()
    fused = io.load_image(str(outs[0]), grayscale=True)
    assert fused.ndim == 2


# --- Workflow 3: register -> crop -> brightness -> colorize ------------------

def test_align_crop_colorize(tmp_path, drift_frames):
    odir = tmp_path / "colorized"
    results = workflows.align_crop_colorize(
        drift_frames, crop=(4, 4, 4, 4), channel="red", output_dir=odir,
    )
    assert len(results) == len(drift_frames)
    # 64x64 cropped 4px on each side -> 56x56, colorized to RGB uint8
    assert results[0].shape == (56, 56, 3)
    assert results[0].dtype == np.uint8
    assert len(list(odir.glob("frame_*.png"))) == len(drift_frames)


# --- Workflow 4: brightfield -> organoid growth graphs ----------------------

def test_brightfield_to_growth_graphs(tmp_path, brightfield_dir):
    odir = tmp_path / "bf_out"
    res = workflows.brightfield_to_growth_graphs(brightfield_dir, odir, pixel_size_um=0.5)
    rows = res["rows"]
    assert len(rows) == 4
    assert sum(1 for r in rows if r.get("contour_found")) == 4
    assert res["csv"].exists()
    for fig in ("organoid_area_over_time.png", "organoid_combined_analysis.png",
                "organoid_analysis_summary.txt"):
        assert (odir / fig).exists(), fig
    # organoid area should grow with timepoint
    areas = [r["area_pixels"] for r in sorted(rows, key=lambda r: r["timepoint"])]
    assert areas[-1] > areas[0]


def test_tracker_accepts_png_inputs(tmp_path, brightfield_dir):
    """Regression: track_organoid_folder must accept PNGs, not just .tif/.tiff."""
    pdir = tmp_path / "bf_png"
    pdir.mkdir()
    for src in brightfield_dir.glob("*.tif"):
        img = io.load_image(str(src), grayscale=True)
        io.save_image(img, str(pdir / (src.stem + ".png")))
    rows = ops.track_organoid_folder(str(pdir), str(tmp_path / "bf_png_out"))
    assert len(rows) == 4


# --- Workflow 5: fluorescence -> summed-intensity trend ---------------------

def test_fluorescence_to_curve(tmp_path, fluorescence_dir):
    csv = tmp_path / "sums.csv"
    svg = tmp_path / "curve.svg"
    res = workflows.fluorescence_to_curve(fluorescence_dir, csv, svg, window_size=3)
    rows = res["rows"]
    assert len(rows) == 8
    assert "sum_brightness" in rows[0]
    # frames were written with rising intensity -> sums should rise
    assert rows[-1]["sum_brightness"] > rows[0]["sum_brightness"]
    assert csv.exists()
    assert svg.exists() and svg.stat().st_size > 0


def test_fluorescence_to_curve_brightfield(tmp_path, fluorescence_dir):
    csv = tmp_path / "od.csv"
    svg = tmp_path / "od.svg"
    res = workflows.fluorescence_to_curve(fluorescence_dir, csv, svg,
                                          brightfield=True, window_size=3)
    assert "sum_optical_density" in res["rows"][0]
    assert svg.exists() and svg.stat().st_size > 0
