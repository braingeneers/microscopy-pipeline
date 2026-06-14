"""Unified task-to-task pipelines built from :mod:`microscopy_pipeline.ops`.

Each function here chains several operations into one named pipeline so a whole
acquisition can be processed with a single call instead of running the legacy
shell scripts one at a time.  The functions follow the canonical data
representation documented in :mod:`microscopy_pipeline.io`:

* a **single image** is an ``np.ndarray`` (2-D grayscale or ``(H, W, 3)`` RGB);
* a **stack** is a ``List[np.ndarray]`` (a list of 2-D frames).

Where ops support in-memory cores the data is chained array-to-array without
touching disk; the disk-oriented analysis ops (organoid tracking, fluorescence
summing, plotting) are wrapped as folder-in / files-out stages.

Pipeline map (stage -> stage)::

    raw PNGs ─▶ align_stack ─▶ crop ─▶ scale_brightness ─▶ grey_to_color   (align_crop_colorize)
    {X}_{Y}.png ─▶ pngs_to_tiff_stacks ─▶ fuse_average / complex_edf       (pngs_to_fused_images)
    focal stacks ─▶ fuse/EDF ─▶ scale_brightness ─▶ clahe
                   ─▶ grey_to_color ─▶ add_scale_bar ─▶ GIF                (focal_stack_to_gif)
    brightfield TIFFs ─▶ track_organoid_folder ─▶ graph_contour_data       (brightfield_to_growth_graphs)
    fluorescence TIFFs ─▶ (extract_times) ─▶ sum_fluorescence ─▶ plot      (fluorescence_to_curve)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import io, ops

PathLike = Union[str, Path]

__all__ = [
    "focal_stack_to_gif",
    "pngs_to_fused_images",
    "align_crop_colorize",
    "brightfield_to_growth_graphs",
    "fluorescence_to_curve",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_png_stacks(input_dir: PathLike) -> List[Tuple[int, List[np.ndarray]]]:
    """Load a folder of ``{X}_{Y}.png`` files as one in-memory stack per X.

    Returns ``[(x, [frame, ...]), ...]`` sorted by X, each frame a 2-D array
    (frames within a group are ordered by Y).
    """
    groups = ops.group_pngs_by_x(input_dir)
    out: List[Tuple[int, List[np.ndarray]]] = []
    for x in sorted(groups):
        frames = [io.load_image(str(p), grayscale=True) for _, p in groups[x]]
        out.append((x, frames))
    return out


def _fuse_stack(frames: Sequence[np.ndarray], *, method: str, bit_depth: int,
                wavelet: str, levels: int, top_n: int, invert: bool) -> np.ndarray:
    """Collapse a focal stack to a single 2-D image via EDF or mean projection."""
    if method == "edf":
        return ops.complex_edf_image(frames, wavelet=wavelet, levels=levels,
                                     bit_depth=bit_depth, top_n=top_n, invert=invert)
    if method == "average":
        return ops.fuse_average(frames, bit_depth=bit_depth)
    raise ValueError("method must be 'edf' or 'average'")


# ---------------------------------------------------------------------------
# Workflow 1 -- focal stack -> annotated animated GIF (mostly in-memory)
# ---------------------------------------------------------------------------

def focal_stack_to_gif(
    stacks: Union[PathLike, Sequence[Sequence[np.ndarray]]],
    output_path: PathLike,
    *,
    method: str = "edf",
    wavelet: str = "db3",
    levels: int = 3,
    top_n: int = 1,
    invert: bool = False,
    bit_depth: int = 16,
    brightness: float = 1.0,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    channel: str = "green",
    gamma: float = 1.0,
    scale_um: Optional[float] = None,
    pixel_size_um: Optional[float] = None,
    bar_height: int = 3,
    bar_margin: int = 5,
    duration_ms: int = 100,
    loop: int = 0,
    normalize: str = "global",
) -> Path:
    """Turn a per-timepoint focal-stack time-series into an annotated GIF.

    For every timepoint the focal stack is fused (extended-depth-of-field with
    ``method='edf'`` or a mean projection with ``method='average'``), brightness
    scaled, CLAHE-enhanced, colorized into one RGB channel, and -- if
    ``scale_um`` and ``pixel_size_um`` are given -- annotated with a scale bar.
    The resulting per-timepoint frames are written as one animated GIF.

    Parameters
    ----------
    stacks
        Either a directory of ``{X}_{Y}.png`` files (X = timepoint, Y = z-slice)
        or an already-loaded sequence of focal stacks (each a list of 2-D arrays).
    output_path
        Destination ``.gif`` path.

    Returns the ``Path`` to the written GIF.
    """
    if isinstance(stacks, (str, Path)):
        loaded = [frames for _x, frames in _load_png_stacks(stacks)]
    else:
        loaded = [list(s) for s in stacks]
    if not loaded:
        raise ValueError(f"no focal stacks found in {stacks!r}")

    frames: List[np.ndarray] = []
    for stack in loaded:
        fused = _fuse_stack(stack, method=method, bit_depth=bit_depth,
                            wavelet=wavelet, levels=levels, top_n=top_n, invert=invert)
        if brightness != 1.0:
            fused = ops.scale_brightness(fused, factor=brightness)
        enhanced = ops.clahe(fused, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        rgb = ops.grey_to_color(enhanced, channel=channel, gamma=gamma)
        if scale_um is not None and pixel_size_um is not None:
            rgb = ops.add_scale_bar(rgb, scale_um=scale_um, pixel_size_um=pixel_size_um,
                                    bar_height=bar_height, margin=bar_margin)
        frames.append(rgb)

    output_path = Path(output_path)
    if output_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        fps = max(1, round(1000.0 / duration_ms)) if duration_ms else 10
        ops.frames_to_video(frames, str(output_path), fps=fps, normalize=normalize)
    else:
        ops.frames_to_gif(frames, str(output_path), duration_ms=duration_ms, loop=loop,
                          normalize=normalize)
    return output_path


# ---------------------------------------------------------------------------
# Workflow 2 -- PNG slices -> TIFF stacks -> fused stills (disk bridge)
# ---------------------------------------------------------------------------

def pngs_to_fused_images(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    method: str = "average",
    bit_depth: Optional[int] = None,
    wavelet: str = "db3",
    levels: int = 3,
    top_n: int = 1,
    invert: bool = False,
    keep_stacks: bool = False,
) -> List[Path]:
    """Bundle ``{X}_{Y}.png`` slices into per-X TIFF stacks, then fuse each.

    Stage 1 groups the PNGs into ImageJ-compatible ``stack_{X}.tiff`` files
    (:func:`ops.pngs_to_tiff_stacks`).  Stage 2 collapses every stack to a single
    image -- a mean projection (``method='average'``) or complex-wavelet EDF
    (``method='edf'``).  Returns the list of fused output paths.
    """
    output_dir = io.ensure_dir(output_dir)
    stack_dir = io.ensure_dir(Path(output_dir) / "_stacks")
    stack_paths = ops.pngs_to_tiff_stacks(input_dir, stack_dir, bit_depth=bit_depth)

    outputs: List[Path] = []
    for sp in stack_paths:
        out = Path(output_dir) / f"{sp.stem}_fused.tif"
        if method == "average":
            ops.fuse_average_file(str(sp), str(out))
        elif method == "edf":
            ops.complex_edf_file(str(sp), str(out), wavelet=wavelet, levels=levels,
                                 bit_depth=bit_depth or 16, top_n=top_n, invert=invert)
        else:
            raise ValueError("method must be 'edf' or 'average'")
        outputs.append(out)

    if not keep_stacks:
        for sp in stack_paths:
            Path(sp).unlink(missing_ok=True)
        try:
            stack_dir.rmdir()
        except OSError:
            pass
    return outputs


# ---------------------------------------------------------------------------
# Workflow 3 -- register -> crop -> brightness-correct -> colorize (in-memory)
# ---------------------------------------------------------------------------

def align_crop_colorize(
    frames: Sequence[np.ndarray],
    *,
    reference: Optional[np.ndarray] = None,
    crop: Tuple[int, int, int, int] = (0, 0, 0, 0),
    auto_brightness: bool = True,
    brightness: float = 1.0,
    channel: str = "red",
    gamma: float = 1.0,
    output_dir: Optional[PathLike] = None,
    warp_mode: Optional[int] = None,
) -> List[np.ndarray]:
    """Register a frame series, crop borders, normalize brightness, colorize.

    ``frames`` is a stack (``List[np.ndarray]``) of 8-bit grayscale images (e.g.
    a z-series or a timepoint series) that are registered to ``reference``
    (default: the first frame) via ECC.  Each aligned frame is cropped, brightness
    scaled, and colorized into one RGB channel.  With ``auto_brightness`` the
    scale factor per frame is chosen from :func:`ops.find_max_brightness` so the
    brightest pixel maps to full range; otherwise the fixed ``brightness`` factor
    is used.

    ``crop`` is ``(left, top, right, bottom)`` pixels.  If ``output_dir`` is
    given, each RGB result is also written as ``frame_NNN.png``.  Returns the list
    of RGB arrays.
    """
    _align_kw = {} if warp_mode is None else {"warp_mode": warp_mode}
    aligned = ops.align_stack(frames, reference=reference, **_align_kw)
    left, top, right, bottom = crop
    out_dir = io.ensure_dir(output_dir) if output_dir is not None else None

    results: List[np.ndarray] = []
    for i, frame in enumerate(aligned):
        cropped = ops.crop(frame, left=left, top=top, right=right, bottom=bottom)
        if auto_brightness:
            mb = ops.find_max_brightness(cropped)
            factor = (1.0 / mb.ratio) if mb.ratio > 0 else 1.0
        else:
            factor = brightness
        scaled = ops.scale_brightness(cropped, factor=factor)
        rgb = ops.grey_to_color(scaled, channel=channel, gamma=gamma)
        results.append(rgb)
        if out_dir is not None:
            io.save_image(rgb, str(Path(out_dir) / f"frame_{i:03d}.png"))
    return results


# ---------------------------------------------------------------------------
# Workflow 4 -- brightfield series -> organoid growth graphs (disk)
# ---------------------------------------------------------------------------

def brightfield_to_growth_graphs(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    pixel_size_um: Optional[float] = None,
    plots: str = "all",
    dpi: int = 300,
    **detection_params,
) -> Dict[str, object]:
    """Track organoids across a brightfield series and plot growth over time.

    Stage 1 (:func:`ops.track_organoid_folder`) segments the organoid in each
    frame and writes ``organoid_analysis_results.csv`` plus overlay images.
    Stage 2 (:func:`ops.graph_contour_data`) turns that CSV into area / darkness /
    combined figures and a text summary.  Extra ``detection_params`` (e.g.
    ``min_area``, ``threshold_method``, ``clahe_clip_limit``) pass through to the
    tracker.

    Returns ``{"rows", "csv", "figures_dir"}``.
    """
    output_dir = io.ensure_dir(output_dir)
    rows = ops.track_organoid_folder(str(input_dir), str(output_dir),
                                     pixel_size_um=pixel_size_um, **detection_params)
    csv_path = Path(output_dir) / "organoid_analysis_results.csv"
    if csv_path.exists():
        ops.graph_contour_data(str(csv_path), str(output_dir),
                               pixel_size=pixel_size_um, plots=plots, dpi=dpi)
    return {"rows": rows, "csv": csv_path, "figures_dir": Path(output_dir)}


# ---------------------------------------------------------------------------
# Workflow 5 -- fluorescence series -> summed-intensity trend (disk)
# ---------------------------------------------------------------------------

def fluorescence_to_curve(
    input_dir: PathLike,
    output_csv: PathLike,
    plot_path: PathLike,
    *,
    brightfield: bool = False,
    timestamps: Optional[PathLike] = None,
    window_size: int = 5,
    time_interval_min: Optional[float] = None,
    confidence_level: float = 0.95,
) -> Dict[str, object]:
    """Sum per-frame intensity over a TIFF series and plot a smoothed trend.

    Stage 1 (:func:`ops.sum_fluorescence_folder`) totals the pixel intensity of
    every ``*_N_*.tif`` frame into a CSV (optical density if ``brightfield``);
    an optional ``timestamps`` CSV (columns ``N``, ``actual_time``, e.g. from
    :func:`ops.extract_times`) is merged in.  Stage 2 (:func:`ops.plot_fluorescence`)
    draws the points, a smoothed trend line and a confidence band as an SVG.

    Returns ``{"rows", "csv", "plot"}``.
    """
    rows = ops.sum_fluorescence_folder(
        str(input_dir), str(output_csv),
        brightfield=brightfield,
        timestamps=str(timestamps) if timestamps is not None else None,
    )
    ops.plot_fluorescence(
        str(output_csv), str(plot_path),
        window_size=window_size,
        time_interval_min=time_interval_min,
        confidence_level=confidence_level,
        brightfield=brightfield,
    )
    return {"rows": rows, "csv": Path(output_csv), "plot": Path(plot_path)}
