"""Identify bright core + outer-ring regions in an organoid image.

This is the unified-CLI port of ``postprocessing/analysis/identifycore.py``.
The algorithm and parameters match the original; only the entry point
(``identify_core_file`` / ``identify_core_folder`` / ``cli``) is new.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure, morphology
from skimage.draw import polygon as sk_polygon

from .. import io
from ..cli import build_io_parser, configure_logging, add_folder_flags

logger = logging.getLogger(__name__)


def identify_core(
    image: np.ndarray,
    *,
    core_thresh_percentile: float = 90,
    outer_thresh_percentile: float = 40,
) -> Optional[dict]:
    """Identify (core_mask, outer_mask) and intensity statistics for a grayscale image."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = image.astype(np.uint8) if image.dtype != np.uint8 else image
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_smooth = cv2.GaussianBlur(img_norm, (3, 3), 0)
    img_core_smooth = cv2.GaussianBlur(img_norm, (9, 9), 2.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_smooth)

    # --- core ---
    core_mask = img_core_smooth > np.percentile(img_core_smooth, core_thresh_percentile)
    core_mask = morphology.remove_small_objects(core_mask, min_size=1000)
    for _ in range(3):
        core_mask = ndimage.binary_fill_holes(core_mask)
        core_mask = morphology.binary_closing(core_mask, morphology.disk(15))
    core_mask = ndimage.binary_fill_holes(core_mask)
    core_mask = morphology.binary_closing(core_mask, morphology.disk(10))
    label = measure.label(core_mask)
    props = measure.regionprops(label)
    if not props:
        return None
    largest = max(props, key=lambda x: x.area)
    core_mask = label == largest.label
    core_mask = ndimage.binary_fill_holes(core_mask)
    convexity = largest.area / largest.convex_area
    if convexity < 0.8:
        smoothed = core_mask.copy()
        for _ in range(10):
            lab = measure.label(smoothed)
            ps = measure.regionprops(lab)
            if not ps:
                break
            big = max(ps, key=lambda x: x.area)
            cv = big.area / big.convex_area
            if cv >= 0.8:
                break
            deficit = 0.8 - cv
            if deficit > 0.15:
                smoothed = morphology.binary_closing(smoothed, morphology.disk(20))
                smoothed = morphology.binary_opening(smoothed, morphology.disk(5))
            elif deficit > 0.08:
                smoothed = morphology.binary_closing(smoothed, morphology.disk(15))
                smoothed = morphology.binary_opening(smoothed, morphology.disk(3))
            else:
                smoothed = morphology.binary_closing(smoothed, morphology.disk(8))
            lab2 = measure.label(smoothed)
            ps2 = measure.regionprops(lab2)
            if ps2:
                big2 = max(ps2, key=lambda x: x.area)
                smoothed = lab2 == big2.label
            if smoothed.sum() / largest.area > 1.5:
                break
        # accept smoothed if it improved without expanding too much
        lab2 = measure.label(smoothed)
        ps2 = measure.regionprops(lab2)
        if ps2:
            big2 = max(ps2, key=lambda x: x.area)
            new_cv = big2.area / big2.convex_area
            if new_cv > convexity and big2.area / largest.area <= 1.5:
                core_mask = smoothed
        # convex hull fallback
        lab3 = measure.label(core_mask)
        ps3 = measure.regionprops(lab3)
        if ps3:
            big3 = max(ps3, key=lambda x: x.area)
            if big3.area / big3.convex_area < 0.75 and len(big3.coords) > 3:
                try:
                    hull = ConvexHull(big3.coords)
                    hull_coords = big3.coords[hull.vertices]
                    convex = np.zeros_like(core_mask)
                    rr, cc = sk_polygon(hull_coords[:, 0], hull_coords[:, 1], convex.shape)
                    convex[rr, cc] = True
                    if convex.sum() / largest.area < 1.3:
                        core_mask = convex
                except Exception:
                    pass
    else:
        core_mask = morphology.binary_closing(core_mask, morphology.disk(5))

    # --- outer ---
    low = np.percentile(img_clahe, outer_thresh_percentile)
    high = np.percentile(img_clahe, 80)
    outer_mask = (img_clahe > low) & (img_clahe < high)
    outer_mask = morphology.remove_small_objects(outer_mask, min_size=5000)
    for _ in range(5):
        outer_mask = ndimage.binary_fill_holes(outer_mask)
        outer_mask = morphology.binary_closing(outer_mask, morphology.disk(25))
    outer_mask = ndimage.binary_fill_holes(outer_mask)
    outer_mask = morphology.binary_closing(outer_mask, morphology.disk(20))
    outer_mask = morphology.binary_dilation(outer_mask, morphology.disk(5))
    outer_mask = ndimage.binary_fill_holes(outer_mask)

    combined = outer_mask | core_mask
    comb_label = measure.label(combined)
    cy, cx = measure.regionprops(measure.label(core_mask))[0].centroid
    region_id = comb_label[int(cy), int(cx)]
    if region_id == 0:
        return None
    containing = comb_label == region_id
    filled = ndimage.binary_fill_holes(containing)
    final_outer = filled & (~core_mask)

    core_area = int(core_mask.sum())
    outer_area = int(final_outer.sum())
    return {
        "core_mask": core_mask,
        "outer_mask": final_outer,
        "core_sum_intensity": float(np.sum(img[core_mask])),
        "outer_sum_intensity": float(np.sum(img[final_outer])),
        "total_image_sum_intensity": float(np.sum(img)),
        "core_area": core_area,
        "outer_area": outer_area,
        "total_pixels": int(img.size),
        "normalized_image": img_norm,
    }


def _save_overlay(result: dict, image_stem: str, output_dir: Path):
    import matplotlib.pyplot as plt
    overlay = np.stack([result["normalized_image"]] * 3, axis=-1)
    core_contours = measure.find_contours(result["core_mask"].astype(float), 0.5)
    outer_contours = measure.find_contours(result["outer_mask"].astype(float), 0.5)

    def draw(contours, color, thickness=3):
        for contour in contours:
            for y, x in contour.astype(int):
                for dy in range(-thickness, thickness + 1):
                    for dx in range(-thickness, thickness + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < overlay.shape[0] and 0 <= nx < overlay.shape[1]:
                            overlay[ny, nx] = color

    draw(outer_contours, [0, 255, 0])
    draw(core_contours, [255, 0, 0])
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"{image_stem}_contours.png", dpi=150, bbox_inches="tight")
    plt.close()


def identify_core_file(input_path, output_dir, *, core_thresh_percentile=90,
                       outer_thresh_percentile=40, save_contours: bool = True):
    output_dir = io.ensure_dir(output_dir)
    img = io.load_image(input_path, grayscale=True)
    res = identify_core(img, core_thresh_percentile=core_thresh_percentile,
                        outer_thresh_percentile=outer_thresh_percentile)
    if res is None:
        logger.warning("no regions found in %s", input_path)
        return None
    if save_contours:
        _save_overlay(res, Path(input_path).stem, Path(output_dir))
    return res


def identify_core_folder(input_dir, output_dir, *, core_thresh_percentile=90,
                         outer_thresh_percentile=40, skip_existing=False):
    """Process a folder; writes ``core_analysis_results.csv`` under ``output_dir``."""
    output_dir = io.ensure_dir(output_dir)
    results_csv = Path(output_dir) / "core_analysis_results.csv"
    if skip_existing and results_csv.exists():
        logger.info("skip (exists): %s", results_csv)
        return []
    contours_dir = io.ensure_dir(Path(output_dir) / "contours")
    outer_dir = io.ensure_dir(Path(output_dir) / "outer_regions")
    rows = []
    for src in io.list_images(input_dir):
        res = identify_core_file(str(src), contours_dir,
                                 core_thresh_percentile=core_thresh_percentile,
                                 outer_thresh_percentile=outer_thresh_percentile,
                                 save_contours=True)
        if res is None:
            continue
        # save cropped outer region
        ys, xs = np.where(res["outer_mask"])
        if ys.size:
            pad = 10
            r0, r1 = max(0, ys.min() - pad), min(res["normalized_image"].shape[0], ys.max() + pad + 1)
            c0, c1 = max(0, xs.min() - pad), min(res["normalized_image"].shape[1], xs.max() + pad + 1)
            cropped = res["normalized_image"].copy()
            cropped[~res["outer_mask"]] = 0
            cv2.imwrite(str(outer_dir / f"{src.stem}_outer_region.png"), cropped[r0:r1, c0:c1])
        total = res["total_image_sum_intensity"] or 1.0
        rows.append({
            "filename": src.name,
            "core_sum_intensity": res["core_sum_intensity"],
            "outer_sum_intensity": res["outer_sum_intensity"],
            "total_image_sum_intensity": res["total_image_sum_intensity"],
            "core_area": res["core_area"],
            "outer_area": res["outer_area"],
            "total_pixels": res["total_pixels"],
            "core_area_fraction": res["core_area"] / res["total_pixels"],
            "outer_area_fraction": res["outer_area"] / res["total_pixels"],
            "core_intensity_fraction": res["core_sum_intensity"] / total,
            "outer_intensity_fraction": res["outer_sum_intensity"] / total,
        })
    if rows:
        pd.DataFrame(rows).to_csv(Path(output_dir) / "core_analysis_results.csv", index=False)
    logger.info("processed %d images", len(rows))
    return rows


def cli(argv=None):
    parser = build_io_parser(
        "Identify bright core + outer-ring regions in an organoid image (or folder)."
    )
    parser.add_argument("--core-threshold", type=float, default=90,
                        help="Percentile threshold for core brightness (0..100).")
    parser.add_argument("--outer-threshold", type=float, default=40,
                        help="Lower percentile threshold for outer brightness (0..100).")
    add_folder_flags(parser, jobs=False)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if not (0 <= args.outer_threshold < args.core_threshold <= 100):
        parser.error("require 0 <= outer-threshold < core-threshold <= 100")
    src = Path(args.input)
    if src.is_dir():
        identify_core_folder(args.input, args.output,
                             core_thresh_percentile=args.core_threshold,
                             outer_thresh_percentile=args.outer_threshold,
                             skip_existing=args.skip_existing)
    else:
        identify_core_file(args.input, args.output,
                           core_thresh_percentile=args.core_threshold,
                           outer_thresh_percentile=args.outer_threshold)


if __name__ == "__main__":
    cli()
