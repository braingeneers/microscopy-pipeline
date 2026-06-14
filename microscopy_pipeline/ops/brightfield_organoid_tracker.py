"""Detect organoid contours in brightfield images and track area / darkness over time."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .. import io
from ..cli import build_io_parser, configure_logging, add_folder_flags

logger = logging.getLogger(__name__)

_TIMEPOINT_PATTERNS = [
    re.compile(r"_(\d+)$"),
    re.compile(r"t(\d+)", re.IGNORECASE),
    re.compile(r"timepoint[_\s]*(\d+)", re.IGNORECASE),
    re.compile(r"tp(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)_"),
    re.compile(r"(\d+)"),
]


def _extract_timepoint(name: str) -> int:
    stem = Path(name).stem
    for rx in _TIMEPOINT_PATTERNS:
        m = rx.search(stem)
        if m:
            return int(m.group(1))
    return 0


def find_organoid_contour(
    image: np.ndarray,
    *,
    min_area: int = 1000,
    gaussian_blur: int = 5,
    threshold_method: str = "otsu",
    threshold_value: int = 127,
    morphology_kernel: int = 5,
    morphology_iterations: int = 2,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size=(8, 8),
    otsu_adjustment: int = 10,
):
    """Return ``(contour, area, binary_image)`` for the largest dark-on-bright object."""
    proc = image.copy()
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    proc = clahe.apply(proc)
    if gaussian_blur > 0:
        proc = cv2.GaussianBlur(proc, (gaussian_blur, gaussian_blur), 0)
    if threshold_method == "otsu":
        otsu_t, _ = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        permissive = max(0, otsu_t - otsu_adjustment)
        _, binary = cv2.threshold(proc, permissive, 255, cv2.THRESH_BINARY_INV)
    elif threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    else:
        _, binary = cv2.threshold(proc, threshold_value, 255, cv2.THRESH_BINARY_INV)

    if morphology_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morphology_iterations)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morphology_iterations)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return None, 0, binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    indices = np.arange(1, n_labels)
    valid = areas >= min_area
    if not np.any(valid):
        return None, 0, binary
    largest = indices[valid][np.argmax(areas[valid])]
    component = (labels == largest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, component
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour, float(cv2.contourArea(largest_contour)), component


def _to_8bit_scaled(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16 or arr.max() > 255:
        arr = (arr // 256).astype(np.uint8)
    arr = arr.astype(np.uint8)
    lo, hi = arr.min(), arr.max()
    if lo == hi:
        return np.full_like(arr, 128)
    return ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)


def track_organoid(image: np.ndarray, **detection_params) -> dict:
    """Detect contour, return area + darkness statistics + the contour points."""
    original = image.copy()
    scaled = _to_8bit_scaled(image)
    contour, area, _ = find_organoid_contour(scaled, **detection_params)
    if contour is None:
        return {"contour_found": False, "area_pixels": 0.0, "perimeter": 0.0,
                "centroid": (0, 0), "bounding_box": (0, 0, 0, 0),
                "width": 0, "height": 0, "aspect_ratio": 0.0, "circularity": 0.0,
                "total_darkness": 0.0, "average_darkness": 0.0,
                "min_darkness": 0.0, "max_darkness": 0.0,
                "raw_intensity_mean": 0.0, "raw_intensity_min": 0.0, "raw_intensity_max": 0.0,
                "contour": None}
    perimeter = float(cv2.arcLength(contour, True))
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] else 0
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    pixels = original[mask > 0]
    if len(pixels):
        inv = 255 - pixels.astype(np.float32)
        total_d = float(inv.sum()); avg_d = float(inv.mean())
        min_d = float(inv.min()); max_d = float(inv.max())
        rim = float(pixels.mean()); rmin = float(pixels.min()); rmax = float(pixels.max())
    else:
        total_d = avg_d = min_d = max_d = rim = rmin = rmax = 0.0
    return {
        "contour_found": True,
        "area_pixels": float(area),
        "perimeter": perimeter,
        "centroid": (cx, cy),
        "bounding_box": (x, y, w, h),
        "width": w, "height": h,
        "aspect_ratio": w / h if h else 0.0,
        "circularity": (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0.0,
        "total_darkness": total_d, "average_darkness": avg_d,
        "min_darkness": min_d, "max_darkness": max_d,
        "raw_intensity_mean": rim, "raw_intensity_min": rmin, "raw_intensity_max": rmax,
        "contour": contour,
    }


def track_organoid_file(
    input_path, output_path,
    *,
    contour_color=(0, 255, 0), contour_thickness: int = 2,
    **detection_params,
) -> dict:
    img = io.load_image(input_path, grayscale=True)
    res = track_organoid(img, **detection_params)
    scaled = _to_8bit_scaled(img)
    overlay = cv2.cvtColor(scaled, cv2.COLOR_GRAY2RGB)
    if res["contour_found"]:
        cv2.drawContours(overlay, [res["contour"]], -1, contour_color, contour_thickness)
        cv2.circle(overlay, res["centroid"], 5, (255, 0, 0), -1)
        x, y, w, h = res["bounding_box"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    res.pop("contour", None)
    return res


def track_organoid_folder(
    input_dir, output_dir,
    *,
    pixel_size_um: Optional[float] = None,
    contour_color=(0, 255, 0), contour_thickness: int = 2,
    skip_existing: bool = False,
    **detection_params,
):
    """Process a folder of brightfield images; writes detailed + summary CSVs."""
    output_dir = io.ensure_dir(output_dir)
    results_csv = Path(output_dir) / "organoid_analysis_results.csv"
    if skip_existing and results_csv.exists():
        logger.info("skip (exists): %s", results_csv)
        return []
    # Route through io.list_images so the tracker accepts whatever its
    # predecessors emit (PNG/TIFF/...), not just .tif/.tiff.  This is what lets
    # a clahe/crop/scale_brightness -> tracker chain actually carry data.
    files = list(io.list_images(input_dir))
    files.sort(key=lambda p: (_extract_timepoint(p.name), p.name))
    detailed = []
    for src in files:
        out = Path(output_dir) / f"{src.stem}_contour{src.suffix}"
        timepoint = _extract_timepoint(src.name)
        res = track_organoid_file(str(src), str(out),
                                  contour_color=contour_color,
                                  contour_thickness=contour_thickness,
                                  **detection_params)
        res["filename"] = src.name
        res["timepoint"] = timepoint
        res["output_filename"] = out.name
        ps = pixel_size_um if pixel_size_um is not None else io.read_pixel_size_um(str(src))
        if ps and res["area_pixels"] > 0:
            res["area_um2"] = res["area_pixels"] * ps ** 2
            res["perimeter_um"] = res["perimeter"] * ps
        else:
            res["area_um2"] = 0.0
            res["perimeter_um"] = 0.0
        detailed.append(res)

    if detailed:
        fieldnames = ["timepoint", "filename", "output_filename", "contour_found",
                      "area_pixels", "area_um2", "perimeter", "perimeter_um",
                      "centroid", "bounding_box", "width", "height",
                      "aspect_ratio", "circularity", "total_darkness", "average_darkness",
                      "min_darkness", "max_darkness", "raw_intensity_mean",
                      "raw_intensity_min", "raw_intensity_max"]
        with open(Path(output_dir) / "organoid_analysis_results.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(detailed)
        with open(Path(output_dir) / "organoid_areas_summary.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timepoint", "filename", "area_pixels", "area_um2",
                             "total_darkness", "average_darkness", "raw_intensity_mean",
                             "contour_found"])
            for r in detailed:
                writer.writerow([r["timepoint"], r["filename"], r["area_pixels"], r["area_um2"],
                                 r["total_darkness"], r["average_darkness"],
                                 r["raw_intensity_mean"], r["contour_found"]])
    return detailed


def cli(argv=None):
    parser = build_io_parser(
        "Detect organoid contours in brightfield TIFFs and track area/darkness over time."
    )
    parser.add_argument("--pixel-size-um", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--gaussian-blur", type=int, default=5)
    parser.add_argument("--threshold-method", choices=("otsu", "adaptive", "manual"), default="otsu")
    parser.add_argument("--threshold-value", type=int, default=127)
    parser.add_argument("--morphology-kernel", type=int, default=5)
    parser.add_argument("--morphology-iterations", type=int, default=2)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)
    parser.add_argument("--otsu-adjustment", type=int, default=10)
    add_folder_flags(parser, jobs=False)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    params = dict(
        min_area=args.min_area,
        gaussian_blur=args.gaussian_blur,
        threshold_method=args.threshold_method,
        threshold_value=args.threshold_value,
        morphology_kernel=args.morphology_kernel,
        morphology_iterations=args.morphology_iterations,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=(args.clahe_tile_grid, args.clahe_tile_grid),
        otsu_adjustment=args.otsu_adjustment,
    )
    src = Path(args.input)
    if src.is_dir():
        track_organoid_folder(args.input, args.output,
                              pixel_size_um=args.pixel_size_um,
                              skip_existing=args.skip_existing, **params)
    else:
        track_organoid_file(args.input, args.output, **params)


if __name__ == "__main__":
    cli()
