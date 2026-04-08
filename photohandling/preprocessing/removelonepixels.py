import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pipeline_log import get_logger
 
log = get_logger("removelonepixels")
 
 
# ---------------------------------------------------------------------------
# File grouping
# ---------------------------------------------------------------------------
 
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
 
def group_images_by_prefix(folder):
    """
    Group image files by their filename prefix (text before the first underscore).
    Returns dict: prefix -> sorted list of filenames.
    """
    if not os.path.isdir(folder):
        log.error("Input folder not found: %s", folder)
        sys.exit(1)
 
    groups = defaultdict(list)
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            prefix = fname.split('_')[0] if '_' in fname else os.path.splitext(fname)[0]
            groups[prefix].append(fname)
 
    log.info("GROUPS FOUND | count=%d input=%s", len(groups), folder)
    for prefix, files in sorted(groups.items()):
        log.info("  group=%s files=%d", prefix, len(files))
 
    return groups
 
 
# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
 
def load_images(folder, filenames):
    """
    Load a list of image files as grayscale NumPy arrays.
    Returns (arrays, loaded_filenames).
    """
    images, loaded = [], []
    for fname in filenames:
        try:
            img = Image.open(os.path.join(folder, fname)).convert('L')
            images.append(np.array(img))
            loaded.append(fname)
        except Exception as e:
            log.warning("Could not load %s: %s", fname, e)
 
    log.info("  Loaded %d/%d images", len(images), len(filenames))
    return images, loaded
 
 
# ---------------------------------------------------------------------------
# Lone-pixel detection (fully vectorized)
# ---------------------------------------------------------------------------
 
def find_lone_pixels(images, threshold=0.2, min_fraction=0.2):
    """
    Find pixels that are brighter than all 8 neighbors by at least `threshold`
    fraction in at least `min_fraction` of the images in the group.
 
    Returns a boolean mask (h, w).
    """
    if not images:
        return np.array([])
 
    h, w = images[0].shape
    n = len(images)
    log.info("  Detection | images=%d size=%dx%d threshold=%.0f%% min_fraction=%.0f%%",
             n, h, w, threshold * 100, min_fraction * 100)
 
    image_stack = np.stack([img.astype(np.float32) for img in images], axis=0)  # (n, h, w)
    padded = np.pad(image_stack, ((0, 0), (1, 1), (1, 1)), mode='edge')         # (n, h+2, w+2)
 
    center = padded[:, 1:-1, 1:-1]  # (n, h, w)
 
    neighbors = np.stack([
        padded[:, 0:-2, 1:-1],  # up
        padded[:, 2:,   1:-1],  # down
        padded[:, 1:-1, 0:-2],  # left
        padded[:, 1:-1, 2:  ],  # right
        padded[:, 0:-2, 0:-2],  # up-left
        padded[:, 0:-2, 2:  ],  # up-right
        padded[:, 2:,   0:-2],  # down-left
        padded[:, 2:,   2:  ],  # down-right
    ], axis=0)  # (8, n, h, w)
 
    # Pixel is "lone" in an image if it beats ALL 8 neighbors by the threshold
    is_lone = np.all(center[np.newaxis] > neighbors * (1 + threshold), axis=0)  # (n, h, w)
 
    count = np.sum(is_lone, axis=0)  # (h, w)
    mask = count >= int(n * min_fraction)
 
    log.info("  Lone pixels found | count=%d fraction=%.4f%%",
             int(mask.sum()), float(mask.sum()) / (h * w) * 100)
    return mask
 
 
# ---------------------------------------------------------------------------
# Lone-pixel correction (vectorized)
# ---------------------------------------------------------------------------
 
def correct_images(images, lone_pixel_mask, output_folder, filenames):
    """
    Replace lone pixels with the mean of their 8 neighbors.
    Uses vectorized NumPy operations — no per-pixel Python loops.
    Saves corrected images to output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
 
    if not np.any(lone_pixel_mask):
        log.info("  No lone pixels to correct — copying images unchanged")
 
    # Build neighbor-average array once (same mask applies to all images in group)
    for idx, (img, fname) in enumerate(zip(images, filenames)):
        arr = img.astype(np.float32)
        padded = np.pad(arr, 1, mode='edge')
 
        # Average of 8 neighbors at every position, vectorized
        neighbor_sum = (
            padded[0:-2, 1:-1] + padded[2:,   1:-1] +
            padded[1:-1, 0:-2] + padded[1:-1, 2:  ] +
            padded[0:-2, 0:-2] + padded[0:-2, 2:  ] +
            padded[2:,   0:-2] + padded[2:,   2:  ]
        )
        neighbor_avg = neighbor_sum / 8.0
 
        corrected = arr.copy()
        corrected[lone_pixel_mask] = neighbor_avg[lone_pixel_mask]
 
        out_img = Image.fromarray(corrected.astype(np.uint8))
        out_img.save(os.path.join(output_folder, fname))
        log.info("  [%d/%d] OK | file=%s corrections=%d",
                 idx + 1, len(images), fname, int(lone_pixel_mask.sum()))
 
 
# ---------------------------------------------------------------------------
# Group processing (one unit of parallel work)
# ---------------------------------------------------------------------------
 
def process_group(args):
    """
    Process a single prefix group: load → detect → correct.
    Designed to be called by a thread/process pool.
    Returns (group_prefix, success, elapsed_seconds).
    """
    input_folder, output_folder, group_prefix, filenames, threshold, min_fraction, index, total = args
 
    log.info("[%d/%d] GROUP START | group=%s files=%d",
             index, total, group_prefix, len(filenames))
    t0 = time.time()
 
    try:
        images, loaded_filenames = load_images(input_folder, filenames)
        if not images:
            log.warning("[%d/%d] GROUP SKIP | group=%s reason=no_images_loaded",
                        index, total, group_prefix)
            return group_prefix, False, 0.0
 
        mask = find_lone_pixels(images, threshold, min_fraction)
        correct_images(images, mask, output_folder, loaded_filenames)
 
        elapsed = time.time() - t0
        log.info("[%d/%d] GROUP END | group=%s status=ok elapsed=%.2fs",
                 index, total, group_prefix, elapsed)
        return group_prefix, True, elapsed
 
    except Exception as e:
        elapsed = time.time() - t0
        log.error("[%d/%d] GROUP FAIL | group=%s error=%s elapsed=%.2fs",
                  index, total, group_prefix, e, elapsed)
        return group_prefix, False, elapsed
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Remove persistent hot/lone pixels from image sets grouped by filename prefix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Basic usage:
    python removelonepixels.py --input ./images --output ./corrected
 
  Custom thresholds:
    python removelonepixels.py --input ./images --output ./corrected --threshold 0.3 --min-fraction 0.1
 
  Resume from a specific prefix:
    python removelonepixels.py --input ./images --output ./corrected --start-prefix 50
 
  Limit parallel workers:
    python removelonepixels.py --input ./images --output ./corrected --workers 4
 
HOW IT WORKS
  Images are grouped by the part of their filename before the first underscore
  (e.g. "042_zs+3.tif" → group "042"). Within each group, pixels that are
  consistently brighter than all 8 neighbors across at least --min-fraction of
  images are identified as stuck/hot pixels and replaced with the local mean.
        """
    )
 
    parser.add_argument("--input",  metavar="DIR", required=True,
                        help="Folder containing input images")
    parser.add_argument("--output", metavar="DIR", required=True,
                        help="Folder to save corrected images")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Fractional brightness excess above neighbors to flag a pixel (default: 0.2)")
    parser.add_argument("--min-fraction", type=float, default=0.2,
                        help="Fraction of images in a group where pixel must be lone to be corrected (default: 0.2)")
    parser.add_argument("--start-prefix", metavar="PREFIX", default=None,
                        help="Skip groups before this prefix (useful for resuming interrupted runs)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for group processing (default: CPU count)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing and run groups sequentially")
 
    args = parser.parse_args()
 
    # Validate
    if not (0 < args.threshold <= 1):
        parser.error("--threshold must be between 0 (exclusive) and 1")
    if not (0 < args.min_fraction <= 1):
        parser.error("--min-fraction must be between 0 (exclusive) and 1")
 
    # Group files
    image_groups = group_images_by_prefix(args.input)
    if not image_groups:
        log.error("No image groups found in: %s", args.input)
        sys.exit(1)
 
    # Sort groups numerically where possible
    def sort_key(prefix):
        try:
            return (0, int(prefix))
        except ValueError:
            return (1, prefix)
 
    sorted_groups = sorted(image_groups.items(), key=lambda kv: sort_key(kv[0]))
 
    # Apply start-prefix filter
    if args.start_prefix:
        try:
            sp_int = int(args.start_prefix)
            sorted_groups = [(p, f) for p, f in sorted_groups
                             if sort_key(p) >= sort_key(args.start_prefix)]
        except ValueError:
            sorted_groups = [(p, f) for p, f in sorted_groups if p >= args.start_prefix]
 
        if not sorted_groups:
            log.error("No groups found at or after prefix: %s", args.start_prefix)
            sys.exit(1)
        log.info("Resuming from prefix=%s | remaining_groups=%d", args.start_prefix, len(sorted_groups))
 
    total = len(sorted_groups)
    workers = args.workers or min(total, multiprocessing.cpu_count())
 
    log.info("BATCH START | groups=%d input=%s output=%s threshold=%.0f%% min_fraction=%.0f%% workers=%s",
             total,
             os.path.abspath(args.input),
             os.path.abspath(args.output),
             args.threshold * 100,
             args.min_fraction * 100,
             workers if not args.no_parallel else "sequential")
 
    work_items = [
        (args.input, args.output, prefix, filenames,
         args.threshold, args.min_fraction, i, total)
        for i, (prefix, filenames) in enumerate(sorted_groups, 1)
    ]
 
    t_overall = time.time()
    successful, failed = 0, 0
 
    if args.no_parallel:
        results = [process_group(item) for item in work_items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(process_group, work_items))
 
    for _, success, _ in results:
        if success:
            successful += 1
        else:
            failed += 1
 
    elapsed = time.time() - t_overall
    status = "complete" if failed == 0 else "complete_with_errors"
    log.info("BATCH END | status=%s success=%d failed=%d elapsed=%.2fs output=%s",
             status, successful, failed, elapsed, os.path.abspath(args.output))
 
    if failed > 0:
        log.warning("STATUS: %s | %d group(s) failed — check errors above", status, failed)
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()