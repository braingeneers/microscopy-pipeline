import argparse
import os
import sys
import re
import glob
import numpy as np
from PIL import Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pipeline_log import get_logger

log = get_logger("pngs_to_tiff")

# File discovery and grouping

def parse_filename(filename):
    """
    Parse filenames in the format X_Y.png.
    Returns (X, Y) as integers, or None if the filename doesn't match.
    X is the stack group (timepoint); Y is the slice index (z-stack position).
    """
    match = re.match(r'^(\d+)_(-?\d+)\.png$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def group_files_by_x(png_dir):
    """
    Scan png_dir for files matching X_Y.png and group them by X value.
    Returns dict: X (int) -> sorted list of (Y, filepath) tuples.
    """
    png_files = glob.glob(os.path.join(png_dir, '*.png'))
    files_by_x = defaultdict(list)
    skipped = 0

    for png_file in png_files:
        parsed = parse_filename(os.path.basename(png_file))
        if parsed:
            x, y = parsed
            files_by_x[x].append((y, png_file))
        else:
            log.warning("Skipping file (name does not match X_Y.png): %s",
                        os.path.basename(png_file))
            skipped += 1

    for x in files_by_x:
        files_by_x[x].sort(key=lambda item: item[0])

    log.info("FILE SCAN | total_png=%d matched=%d skipped=%d groups=%d",
             len(png_files),
             len(png_files) - skipped,
             skipped,
             len(files_by_x))
    return files_by_x


# Bit-depth detection

def get_bit_depth(img):
    """Return 8 or 16 based on the PIL image mode."""
    if img.mode in ('I;16', 'I;16B', 'I;16L'):
        return 16
    if img.mode == 'I' and hasattr(img, 'tag') and img.tag.get(258):
        bps = img.tag[258]
        return bps[0] if isinstance(bps, (list, tuple)) else bps
    return 8

# ImageJ-compatible TIFF metadata

def imagej_tiff_tags(num_slices, bit_depth):
    """Return a tiffinfo dict with ImageJ-compatible metadata."""
    max_val = 65535.0 if bit_depth == 16 else 255.0
    description = (
        f"ImageJ=1.53t\nimages={num_slices}\nchannels=1\n"
        f"slices={num_slices}\nframes=1\nhyperstack=false\n"
        f"mode=grayscale\nloop=false\nmin=0.0\nmax={max_val}\n"
    )
    return {
        50839: description.encode('utf-8'),   # ImageJ metadata
        305:   "ImageJ",                       # Software
        269:   f"Stack ({num_slices} slices)", # Document name
        270:   description,                    # Image description
    }



# Single-stack creation (one unit of parallel work)


def create_stack(args):
    """Build a multi-page TIFF stack for one X value and write it to output_dir.
    Designed to be called by a thread pool.
    Returns (x_value, success).  """

    x_value, y_files, output_dir, index, total = args

    log.info("[%d/%d] STACK START | x=%d slices=%d", index, total, x_value, len(y_files))

    if not y_files:
        log.warning("[%d/%d] STACK SKIP | x=%d reason=no_files", index, total, x_value)
        return x_value, False

    try:
        frames = []
        bit_depth = None

        for slice_idx, (y, filepath) in enumerate(y_files):
            img = Image.open(filepath)

            if bit_depth is None:
                bit_depth = get_bit_depth(img)

            if bit_depth == 16:
                if img.mode in ('I;16', 'I;16B', 'I;16L'):
                    frames.append(img.copy())
                elif img.mode == 'I':
                    frames.append(img.point(lambda v: min(v, 65535)).convert('I;16'))
                elif img.mode == 'L':
                    frames.append(img.point(lambda v: v * 257).convert('I;16'))
                else:
                    frames.append(img.convert('L').point(lambda v: v * 257).convert('I;16'))
            else:
                if img.mode in ('I;16', 'I;16B', 'I;16L'):
                    frames.append(img.point(lambda v: v // 257).convert('L'))
                elif img.mode != 'L':
                    frames.append(img.convert('L'))
                else:
                    frames.append(img.copy())

            img.close()

        output_path = os.path.join(output_dir, f"stack_{x_value}.tiff")
        tiff_tags = imagej_tiff_tags(len(frames), bit_depth)
        compression = None if bit_depth == 16 else 'tiff_lzw'

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            compression=compression,
            tiffinfo=tiff_tags,
            resolution_unit=1,
            resolution=(72.0, 72.0),
        )

        w, h = frames[0].size
        log.info("[%d/%d] STACK END | x=%d status=ok slices=%d size=%dx%d depth=%dbit path=%s",
                 index, total, x_value, len(frames), w, h, bit_depth, output_path)
        return x_value, True

    except Exception as e:
        log.error("[%d/%d] STACK FAIL | x=%d error=%s", index, total, x_value, e)
        return x_value, False

# Batch orchestration

def convert_all(png_dir, output_dir, max_workers=None, parallel=True):
    """Convert all X_Y.png groups in png_dir into per-X TIFF stacks."""
    os.makedirs(output_dir, exist_ok=True)

    files_by_x = group_files_by_x(png_dir)
    if not files_by_x:
        log.error("No valid X_Y.png files found in: %s", png_dir)
        sys.exit(1)

    x_values = sorted(files_by_x.keys())
    total = len(x_values)
    workers = max_workers or min(total, multiprocessing.cpu_count())

    log.info("BATCH START | stacks=%d input=%s output=%s workers=%s",
             total,
             os.path.abspath(png_dir),
             os.path.abspath(output_dir),
             workers if parallel else "sequential")

    work_items = [
        (x, files_by_x[x], output_dir, i, total)
        for i, x in enumerate(x_values, 1)
    ]

    successful, failed = 0, 0

    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(create_stack, work_items))
    else:
        results = [create_stack(item) for item in work_items]

    for _, success in results:
        if success:
            successful += 1
        else:
            failed += 1

    status = "complete" if failed == 0 else "complete_with_errors"
    log.info("BATCH END | status=%s success=%d failed=%d output=%s",
             status, successful, failed, os.path.abspath(output_dir))

    if failed > 0:
        log.warning("STATUS: %s | %d stack(s) failed — check errors above", status, failed)
        sys.exit(1)

# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Convert PNG image sets (named X_Y.png) into per-X ImageJ-compatible TIFF stacks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
"""EX
  Basic conversion (output alongside input):
    python pngs_to_tiff.py --input ./pngs --output ./stacks

  Sequential mode:
    python pngs_to_tiff.py --input ./pngs --output ./stacks --no-parallel

  Limit workers:
    python pngs_to_tiff.py --input ./pngs --output ./stacks --workers 4

FILENAME FORMAT
  Files must be named X_Y.png where:
    X = stack group (positive integer, e.g. timepoint)
    Y = slice index (positive or negative integer, e.g. z-stack position)
  Examples: 1_0.png  1_5.png  2_-10.png  2_15.png

OUTPUT
  One TIFF stack per X value: stack_<X>.tiff
  Slices are ordered by ascending Y value.
  Bit depth is auto-detected from the source PNGs."""
    )

    parser.add_argument("--input",  metavar="DIR", required=True,
                        help="Directory containing PNG files named X_Y.png")
    parser.add_argument("--output", metavar="DIR", required=True,
                        help="Directory to write TIFF stacks into")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing and build stacks sequentially")

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        parser.error(f"Input directory not found: {args.input}")

    convert_all(
        args.input,
        args.output,
        max_workers=args.workers,
        parallel=not args.no_parallel,
    )


if __name__ == "__main__":
    main()