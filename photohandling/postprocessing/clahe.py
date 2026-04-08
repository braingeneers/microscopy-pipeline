import cv2
import sys
import os
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from pipeline_log import get_logger
log = get_logger("clahe")


def apply_clahe_single(input_path, output_path, clip_limit=2.0,
                       tile_grid_size=(8, 8), channel_only=None):
    """Apply CLAHE to a single image, optionally restricted to one RGB channel."""
    if channel_only is not None:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            log.error("Unable to read image: %s", input_path)
            return False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        channel_map = {'red': 0, 'green': 1, 'blue': 2}
        channel_index = channel_map[channel_only]
        target_channel = img_rgb[:, :, channel_index]

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_channel = clahe.apply(target_channel)

        img_rgb_clahe = img_rgb.copy()
        img_rgb_clahe[:, :, channel_index] = enhanced_channel
        img_output = cv2.cvtColor(img_rgb_clahe, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            log.error("Unable to read image: %s", input_path)
            return False
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_output = clahe.apply(img)

    success = cv2.imwrite(output_path, img_output)
    if not success:
        log.error("Failed to save image to: %s", output_path)
        return False

    return True


# Parallel batch processing 


def _worker(args):
    img_file, output_file, clip_limit, tile_grid_size, channel_only, index, total = args
    fname = img_file.name
    log.info("[%d/%d] START | file=%s", index, total, fname)

    try:
        if channel_only:
            img_temp = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
            if img_temp is not None:
                log.info("    shape=%s", img_temp.shape)
        else:
            img_temp = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img_temp is not None:
                log.info("    shape=%s", img_temp.shape)

        success = apply_clahe_single(
            str(img_file), str(output_file),
            clip_limit, tile_grid_size, channel_only)

        if success:
            log.info("[%d/%d] OK    | file=%s", index, total, fname)
        else:
            log.error("[%d/%d] FAIL  | file=%s", index, total, fname)
        return success, fname

    except Exception as e:
        log.error("[%d/%d] FAIL  | file=%s error=%s", index, total, fname, e)
        return False, fname

def batch_process_folder(input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8, 8),
                         channel_only=None, max_workers=None, parallel=True):
    """Apply CLAHE to all images in a folder."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif',
                        '.bmp', '.webp', '.jp2'}
    image_files = sorted(
        f for ext in image_extensions
        for f in list(input_path.glob(f'*{ext}')) + list(input_path.glob(f'*{ext.upper()}'))
    )
    image_files = sorted(set(image_files))

    if not image_files:
        log.warning("No supported image files found in: %s", input_dir)
        return

    total = len(image_files)
    workers = max_workers or min(total, multiprocessing.cpu_count())
    mode_str = f"{channel_only} channel" if channel_only else "grayscale"

    log.info("BATCH START | files=%d input=%s output=%s mode=%s grid=%s clip=%.1f workers=%s",
             total, input_dir, output_dir, mode_str, tile_grid_size, clip_limit,
             workers if parallel else "sequential")

    work_items = [
        (f, output_path / f.name, clip_limit, tile_grid_size, channel_only, i, total)
        for i, f in enumerate(image_files, 1)
    ]

    successful, failed = 0, 0

    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for success, _ in executor.map(_worker, work_items):
                if success: successful += 1
                else: failed += 1
    else:
        for item in work_items:
            success, _ = _worker(item)
            if success: successful += 1
            else: failed += 1

    status = "complete" if failed == 0 else "complete_with_errors"
    log.info("BATCH END | status=%s success=%d failed=%d output=%s",
             status, successful, failed, output_dir)
    if failed > 0:
        log.warning("STATUS: %s | %d file(s) failed", status, failed)


# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) "
                    "to a single image or all images in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Single file:
    python clahe.py --input image.jpg --output enhanced.jpg
    python clahe.py --input image.jpg --output enhanced.jpg --red-channel --grid 16 --clip 3.0

  Folder (parallel by default):
    python clahe.py --input-folder ./images --output-folder ./enhanced
    python clahe.py --input-folder ./microscopy --output-folder ./enhanced --green-channel --workers 4

CHANNEL OPTIONS
  --red-channel    RFP, mCherry, tdTomato
  --green-channel  GFP, EGFP, FITC
  --blue-channel   DAPI, Hoechst, CFP
  (default: process as grayscale)

NOTE
  CLAHE is useful for visualization but not recommended in default pipelines —
  its results can vary between samples for non-biological reasons.
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input",        metavar="FILE",
                             help="Single input image file")
    input_group.add_argument("--input-folder", metavar="DIR",
                             help="Folder of images to process in batch")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output",        metavar="FILE",
                              help="Output file (single file mode)")
    output_group.add_argument("--output-folder", metavar="DIR",
                              help="Output folder (batch mode)")

    parser.add_argument("--grid",  type=int, default=8,
                        help="Grid size for CLAHE tiles (default: 8, creates 8x8 grid)")
    parser.add_argument("--clip",  type=float, default=2.0,
                        help="Clip limit for contrast limiting (default: 2.0)")

    channel_group = parser.add_mutually_exclusive_group()
    channel_group.add_argument("--red-channel",   action="store_true",
                               help="Apply CLAHE only to red channel")
    channel_group.add_argument("--green-channel", action="store_true",
                               help="Apply CLAHE only to green channel")
    channel_group.add_argument("--blue-channel",  action="store_true",
                               help="Apply CLAHE only to blue channel")

    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for batch mode (default: CPU count)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing")

    args = parser.parse_args()

    if args.grid <= 0:
        parser.error("--grid must be positive")
    if args.clip <= 0:
        parser.error("--clip must be positive")

    channel_only = None
    if args.red_channel:   channel_only = 'red'
    elif args.green_channel: channel_only = 'green'
    elif args.blue_channel:  channel_only = 'blue'

    tile_grid_size = (args.grid, args.grid)

    if args.input_folder:
        if not args.output_folder:
            parser.error("--output-folder is required with --input-folder")
        batch_process_folder(args.input_folder, args.output_folder,
                             args.clip, tile_grid_size, channel_only,
                             max_workers=args.workers,
                             parallel=not args.no_parallel)
    else:
        if not args.output:
            parser.error("--output is required with --input")
        if not Path(args.input).is_file():
            parser.error(f"Input file not found: {args.input}")

        log.info("Processing single file: %s mode=%s grid=%s clip=%.1f",
                 args.input,
                 f"{channel_only} channel" if channel_only else "grayscale",
                 tile_grid_size, args.clip)

        success = apply_clahe_single(args.input, args.output,
                                     args.clip, tile_grid_size, channel_only)
        if success:
            log.info("OK | output=%s", args.output)
        else:
            sys.exit(1)