import argparse
import numpy as np
import os
import sys
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pipeline_log import get_logger

log = get_logger("scalecolors")


# ---------------------------------------------------------------------------
# Core image processing
# ---------------------------------------------------------------------------

def scale_brightness(img_array, top, bottom):
    """
    Linearly remap pixel values from [bottom, top] to [0, 65535].
    Input may be uint8 (0-255) or uint16 (0-65535).
    Output is always uint16.
    Values outside [bottom, top] are clipped.
    """
    img_float = img_array.astype(np.float32)

    # Normalize uint16 input to 0-255 range so that top/bottom
    # can always be specified in 8-bit terms by the caller.
    if img_array.dtype == np.uint16:
        img_float *= 255.0 / 65535.0

    scaled = np.clip(
        (img_float - bottom) * (65535.0 / (top - bottom)),
        0, 65535
    )
    return scaled.astype(np.uint16)


def load_grayscale(path):
    """Open an image and return it as a grayscale NumPy array."""
    with Image.open(path) as img:
        if img.mode not in ("L", "I;16"):
            img = img.convert("L")
        return np.asarray(img)


def save_tiff(array, path):
    """Save a uint16 array as a LZW-compressed TIFF."""
    out_img = Image.fromarray(array, mode="I;16")
    out_img.save(path, compression="tiff_lzw")


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def process_single(input_path, output_path, top, bottom):
    """
    Scale a single image and write it to output_path.
    Returns True on success, False on failure.
    """
    log.info("Processing single file: %s", input_path)
    try:
        img_array = load_grayscale(input_path)
        log.info("  Input  | shape=%s dtype=%s range=%d-%d",
                 img_array.shape, img_array.dtype,
                 img_array.min(), img_array.max())

        scaled = scale_brightness(img_array, top, bottom)
        log.info("  Output | range=%d-%d", scaled.min(), scaled.max())

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        save_tiff(scaled, output_path)
        log.info("  Saved  | path=%s", output_path)
        return True

    except Exception as e:
        log.error("  FAILED | path=%s error=%s", input_path, e)
        return False


# ---------------------------------------------------------------------------
# Batch worker (called by thread pool)
# ---------------------------------------------------------------------------

def _worker(args):
    """Worker function unpacked by ThreadPoolExecutor.map()."""
    input_path, output_path, top, bottom, index, total = args
    filename = os.path.basename(input_path)
    log.info("[%d/%d] START | file=%s", index, total, filename)

    try:
        img_array = load_grayscale(input_path)
        scaled = scale_brightness(img_array, top, bottom)
        save_tiff(scaled, output_path)
        log.info("[%d/%d] OK    | file=%s range=%d-%d",
                 index, total, filename, scaled.min(), scaled.max())
        return True, filename

    except Exception as e:
        log.error("[%d/%d] FAIL  | file=%s error=%s", index, total, filename, e)
        return False, filename


# ---------------------------------------------------------------------------
# Batch folder processing
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def process_folder(input_folder, output_folder, top, bottom,
                   max_workers=None, parallel=True):
    """
    Scale all images in input_folder and write results to output_folder.
    Parallel by default; pass parallel=False for sequential execution.
    """
    if not os.path.isdir(input_folder):
        log.error("Input folder not found: %s", input_folder)
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
        and os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        log.warning("No supported image files found in: %s", input_folder)
        return

    total = len(image_files)
    workers = max_workers or min(total, multiprocessing.cpu_count())

    log.info("BATCH START | files=%d input=%s output=%s top=%d bottom=%d workers=%s",
             total,
             os.path.abspath(input_folder),
             os.path.abspath(output_folder),
             top, bottom,
             workers if parallel else "sequential")

    # Build argument list for workers
    work_items = []
    for i, fname in enumerate(image_files, 1):
        base = os.path.splitext(fname)[0]
        work_items.append((
            os.path.join(input_folder, fname),
            os.path.join(output_folder, f"{base}.tif"),
            top, bottom, i, total
        ))

    successful = 0
    failed = 0

    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for success, _ in executor.map(_worker, work_items):
                if success:
                    successful += 1
                else:
                    failed += 1
    else:
        for item in work_items:
            success, _ = _worker(item)
            if success:
                successful += 1
            else:
                failed += 1

    # Machine-parseable terminal status line
    status = "complete" if failed == 0 else "complete_with_errors"
    log.info("BATCH END | status=%s success=%d failed=%d output=%s",
             status, successful, failed, os.path.abspath(output_folder))

    if failed > 0:
        log.warning("STATUS: %s | %d file(s) failed — check errors above", status, failed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scale grayscale image brightness to 16-bit TIFF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Single file:
    python scalecolors.py --input image.png --output scaled.tif --top 200 --bottom 50

  Batch folder (parallel, default):
    python scalecolors.py --input-folder ./raw --output-folder ./scaled --top 200 --bottom 50

  Batch folder (sequential):
    python scalecolors.py --input-folder ./raw --output-folder ./scaled --top 200 --bottom 50 --no-parallel

  Limit parallel workers:
    python scalecolors.py --input-folder ./raw --output-folder ./scaled --top 200 --bottom 50 --workers 4

SCALING BEHAVIOUR
  Pixels <= bottom  →  0
  Pixels >= top     →  65535
  Pixels in between →  linearly interpolated
  Formula: output = ((input - bottom) / (top - bottom)) * 65535
        """
    )

    # Input — mutually exclusive: single file vs folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", metavar="FILE",
                             help="Single input image file")
    input_group.add_argument("--input-folder", metavar="DIR",
                             help="Folder of images to process in batch")

    # Output — mutually exclusive: single file vs folder
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", metavar="FILE",
                              help="Output file path (single-file mode)")
    output_group.add_argument("--output-folder", metavar="DIR",
                              help="Output folder (batch mode)")

    # Brightness thresholds
    parser.add_argument("--top", type=int, required=True,
                        help="Upper brightness threshold (8-bit 0-255); pixels at or above become white")
    parser.add_argument("--bottom", type=int, required=True,
                        help="Lower brightness threshold (8-bit 0-255); pixels at or below become black")

    # Parallelism
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: number of CPU cores)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing and run sequentially")

    args = parser.parse_args()

    # Validate thresholds
    if not (0 <= args.bottom <= 255):
        parser.error("--bottom must be between 0 and 255")
    if not (0 <= args.top <= 255):
        parser.error("--top must be between 0 and 255")
    if args.bottom >= args.top:
        parser.error("--bottom must be less than --top")

    # Dispatch
    if args.input_folder:
        if not args.output_folder:
            parser.error("--output-folder is required with --input-folder")
        process_folder(
            args.input_folder, args.output_folder,
            args.top, args.bottom,
            max_workers=args.workers,
            parallel=not args.no_parallel
        )

    else:
        if not args.output:
            parser.error("--output is required with --input")
        success = process_single(args.input, args.output, args.top, args.bottom)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()