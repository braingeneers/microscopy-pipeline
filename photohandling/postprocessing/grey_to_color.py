import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import numpy as np
from PIL import Image

from pipeline_log import get_logger
log = get_logger("grey_to_color")


def apply_gamma_correction(rgb_arr, gamma, max_val):
    """Apply gamma correction to RGB array"""
    if gamma == 1.0:
        return rgb_arr
    normalized = rgb_arr.astype(np.float64) / max_val
    gamma_corrected = np.power(normalized, gamma)
    return (gamma_corrected * max_val).astype(rgb_arr.dtype)

def grey_to_color(input_path, output_path, color_channel, gamma=1.0):
    """Convert grayscale image to colored version by placing values into one channel"""
    with Image.open(input_path) as img:
        log.info("Processing %s: mode=%s size=%s", input_path.name, img.mode, img.size)

        if img.mode == 'L':
            arr = np.array(img, dtype=np.uint8)
            max_val = 255
            output_dtype = np.uint8
            bit_depth = "8-bit"
        elif img.mode in ['I;16', 'I;16B', 'I;16L']:
            arr = np.array(img, dtype=np.uint16)
            max_val = 65535
            output_dtype = np.uint16
            bit_depth = "16-bit"
        elif img.mode == 'I':
            arr_temp = np.array(img)
            if arr_temp.max() <= 65535:
                arr = arr_temp.astype(np.uint16)
                max_val = 65535
                output_dtype = np.uint16
                bit_depth = "16-bit (from I mode)"
            else:
                arr = (arr_temp * (65535 / arr_temp.max())).astype(np.uint16)
                max_val = 65535
                output_dtype = np.uint16
                bit_depth = "16-bit (scaled from 32-bit)"
        else:
            img_gray = img.convert("L")
            arr = np.array(img_gray, dtype=np.uint8)
            max_val = 255
            output_dtype = np.uint8
            bit_depth = "8-bit (converted from color)"

        log.info("  Detected: %s max_val=%d", bit_depth, max_val)

        if output_dtype == np.uint16:
            rgb_arr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint16)
        else:
            rgb_arr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

        channels = {'red': 0, 'green': 1, 'blue': 2}
        if color_channel not in channels:
            raise ValueError(f"Invalid color '{color_channel}'. Choose from red, green, blue.")

        for ch_name, ch_idx in channels.items():
            rgb_arr[..., ch_idx] = arr if ch_name == color_channel else 0

        if gamma != 1.0:
            rgb_arr = apply_gamma_correction(rgb_arr, gamma, max_val)
            log.info("  Applied gamma correction: γ=%.3f", gamma)

        if output_dtype == np.uint16:
            rgb_arr_8bit = (rgb_arr / (max_val / 255)).astype(np.uint8)
            color_img = Image.fromarray(rgb_arr_8bit, mode="RGB")
            log.info("  Note: 16-bit input converted to 8-bit RGB for compatibility")
        else:
            color_img = Image.fromarray(rgb_arr, mode="RGB")

        color_img.save(output_path)

        gamma_suffix = f", γ={gamma}" if gamma != 1.0 else ""
        log.info("  Saved -> %s (%s%s)", output_path.name, color_channel, gamma_suffix)


# Parallel batch processing 

def _worker(args):
    img_path, output_path, color_channel, gamma, index, total = args
    log.info("[%d/%d] START | file=%s", index, total, img_path.name)
    try:
        grey_to_color(img_path, output_path, color_channel, gamma)
        log.info("[%d/%d] OK    | file=%s", index, total, img_path.name)
        return True, img_path.name
    except Exception as e:
        log.error("[%d/%d] FAIL  | file=%s error=%s", index, total, img_path.name, e)
        return False, img_path.name

def batch_process(input_dir, output_dir, color_channel, gamma=1.0,
                  max_workers=None, parallel=True):
    """Process all PNG and TIFF files in input directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ["*.png", "*.PNG", "*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(image_files)

    if not image_files:
        log.warning("No PNG or TIFF files found in: %s", input_dir)
        return

    total = len(image_files)
    workers = max_workers or min(total, multiprocessing.cpu_count())

    log.info("BATCH START | files=%d input=%s output=%s color=%s gamma=%.3f workers=%s",
             total, input_dir, output_dir, color_channel, gamma,
             workers if parallel else "sequential")

    work_items = [
        (f, output_dir / f.name, color_channel, gamma, i, total)
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
        description="Pseudocolorize grayscale fluorescence images by placing pixel values "
                    "into a single RGB channel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  python grey_to_color.py --input ./grayscale --output ./red_output --color red
  python grey_to_color.py --input ./images --output ./green_output --color green --gamma 0.7
  python grey_to_color.py --input ./16bit --output ./blue_output --color blue --gamma 2.2 --workers 4

GAMMA
  < 1.0  brightens dark areas (e.g. 0.5)
  = 1.0  no correction (default)
  > 1.0  darkens bright areas (e.g. 2.0)

BIT DEPTH
  8-bit grayscale  → 8-bit RGB
  16-bit grayscale → 8-bit RGB (scaled, due to PIL RGB mode limitation)
        """
    )

    parser.add_argument("--input",  metavar="DIR", required=True,
                        help="Directory containing grayscale PNG/TIFF images")
    parser.add_argument("--output", metavar="DIR", required=True,
                        help="Directory to save colored images")
    parser.add_argument("--color",  required=True, choices=["red", "green", "blue"],
                        help="Target color channel")
    parser.add_argument("--gamma",  type=float, default=1.0,
                        help="Gamma correction value (default: 1.0)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing")

    args = parser.parse_args()

    if args.gamma <= 0:
        parser.error("--gamma must be positive")

    if not Path(args.input).is_dir():
        parser.error(f"Input directory not found: {args.input}")

    batch_process(args.input, args.output, args.color, args.gamma,
                  max_workers=args.workers, parallel=not args.no_parallel)