import sys
import argparse
import re
import glob
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from pipeline_log import get_logger
log = get_logger("fusetiffs")

def parse_stack_filename(filename):
    """Parse filename in format stack_X.tiff and return X as integer."""
    match = re.match(r'stack_(\d+)\.tiff?$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None

def get_stack_files_sorted(folder_path):
    """Get stack_X.tiff files from folder, sorted by X value."""
    tiff_files = []
    for ext in ['*.tiff', '*.tif']:
        tiff_files.extend(glob.glob(str(Path(folder_path) / ext)))

    stack_files = []
    for tiff_file in tiff_files:
        filename = Path(tiff_file).name
        x_value = parse_stack_filename(filename)
        if x_value is not None:
            stack_files.append((x_value, tiff_file))
        else:
            log.warning("Skipping file (name does not match stack_X.tiff): %s", filename)

    stack_files.sort(key=lambda item: item[0])
    return stack_files

def process_single_file(input_path, output_path=None, brightfield_mode=False):
    """Process a single TIFF stack file, supporting both 8-bit and 16-bit images."""
    if not Path(input_path).exists():
        log.error("Input file not found: %s", input_path)
        return False

    if output_path is None:
        input_file = Path(input_path)
        suffix = "_brightfield_superimposed" if brightfield_mode else "_superimposed"
        output_path = input_file.parent / f"{input_file.stem}{suffix}{input_file.suffix}"

    log.info("Processing: %s", Path(input_path).name)
    if brightfield_mode:
        log.info("Brightfield mode: scaling to maximum dynamic range")

    stack = []
    is_16bit = False
    original_mode = None

    with Image.open(input_path) as img:
        log.info("Input image mode: %s", img.mode)
        original_mode = img.mode
        is_16bit = img.mode in ['I;16', 'I;16B', 'I;16L', 'I']

        if is_16bit:
            log.info("Detected 16-bit input image")
            max_value = 65535
            dtype_processing = np.uint64
            dtype_output = np.uint16
        else:
            log.info("Detected 8-bit input image")
            max_value = 255
            dtype_processing = np.uint32
            dtype_output = np.uint8

        try:
            while True:
                if img.mode in ['I;16', 'I;16B', 'I;16L']:
                    frame = np.array(img, dtype=np.uint16)
                    is_16bit = True
                    max_value = 65535
                    dtype_processing = np.uint64
                    dtype_output = np.uint16
                elif img.mode == 'I':
                    frame_array = np.array(img)
                    if frame_array.max() <= 65535:
                        frame = frame_array.astype(np.uint16)
                        is_16bit = True
                        max_value = 65535
                        dtype_processing = np.uint64
                        dtype_output = np.uint16
                    else:
                        frame = (frame_array / (frame_array.max() / 65535)).astype(np.uint16)
                        is_16bit = True
                        max_value = 65535
                        dtype_processing = np.uint64
                        dtype_output = np.uint16
                elif img.mode == 'L':
                    frame = np.array(img, dtype=np.uint8)
                else:
                    if is_16bit:
                        rgb_img = img.convert('RGB')
                        rgb_array = np.array(rgb_img, dtype=np.float32)
                        gray_array = (0.299 * rgb_array[:,:,0] +
                                      0.587 * rgb_array[:,:,1] +
                                      0.114 * rgb_array[:,:,2])
                        frame = (gray_array * (65535 / 255)).astype(np.uint16)
                        max_value = 65535
                        dtype_processing = np.uint64
                        dtype_output = np.uint16
                    else:
                        frame = np.array(img.convert('L'), dtype=np.uint8)

                stack.append(frame)
                img.seek(img.tell() + 1)
        except EOFError:
            pass

    if len(stack) <= 1:
        log.warning("Input is a single image, nothing to stack: %s", input_path)
        return False

    log.info("Loaded %d images from stack, processing in %s mode, max_value=%d",
             len(stack), "16-bit" if is_16bit else "8-bit", max_value)

    if is_16bit:
        stack_processed = [f.astype(np.uint16) for f in stack]
    else:
        stack_processed = [f.astype(np.uint8) for f in stack]

    summed = np.zeros_like(stack_processed[0], dtype=dtype_processing)
    for frame in stack_processed:
        summed += frame.astype(dtype_processing)
    averaged = summed / len(stack)

    if brightfield_mode and is_16bit:
        current_min, current_max = np.min(averaged), np.max(averaged)
        log.info("Original value range: %.1f - %.1f", current_min, current_max)
        if current_max > current_min:
            scaled = ((averaged - current_min) / (current_max - current_min)) * 65535
            averaged_final = np.clip(scaled, 0, 65535).astype(dtype_output)
            log.info("Scaled to full 16-bit range: 0 - 65535")
        else:
            averaged_final = np.clip(averaged, 0, 65535).astype(dtype_output)
            log.info("No scaling applied (uniform image)")
    elif brightfield_mode and not is_16bit:
        current_min, current_max = np.min(averaged), np.max(averaged)
        log.info("Original value range: %.1f - %.1f", current_min, current_max)
        if current_max > current_min:
            scaled = ((averaged - current_min) / (current_max - current_min)) * 255
            averaged_final = np.clip(scaled, 0, 255).astype(dtype_output)
            log.info("Scaled to full 8-bit range: 0 - 255")
        else:
            averaged_final = np.clip(averaged, 0, 255).astype(dtype_output)
            log.info("No scaling applied (uniform image)")
    else:
        if is_16bit:
            averaged_final = np.clip(averaged, 0, 65535).astype(dtype_output)
        else:
            averaged_final = np.clip(averaged, 0, 255).astype(dtype_output)

    if is_16bit:
        result_img = Image.fromarray(averaged_final, mode='I;16')
        log.info("Saving as I;16 mode")
    else:
        result_img = Image.fromarray(averaged_final, mode='L')
        log.info("Saving as L mode")

    try:
        with Image.open(input_path) as original_img:
            if hasattr(original_img, 'tag'):
                result_img.save(output_path, tiffinfo=original_img.tag)
            else:
                result_img.save(output_path)
    except Exception as e:
        log.warning("Could not preserve metadata: %s", e)
        result_img.save(output_path)

    log.info("Saved %s %s image -> %s",
             "brightfield" if brightfield_mode else "averaged",
             "16-bit" if is_16bit else "8-bit",
             output_path)
    return True

# Parallel folder processing

def _worker(args):
    x_value, filepath, output_path, brightfield_mode, index, total = args
    log.info("[%d/%d] START | x=%d file=%s", index, total, x_value, Path(filepath).name)
    success = process_single_file(filepath, output_path, brightfield_mode)
    if success:
        log.info("[%d/%d] OK    | x=%d", index, total, x_value)
    else:
        log.error("[%d/%d] FAIL  | x=%d file=%s", index, total, x_value, Path(filepath).name)
    return success, x_value

def process_folder(folder_path, output_dir=None, brightfield_mode=False,
                   max_workers=None, parallel=True):
    """Process all stack_X.tiff files in a folder."""
    folder_path = Path(folder_path)

    if not folder_path.exists():
        log.error("Folder not found: %s", folder_path)
        return False

    if output_dir is None:
        suffix = "brightfield_superimposed" if brightfield_mode else "superimposed"
        output_dir = folder_path / suffix
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    stack_files = get_stack_files_sorted(folder_path)
    if not stack_files:
        log.warning("No stack_X.tiff files found in: %s", folder_path)
        return False

    total = len(stack_files)
    workers = max_workers or min(total, multiprocessing.cpu_count())

    log.info("BATCH START | stacks=%d input=%s output=%s workers=%s",
             total, folder_path, output_dir,
             workers if parallel else "sequential")

    work_items = []
    for i, (x_value, filepath) in enumerate(stack_files, 1):
        input_file = Path(filepath)
        suffix = "_brightfield_superimposed" if brightfield_mode else "_superimposed"
        output_path = output_dir / f"{input_file.stem}{suffix}{input_file.suffix}"
        work_items.append((x_value, filepath, str(output_path), brightfield_mode, i, total))

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
        log.warning("STATUS: %s | %d stack(s) failed", status, failed)

    return successful > 0


def main():
    parser = argparse.ArgumentParser(
        description="Fuse TIFF stacks by averaging frames. Supports 8-bit and 16-bit images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Single file:
    python fusetiffs.py --input stack_1.tiff
    python fusetiffs.py --input stack_1.tiff --output averaged.tiff --brightfield

  Folder (parallel by default):
    python fusetiffs.py --input-folder ./stacks
    python fusetiffs.py --input-folder ./stacks --output ./results --brightfield --workers 4

OUTPUT NAMING (when --output not specified)
  Normal mode:     filename_superimposed.tiff
  Brightfield:     filename_brightfield_superimposed.tiff
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input',        metavar='FILE',
                             help='Single TIFF stack file to process')
    input_group.add_argument('--input-folder', metavar='DIR',
                             help='Folder containing stack_X.tiff files to batch process')

    parser.add_argument('--output', metavar='PATH',
                        help='Output file (single mode) or output directory (folder mode)')
    parser.add_argument('--brightfield', action='store_true',
                        help='Scale averaged image to full dynamic range')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers for folder mode (default: CPU count)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')

    args = parser.parse_args()

    if args.input_folder:
        if not Path(args.input_folder).is_dir():
            parser.error(f"Input folder not found: {args.input_folder}")
        success = process_folder(args.input_folder, args.output, args.brightfield,
                                 max_workers=args.workers,
                                 parallel=not args.no_parallel)
    else:
        if not Path(args.input).exists():
            parser.error(f"Input file not found: {args.input}")
        success = process_single_file(args.input, args.output, args.brightfield)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()