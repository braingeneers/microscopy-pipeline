import argparse
import os
import sys
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from pipeline_log import get_logger
log = get_logger("crop")


def is_tiff_stack(image_path):
    """Check if a TIFF file is a multi-page stack"""
    try:
        with Image.open(image_path) as img:
            img.seek(1)
            return True
    except (EOFError, AttributeError):
        return False
    except Exception:
        return False

def crop_tiff_stack(input_path, output_path, left, top, right, bottom):
    """Crop all frames in a TIFF stack"""
    try:
        with Image.open(input_path) as img:
            cropped_frames = []
            frame_count = 0

            try:
                while True:
                    log.info("  Processing frame %d...", frame_count + 1)
                    current_frame = img.copy()
                    width, height = current_frame.size
                    cropped_frame = current_frame.crop((left, top, width - right, height - bottom))
                    cropped_frames.append(cropped_frame)
                    frame_count += 1
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if not cropped_frames:
                log.warning("No frames found in stack: %s", input_path)
                return False

            log.info("  Found %d frames in stack", frame_count)

            save_kwargs = {
                'save_all': True,
                'append_images': cropped_frames[1:] if len(cropped_frames) > 1 else []
            }

            try:
                if hasattr(img, 'tag'):
                    compression_tag = img.tag.get(259)
                    if compression_tag == 5:
                        save_kwargs['compression'] = 'tiff_lzw'
                    elif compression_tag == 1:
                        save_kwargs['compression'] = None
                imagej_tag = img.tag.get(50839)
                if imagej_tag:
                    imagej_info = imagej_tag.decode('utf-8', errors='ignore')
                    if 'images=' in imagej_info:
                        lines = imagej_info.split('\n')
                        updated_lines = [
                            f'images={frame_count}' if l.startswith('images=') else
                            f'slices={frame_count}' if l.startswith('slices=') else l
                            for l in lines
                        ]
                        updated_imagej_info = '\n'.join(updated_lines)
                        save_kwargs['tiffinfo'] = {
                            50839: updated_imagej_info.encode('utf-8'),
                            270: updated_imagej_info.encode('utf-8')
                        }
            except Exception as e:
                log.warning("Could not preserve all metadata: %s", e)

            cropped_frames[0].save(output_path, **save_kwargs)
            log.info("  Saved cropped stack with %d frames", frame_count)
            return True

    except Exception as e:
        log.error("Error processing TIFF stack %s: %s", input_path, e)
        return False

def crop_image(input_path, output_path, left, top, right, bottom):
    """Crop a single image or TIFF stack"""
    try:
        if input_path.lower().endswith(('.tif', '.tiff')) and is_tiff_stack(input_path):
            log.info("Detected TIFF stack: %s", input_path)
            return crop_tiff_stack(input_path, output_path, left, top, right, bottom)
        else:
            img = Image.open(input_path)
            width, height = img.size
            cropped = img.crop((left, top, width - right, height - bottom))
            cropped.save(output_path)
            log.info("Cropped %s -> %s", input_path, output_path)
            return True
    except Exception as e:
        log.error("Error processing %s: %s", input_path, e)
        return False


# Parallel folder processing (new)


def _worker(args):
    image_file, output_file, left, top, right, bottom, index, total = args
    fname = os.path.basename(image_file)
    log.info("[%d/%d] START | file=%s", index, total, fname)
    success = crop_image(str(image_file), str(output_file), left, top, right, bottom)
    if success:
        log.info("[%d/%d] OK    | file=%s", index, total, fname)
    else:
        log.error("[%d/%d] FAIL  | file=%s", index, total, fname)
    return success, fname

def crop_folder(input_folder, output_folder, left, top, right, bottom,
                max_workers=None, parallel=True):
    """Crop all images in a folder, in parallel by default"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    image_files = sorted(f for f in input_path.iterdir()
                         if f.is_file() and f.suffix.lower() in image_extensions)

    if not image_files:
        log.warning("No image files found in: %s", input_folder)
        return

    total = len(image_files)
    workers = max_workers or min(total, multiprocessing.cpu_count())

    log.info("BATCH START | files=%d input=%s output=%s left=%d top=%d right=%d bottom=%d workers=%s",
             total, input_folder, output_folder, left, top, right, bottom,
             workers if parallel else "sequential")

    work_items = [
        (str(f), str(output_path / f.name), left, top, right, bottom, i, total)
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
             status, successful, failed, output_folder)
    if failed > 0:
        log.warning("STATUS: %s | %d file(s) failed", status, failed)

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop images by specifying pixels to remove from each edge. "
                    "Supports single images, folders, and TIFF stacks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Single file:
    python crop.py --input image.jpg --output cropped.jpg --left 50 --top 30 --right 50 --bottom 30
    python crop.py --input stack.tiff --output cropped.tiff --left 100 --right 100

  Folder (parallel by default):
    python crop.py --input-folder ./images --output-folder ./cropped --left 100 --right 100
    python crop.py --input-folder ./raw --output-folder ./processed --left 50 --top 25 --right 75 --bottom 25 --workers 4
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", metavar="FILE",
                             help="Input image or TIFF stack (single file mode)")
    input_group.add_argument("--input-folder", metavar="DIR",
                             help="Folder of images to process (batch mode)")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", metavar="FILE",
                              help="Output file path (single file mode)")
    output_group.add_argument("--output-folder", metavar="DIR",
                              help="Output folder (batch mode)")

    parser.add_argument("--left",   type=int, default=0, help="Pixels to crop from left edge (default: 0)")
    parser.add_argument("--top",    type=int, default=0, help="Pixels to crop from top edge (default: 0)")
    parser.add_argument("--right",  type=int, default=0, help="Pixels to crop from right edge (default: 0)")
    parser.add_argument("--bottom", type=int, default=0, help="Pixels to crop from bottom edge (default: 0)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for batch mode (default: CPU count)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing")

    args = parser.parse_args()

    if any(val < 0 for val in [args.left, args.top, args.right, args.bottom]):
        parser.error("Crop values must be non-negative")

    if args.input_folder:
        if not args.output_folder:
            parser.error("--output-folder is required with --input-folder")
        if not os.path.isdir(args.input_folder):
            parser.error(f"Input folder not found: {args.input_folder}")
        crop_folder(args.input_folder, args.output_folder,
                    args.left, args.top, args.right, args.bottom,
                    max_workers=args.workers, parallel=not args.no_parallel)
    else:
        if not args.output:
            parser.error("--output is required with --input")
        if not os.path.isfile(args.input):
            parser.error(f"Input file not found: {args.input}")
        success = crop_image(args.input, args.output,
                             args.left, args.top, args.right, args.bottom)
        if not success:
            sys.exit(1)