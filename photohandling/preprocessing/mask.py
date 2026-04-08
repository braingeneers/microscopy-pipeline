import argparse
import os
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sys

from pipeline_log import get_logger
log = get_logger("mask")



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

def feather_mask(mask, feather_radius):
    """Apply feathering (Gaussian blur) to mask edges for smooth transitions"""
    if feather_radius <= 0:
        return mask
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
    blurred_mask = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return blurred_mask

def apply_mask_to_frame(image, mask, replace_color=(255, 0, 0), feather_radius=0):
    """Apply a mask to a single image frame using vectorized operations with optional feathering"""
    if feather_radius > 0:
        mask = feather_mask(mask, feather_radius)
        log.info("    Applied feathering with radius %s", feather_radius)

    is_grayscale = image.mode in ['L', 'LA', 'I;16', 'I;16B', 'I;16L']
    is_16bit = image.mode in ['I;16', 'I;16B', 'I;16L']

    if is_grayscale:
        if is_16bit:
            if image.mode != 'I;16':
                image = image.convert('I;16')
            image_array = np.asarray(image, dtype=np.float32)
            r, g, b = replace_color
            if r == g == b:
                replace_value = r * 257 if 0 <= r <= 255 else min(max(r, 0), 65535)
            else:
                gray_8bit = int(0.299 * r + 0.587 * g + 0.114 * b)
                replace_value = gray_8bit * 257
            replace_value = np.float32(replace_value)
        else:
            image = image.convert('L')
            image_array = np.asarray(image, dtype=np.float32)
            r, g, b = replace_color
            if r == g == b:
                replace_value = r if 0 <= r <= 255 else (min(r // 257, 255) if r > 255 else 0)
            else:
                replace_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            replace_value = np.float32(replace_value)
    else:
        image = image.convert('RGB')
        image_array = np.asarray(image, dtype=np.float32)
        replace_value = np.array(replace_color, dtype=np.float32)

    if image.size != mask.size:
        raise ValueError(f"Image and mask must be the same size. "
                         f"Image: {image.size}, Mask: {mask.size}")

    mask_array = np.asarray(mask, dtype=np.float32) / 255.0

    if feather_radius > 0:
        alpha = 1.0 - mask_array
        if is_grayscale:
            blended_array = image_array * (1.0 - alpha) + replace_value * alpha
        else:
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            blended_array = image_array * (1.0 - alpha_3d) + replace_value * alpha_3d
        pixels_replaced = np.count_nonzero(alpha > 0.1)
        if is_16bit:
            final_array = np.clip(blended_array, 0, 65535).astype(np.uint16)
            masked_image = Image.fromarray(final_array, mode='I;16')
        elif is_grayscale:
            final_array = np.clip(blended_array, 0, 255).astype(np.uint8)
            masked_image = Image.fromarray(final_array, mode='L')
        else:
            final_array = np.clip(blended_array, 0, 255).astype(np.uint8)
            masked_image = Image.fromarray(final_array, mode='RGB')
    else:
        mask_bool = mask_array < 0.5
        pixels_replaced = np.count_nonzero(mask_bool)
        image_array[mask_bool] = replace_value
        if is_16bit:
            masked_image = Image.fromarray(image_array.astype(np.uint16), mode='I;16')
        elif is_grayscale:
            masked_image = Image.fromarray(image_array.astype(np.uint8), mode='L')
        else:
            masked_image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')

    return masked_image, pixels_replaced

def apply_mask_to_stack(image_path, mask_path, output_path, replace_color=(255, 0, 0), feather_radius=0):
    """Apply a mask to all frames in a TIFF stack"""
    try:
        mask = Image.open(mask_path).convert('L')
        if feather_radius > 0:
            log.info("  Applying feathering to mask with radius %s...", feather_radius)
            mask = feather_mask(mask, feather_radius)

        with Image.open(image_path) as img:
            masked_frames = []
            frame_count = 0
            total_pixels_replaced = 0
            is_grayscale = None
            is_16bit = None

            try:
                while True:
                    log.info("  Processing frame %d...", frame_count + 1)
                    current_frame = img.copy()

                    if is_grayscale is None:
                        is_grayscale = current_frame.mode in ['L', 'LA', 'I;16', 'I;16B', 'I;16L']
                        is_16bit = current_frame.mode in ['I;16', 'I;16B', 'I;16L']
                        bit_depth = "16-bit" if is_16bit else "8-bit"
                        log.info("    Stack contains %s %s images",
                                 bit_depth, "grayscale" if is_grayscale else "color")

                    masked_frame, pixels_replaced = apply_mask_to_frame(
                        current_frame, mask, replace_color, feather_radius=0)
                    masked_frames.append(masked_frame)
                    total_pixels_replaced += pixels_replaced
                    frame_count += 1
                    img.seek(img.tell() + 1)

            except EOFError:
                pass

            if not masked_frames:
                log.warning("No frames found in stack: %s", image_path)
                return 0, False

            log.info("  Found %d frames in stack", frame_count)

            save_kwargs = {
                'save_all': True,
                'append_images': masked_frames[1:] if len(masked_frames) > 1 else []
            }

            try:
                if hasattr(img, 'tag'):
                    compression_tag = img.tag.get(259)
                    if compression_tag == 5:
                        save_kwargs['compression'] = 'tiff_lzw'
                    elif compression_tag == 1:
                        save_kwargs['compression'] = None
                if is_16bit:
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

            masked_frames[0].save(output_path, **save_kwargs)
            log.info("  Saved masked stack with %d frames, %d pixels replaced",
                     frame_count, total_pixels_replaced)
            return total_pixels_replaced, is_grayscale

    except Exception as e:
        log.error("Error processing TIFF stack %s: %s", image_path, e)
        return 0, False

def apply_mask(image_path, mask_path, output_path, replace_color=(255, 0, 0), feather_radius=0):
    """Apply a mask to an image or TIFF stack"""
    if image_path.lower().endswith(('.tif', '.tiff')) and is_tiff_stack(image_path):
        log.info("Detected TIFF stack: %s", image_path)
        return apply_mask_to_stack(image_path, mask_path, output_path, replace_color, feather_radius)
    else:
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')
        is_grayscale = image.mode in ['L', 'LA', 'I;16', 'I;16B', 'I;16L']
        is_16bit = image.mode in ['I;16', 'I;16B', 'I;16L']
        log.info("  Image mode: %s (%s)", image.mode,
                 ("16-bit grayscale" if is_16bit else "8-bit grayscale") if is_grayscale else "color")
        masked_image, pixels_replaced = apply_mask_to_frame(image, mask, replace_color, feather_radius)
        masked_image.save(output_path)
        return pixels_replaced, is_grayscale

def parse_color(color_str):
    """Parse color string in format 'R,G,B', hex codes, grayscale values, or common color names"""
    color_names = {
        'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
        'black': (0, 0, 0), 'white': (255, 255, 255), 'yellow': (255, 255, 0),
        'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'orange': (255, 165, 0),
        'purple': (128, 0, 128), 'pink': (255, 192, 203), 'gray': (128, 128, 128),
        'grey': (128, 128, 128), 'lightgray': (211, 211, 211), 'lightgrey': (211, 211, 211),
        'darkgray': (64, 64, 64), 'darkgrey': (64, 64, 64),
        'white16': (65535, 65535, 65535), 'lightgray16': (49151, 49151, 49151),
        'gray16': (32767, 32767, 32767), 'darkgray16': (16383, 16383, 16383),
        'black16': (0, 0, 0)
    }
    if color_str.lower() in color_names:
        return color_names[color_str.lower()]
    if color_str.startswith('#'):
        try:
            hex_color = color_str[1:]
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            elif len(hex_color) != 6:
                raise ValueError("Invalid hex length")
            return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
        except ValueError:
            raise ValueError(f"Invalid hex color format: '{color_str}'.")
    if ',' not in color_str:
        try:
            gray_value = int(color_str)
            if not 0 <= gray_value <= 65535:
                raise ValueError("Grayscale value must be between 0 and 65535")
            return (gray_value, gray_value, gray_value)
        except ValueError:
            pass
    try:
        r, g, b = map(int, color_str.split(','))
        if not all(0 <= val <= 65535 for val in [r, g, b]):
            raise ValueError("RGB values must be between 0 and 65535")
        return (r, g, b)
    except ValueError:
        raise ValueError(f"Invalid color format: '{color_str}'.")


# Parallel folder processing (new)


def _worker(args):
    image_file, mask_path, output_file, replace_color, feather_radius, index, total = args
    fname = os.path.basename(image_file)
    log.info("[%d/%d] START | file=%s", index, total, fname)
    try:
        pixels_replaced, was_grayscale = apply_mask(
            str(image_file), mask_path, str(output_file), replace_color, feather_radius)
        log.info("[%d/%d] OK    | file=%s pixels_replaced=%d", index, total, fname, pixels_replaced)
        return True, fname
    except Exception as e:
        log.error("[%d/%d] FAIL  | file=%s error=%s", index, total, fname, e)
        return False, fname

def apply_mask_folder(input_folder, mask_path, output_folder, replace_color=(255, 0, 0),
                      feather_radius=0, max_workers=None, parallel=True):
    """Apply a mask to all images in a folder, in parallel by default"""
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

    log.info("BATCH START | files=%d input=%s output=%s workers=%s",
             total, input_folder, output_folder,
             workers if parallel else "sequential")

    work_items = [
        (str(f), mask_path, str(output_path / f.name),
         replace_color, feather_radius, i, total)
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
        description="Apply a mask to images, replacing masked areas with a specified color. "
                    "Supports single images, TIFF stacks, and batch processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Single file:
    python mask.py --input image.jpg --mask mask.png --output output.jpg
    python mask.py --input stack.tiff --mask mask.png --output masked.tiff --feather 5

  Folder (parallel by default):
    python mask.py --input-folder ./images --mask mask.png --output-folder ./masked
    python mask.py --input-folder ./images --mask mask.png --output-folder ./masked --color 128 --feather 3 --workers 4
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", metavar="FILE",
                             help="Input image or TIFF stack (single file mode)")
    input_group.add_argument("--input-folder", metavar="DIR",
                             help="Folder of images to process (batch mode)")

    parser.add_argument("--mask", required=True, metavar="FILE",
                        help="Grayscale mask image (black areas will be replaced)")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", metavar="FILE",
                              help="Output file path (single file mode)")
    output_group.add_argument("--output-folder", metavar="DIR",
                              help="Output folder (batch mode)")

    parser.add_argument("--color", default="red",
                        help="Replacement color for masked areas (default: red). "
                             "Accepts: grayscale value, R,G,B, hex (#FF0000), or color name")
    parser.add_argument("--feather", type=float, default=0,
                        help="Feathering radius in pixels for soft mask edges (default: 0)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for batch mode (default: CPU count)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing")

    args = parser.parse_args()

    if args.feather < 0:
        parser.error("--feather must be non-negative")

    try:
        replace_color = parse_color(args.color)
    except ValueError as e:
        parser.error(str(e))

    if not os.path.isfile(args.mask):
        parser.error(f"Mask file not found: {args.mask}")

    if args.input_folder:
        if not args.output_folder:
            parser.error("--output-folder is required with --input-folder")
        if not os.path.isdir(args.input_folder):
            parser.error(f"Input folder not found: {args.input_folder}")
        apply_mask_folder(args.input_folder, args.mask, args.output_folder,
                          replace_color, args.feather,
                          max_workers=args.workers,
                          parallel=not args.no_parallel)
    else:
        if not args.output:
            parser.error("--output is required with --input")
        if not os.path.isfile(args.input):
            parser.error(f"Input file not found: {args.input}")
        try:
            pixels_replaced, was_grayscale = apply_mask(
                args.input, args.mask, args.output, replace_color, args.feather)
            log.info("OK | pixels_replaced=%d type=%s output=%s",
                     pixels_replaced, "grayscale" if was_grayscale else "color", args.output)
        except Exception as e:
            log.error("FAILED | %s", e)
            sys.exit(1)