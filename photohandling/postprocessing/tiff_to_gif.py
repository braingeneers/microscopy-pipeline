import os
import re
import sys
import argparse
from PIL import Image
import numpy as np

from pipeline_log import get_logger
log = get_logger("tiff_to_gif")


def normalize_16bit_to_8bit(image):
    """Convert 16-bit image to 8-bit by normalizing the intensity range"""
    if image.mode in ['I;16', 'I;16B', 'I;16L', 'I']:
        arr = np.array(image)
        if arr.max() > 255:
            log.info("    16-bit image detected (max value: %d)", arr.max())
            if arr.max() > arr.min():
                arr_normalized = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
            else:
                arr_normalized = np.zeros_like(arr, dtype=np.uint8)
            return Image.fromarray(arr_normalized, mode='L')
        else:
            return image.convert('L')
    else:
        return image

def get_tiff_files(folder):
    pattern = re.compile(r'stack_(\d+)_edf\.tif$', re.IGNORECASE)
    files = []
    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            n = int(match.group(1))
            files.append((n, fname))
    files.sort(key=lambda x: x[0])
    return [os.path.join(folder, fname) for _, fname in files]

def tiff_to_gif(folder, output_gif):
    tiff_files = get_tiff_files(folder)
    if not tiff_files:
        log.warning("No matching TIFF files found in: %s", folder)
        return

    log.info("CONVERT START | files=%d input=%s output=%s",
             len(tiff_files), folder, output_gif)
    images = []

    for i, tiff_file in enumerate(tiff_files, 1):
        log.info("[%d/%d] Processing: %s", i, len(tiff_files), os.path.basename(tiff_file))
        try:
            img = Image.open(tiff_file)
            log.info("    mode=%s size=%s", img.mode, img.size)

            if img.mode in ['I;16', 'I;16B', 'I;16L', 'I']:
                img_rgb = normalize_16bit_to_8bit(img).convert('RGB')
                log.info("    Normalized 16-bit to 8-bit RGB")
            else:
                img_rgb = img.convert('RGB')
                log.info("    Converted to RGB")

            images.append(img_rgb)

        except Exception as e:
            log.error("    Error processing %s: %s", tiff_file, e)
            continue

    if not images:
        log.error("No images could be processed successfully")
        sys.exit(1)

    log.info("Creating animated GIF | frames=%d", len(images))

    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=10,
        loop=0,
        optimize=True
    )

    log.info("CONVERT END | status=complete frames=%d duration_per_frame=10ms output=%s",
             len(images), output_gif)

# CLI
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a sequence of TIFF files into an animated GIF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  python tiff_to_gif.py --input ./tiff_stack --output animation.gif
  python tiff_to_gif.py --input ./microscopy_images --output timelapse.gif

FILE PATTERN
  Looks for files matching: stack_N_edf.tif
  Files are sorted by N value and assembled in order.

BIT DEPTH
  8-bit TIFFs are used as-is.
  16-bit TIFFs are normalized to 0-255 before inclusion.
        """
    )

    parser.add_argument("--input",  metavar="DIR", required=True,
                        help="Folder containing TIFF files")
    parser.add_argument("--output", metavar="FILE", required=True,
                        help="Output GIF file path")

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        parser.error(f"Input folder not found: {args.input}")

    tiff_to_gif(args.input, args.output)