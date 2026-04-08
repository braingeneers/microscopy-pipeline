import numpy as np
import pywt
from PIL import Image
import os
import sys
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from pipeline_log import get_logger
log = get_logger("complexedf")



def complex_wavelet_edf(images, wavelet='db3', levels=3, bit_depth=16, top_n=1):
    """
    Extended Depth of Field using Complex Wavelets.
    Args:
        images (list of np.ndarray): List of grayscale images (2D arrays) to fuse.
        wavelet (str): Wavelet type (default 'db3').
        levels (int): Number of decomposition levels.
        bit_depth (int): Output bit depth (8 or 16).
        top_n (int): Number of top layers to average based on wavelet metric.
    Returns:
        tuple: (fused_image, min_z_used, max_z_used)
    """
    max_val = 255 if bit_depth == 8 else 65535
    float_images = [img.astype(np.float64) for img in images]

    def complex_wavelet_decompose(img, wavelet, levels):
        coeffs = pywt.wavedec2(img, wavelet, level=levels)
        complex_coeffs = []
        for c in coeffs:
            if isinstance(c, tuple):
                cH, cV, cD = c
                complex_c = cH + 1j * cV
                complex_coeffs.append((complex_c, cD))
            else:
                complex_coeffs.append(c)
        return complex_coeffs

    def complex_wavelet_reconstruct(coeffs, wavelet):
        real_coeffs = []
        for c in coeffs:
            if isinstance(c, tuple):
                complex_c, cD = c
                real_coeffs.append((np.real(complex_c), np.imag(complex_c), cD))
            else:
                real_coeffs.append(c)
        return pywt.waverec2(real_coeffs, wavelet)

    z_indices_used = []
    decomposed = [complex_wavelet_decompose(img, wavelet, levels) for img in float_images]

    fused_coeffs = []
    for level_idx, coeffs_per_level in enumerate(zip(*decomposed)):
        if isinstance(coeffs_per_level[0], tuple):
            complex_c_list, cD_list = zip(*coeffs_per_level)
            mags = [np.abs(c) for c in complex_c_list]
            mag_stack = np.stack(mags, axis=0)
            top_n_clamped = min(top_n, len(complex_c_list))
            top_indices = np.argpartition(-mag_stack, top_n_clamped-1, axis=0)[:top_n_clamped]

            if level_idx == 0:
                z_indices_used.extend(top_indices.flatten())

            complex_c_stack = np.stack(complex_c_list, axis=0)
            cD_stack = np.stack(cD_list, axis=0)
            fused_complex_c = np.zeros_like(complex_c_list[0])
            fused_cD = np.zeros_like(cD_list[0])

            for i in range(top_n_clamped):
                idx_map = top_indices[i]
                fused_complex_c += np.choose(idx_map, complex_c_stack)
                fused_cD += np.choose(idx_map, cD_stack)

            fused_complex_c /= top_n_clamped
            fused_cD /= top_n_clamped
            fused_coeffs.append((fused_complex_c, fused_cD))
        else:
            mags = [np.abs(coeff) for coeff in coeffs_per_level]
            mag_stack = np.stack(mags, axis=0)
            coeff_stack = np.stack(coeffs_per_level, axis=0)
            top_n_clamped = min(top_n, len(coeffs_per_level))
            top_indices = np.argpartition(-mag_stack, top_n_clamped-1, axis=0)[:top_n_clamped]

            if level_idx == 0:
                z_indices_used.extend(top_indices.flatten())

            fused_approx = np.zeros_like(coeffs_per_level[0])
            for i in range(top_n_clamped):
                fused_approx += np.choose(top_indices[i], coeff_stack)
            fused_approx /= top_n_clamped
            fused_coeffs.append(fused_approx)

    fused_img = complex_wavelet_reconstruct(fused_coeffs, wavelet)

    if z_indices_used:
        min_z_used = int(np.min(z_indices_used))
        max_z_used = int(np.max(z_indices_used))
    else:
        min_z_used = 0
        max_z_used = len(images) - 1

    return np.clip(fused_img, 0, max_val), min_z_used, max_z_used


def complex_wavelet_edf_from_file(input_path, output_path, wavelet='db3', levels=3,
                                   bit_depth=16, top_n=1):
    """Load a TIFF stack, run EDF fusion, and save result."""
    images = []
    original_mode = None
    input_bit_depth = 8

    with Image.open(input_path) as img:
        original_mode = img.mode
        log.info("  Input mode: %s", original_mode)

        if img.mode == 'I;16':
            input_bit_depth = 16
        elif img.mode == 'L':
            input_bit_depth = 8

        log.info("  Input bit depth: %d  Output bit depth: %d", input_bit_depth, bit_depth)

        try:
            while True:
                if img.mode == 'I;16':
                    frame = img
                elif img.mode == 'L':
                    frame = img
                else:
                    frame = img.convert('L')

                img_array = np.array(frame)

                if input_bit_depth == 8 and bit_depth == 16:
                    img_array = img_array.astype(np.uint16) * 257
                elif input_bit_depth == 16 and bit_depth == 8:
                    img_array = (img_array.astype(np.float32) / 257).astype(np.uint8)

                images.append(img_array)
                img.seek(img.tell() + 1)
        except EOFError:
            pass

    log.info("  Loaded %d images, dtype=%s range=%d-%d",
             len(images), images[0].dtype, images[0].min(), images[0].max())

    fused_image, min_z_used, max_z_used = complex_wavelet_edf(
        images, wavelet, levels, bit_depth, top_n)

    if bit_depth == 16:
        result_img = Image.fromarray(fused_image.astype(np.uint16), mode='I;16')
    else:
        result_img = Image.fromarray(fused_image.astype(np.uint8), mode='L')

    result_img.save(output_path)
    log.info("  Fused %d-bit image saved | z_range=%d-%d (of %d)",
             bit_depth, min_z_used, max_z_used, len(images) - 1)

    return min_z_used, max_z_used


# Parallel folder processing 

def _worker(args):
    tiff_file, input_file_path, output_file_path, wavelet, levels, bit_depth, top_n, index, total = args
    log.info("[%d/%d] START | file=%s", index, total, tiff_file)

    try:
        # Count frames first
        frame_count = 0
        with Image.open(input_file_path) as img:
            try:
                while True:
                    frame_count += 1
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

        if frame_count < 2:
            log.warning("[%d/%d] SKIP  | file=%s reason=single_frame", index, total, tiff_file)
            return None, tiff_file  # None signals skip, not failure

        log.info("[%d/%d]        | file=%s frames=%d", index, total, tiff_file, frame_count)

        min_z, max_z = complex_wavelet_edf_from_file(
            input_file_path, output_file_path, wavelet, levels, bit_depth, top_n)

        log.info("[%d/%d] OK    | file=%s z_range=%d-%d", index, total, tiff_file, min_z, max_z)
        return (tiff_file, min_z, max_z, frame_count), tiff_file

    except Exception as e:
        log.error("[%d/%d] FAIL  | file=%s error=%s", index, total, tiff_file, e)
        return False, tiff_file


def process_folder(input_folder, output_folder, wavelet='db3', levels=3, bit_depth=16,
                   top_n=1, max_workers=None, parallel=True):
    """Batch process all TIFF stack files in a folder."""
    input_path = os.path.abspath(input_folder)
    output_path = os.path.abspath(output_folder)
    os.makedirs(output_path, exist_ok=True)

    tiff_extensions = {'.tif', '.tiff'}
    tiff_files = sorted(
        f for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f)) and
        os.path.splitext(f.lower())[1] in tiff_extensions
    )

    if not tiff_files:
        log.warning("No TIFF files found in: %s", input_folder)
        return

    total = len(tiff_files)
    workers = max_workers or min(total, multiprocessing.cpu_count())

    log.info("BATCH START | files=%d input=%s output=%s wavelet=%s levels=%d bit_depth=%d top_n=%d workers=%s",
             total, input_path, output_path, wavelet, levels, bit_depth, top_n,
             workers if parallel else "sequential")

    work_items = []
    for i, tiff_file in enumerate(tiff_files, 1):
        base_name = os.path.splitext(tiff_file)[0]
        output_filename = f"{base_name}_edf.tif"
        work_items.append((
            tiff_file,
            os.path.join(input_path, tiff_file),
            os.path.join(output_path, output_filename),
            wavelet, levels, bit_depth, top_n, i, total
        ))

    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_worker, work_items))
    else:
        results = [_worker(item) for item in work_items]

    # Write CSV summary and tally outcomes
    csv_path = os.path.join(output_path, "edf_z_ranges.csv")
    successful, failed = 0, 0

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'timepoint', 'min_z_used', 'max_z_used',
                         'total_frames', 'z_range_used'])
        for result, fname in results:
            if result is None:
                pass  # skipped (single frame)
            elif result is False:
                failed += 1
            else:
                tiff_file, min_z, max_z, frame_count = result
                base_name = os.path.splitext(tiff_file)[0]
                writer.writerow([tiff_file, base_name, min_z, max_z,
                                  frame_count, max_z - min_z + 1])
                successful += 1

    status = "complete" if failed == 0 else "complete_with_errors"
    log.info("BATCH END | status=%s success=%d failed=%d csv=%s output=%s",
             status, successful, failed, csv_path, output_path)
    if failed > 0:
        log.warning("STATUS: %s | %d file(s) failed", status, failed)


# CLI

def main():
    parser = argparse.ArgumentParser(
        description='Complex Wavelet Extended Depth of Field fusion for TIFF stacks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Single file:
    python complexedf.py --input stack.tif --output fused.tif
    python complexedf.py --input stack.tif --output fused.tif --wavelet db5 --levels 8 --bit-depth 8

  Folder (parallel by default):
    python complexedf.py --input-folder ./stacks --output-folder ./fused
    python complexedf.py --input-folder ./stacks --output-folder ./fused --top-n 3 --workers 4

WAVELET OPTIONS
  Common wavelets: db1, db2, db3, db4, db5, haar, bior2.2, bior4.4, coif2, coif4

OUTPUT (folder mode)
  Per-file:  {original_name}_edf.tif
  Summary:   edf_z_ranges.csv  (z-range stats for all timepoints)
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input',        metavar='FILE',
                             help='Input TIFF stack file (single file mode)')
    input_group.add_argument('--input-folder', metavar='DIR',
                             help='Folder containing TIFF stack files (batch mode)')

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--output',        metavar='FILE',
                              help='Output fused TIFF file (single file mode)')
    output_group.add_argument('--output-folder', metavar='DIR',
                              help='Output folder (batch mode)')

    parser.add_argument('-w', '--wavelet',   default='db3',
                        help='Wavelet type (default: db3)')
    parser.add_argument('-l', '--levels',    type=int, default=11,
                        help='Number of decomposition levels (default: 11)')
    parser.add_argument('-b', '--bit-depth', type=int, choices=[8, 16], default=16,
                        help='Output bit depth: 8 or 16 (default: 16)')
    parser.add_argument('-n', '--top-n',     type=int, default=1,
                        help='Number of top layers to average (default: 1)')
    parser.add_argument('--workers',         type=int, default=None,
                        help='Number of parallel workers for batch mode (default: CPU count)')
    parser.add_argument('--no-parallel',     action='store_true',
                        help='Disable parallel processing')

    args = parser.parse_args()

    if args.input_folder:
        if not args.output_folder:
            parser.error("--output-folder is required with --input-folder")
        if not os.path.isdir(args.input_folder):
            parser.error(f"Input folder not found: {args.input_folder}")
        process_folder(args.input_folder, args.output_folder,
                       args.wavelet, args.levels, args.bit_depth, args.top_n,
                       max_workers=args.workers, parallel=not args.no_parallel)
    else:
        if not args.output:
            parser.error("--output is required with --input")
        if not os.path.exists(args.input):
            parser.error(f"Input file not found: {args.input}")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            min_z, max_z = complex_wavelet_edf_from_file(
                args.input, args.output,
                args.wavelet, args.levels, args.bit_depth, args.top_n)
            log.info("DONE | z_range=%d-%d output=%s", min_z, max_z, args.output)
        except Exception as e:
            log.error("FAILED | %s", e)
            sys.exit(1)

if __name__ == "__main__":
    main()