import cv2
import numpy as np
import argparse
import shutil
import os
import sys

from pipeline_log import get_logger
log = get_logger("alignphotos")


# Core alignment 


def align(im1, im2, timepoint, num, output_folder):
    """
    Aligns im2 to im1 using ECC algorithm and saves the aligned image.

    Parameters:
        im1 (numpy.ndarray): Reference image to align to.
        im2 (numpy.ndarray): Image to be aligned.
        timepoint (int): Timepoint identifier for output naming.
        num (int): Z-stack or image number for output naming.
        output_folder (str): Directory where output image will be saved.
    """
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    sz = im1.shape
    warp_mode = cv2.MOTION_TRANSLATION

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(
        im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        im2_aligned = cv2.warpPerspective(
            im2, warp_matrix, (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        im2_aligned = cv2.warpAffine(
            im2, warp_matrix, (sz[1], sz[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    output_path = os.path.join(output_folder, f"{timepoint}_{num}.png")
    cv2.imwrite(output_path, im2_aligned)
    log.info("Aligned tp=%d z=%d -> %s", timepoint, num, output_path)


# CLI + main loop — print() replaced with log

def main():
    parser = argparse.ArgumentParser(
        description="Align photos with specified parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Standard temporal alignment:
    python alignphotos.py --mintimepoint 5 --maxtimepoint 10 --minzstack 1 --maxzstack 10 --zstackincrement 1 --base 5 --workingfolder ./images

  Custom output folder:
    python alignphotos.py --mintimepoint 5 --maxtimepoint 10 --minzstack 1 --maxzstack 10 --zstackincrement 1 --base 5 --workingfolder ./images --output-folder ./results/aligned

  Global reference alignment (all timepoints aligned to one reference):
    python alignphotos.py --mintimepoint 5 --maxtimepoint 10 --minzstack 1 --maxzstack 10 --zstackincrement 1 --base 5 --workingfolder ./images --reference-image ./reference.png --no-temporal

ALIGNMENT MODES
  Default:                    Temporal chain (t1→t2→t3→...)
  --no-temporal:              Each timepoint independent
  --reference-image:          First timepoint aligned to reference, then temporal chain continues
  --reference-image + --no-temporal: All timepoints aligned to reference (global alignment)
        """
    )

    parser.add_argument('--mintimepoint',    type=int, required=True, help='Minimum timepoint')
    parser.add_argument('--maxtimepoint',    type=int, required=True, help='Maximum timepoint')
    parser.add_argument('--minzstack',       type=int, required=True, help='Minimum z-stack')
    parser.add_argument('--maxzstack',       type=int, required=True, help='Maximum z-stack')
    parser.add_argument('--zstackincrement', type=int, required=True, help='Z-stack increment')
    parser.add_argument('--base',            type=int, required=True, help='Base z-stack value')
    parser.add_argument('--workingfolder',   type=str, required=True, help='Working folder for input images')
    parser.add_argument('--output-folder',   type=str,
                        help='Output folder for aligned images (default: workingfolder/aligned)')
    parser.add_argument('--reference-image', type=str,
                        help='Path to reference image to align first timepoint base image to')
    parser.add_argument('--no-temporal',     action='store_true',
                        help='Disable temporal alignment between timepoints')

    args = parser.parse_args()

    mintimepoint    = args.mintimepoint
    maxtimepoint    = args.maxtimepoint
    minzstack       = args.minzstack
    maxzstack       = args.maxzstack
    zstackincrement = args.zstackincrement
    base            = args.base
    workingfolder   = args.workingfolder
    output_folder   = args.output_folder or os.path.join(workingfolder, "aligned")
    reference_image_path = args.reference_image
    no_temporal     = args.no_temporal

    os.makedirs(output_folder, exist_ok=True)

    # Load reference image if specified
    reference_image = None
    if reference_image_path:
        if not os.path.exists(reference_image_path):
            log.error("Reference image not found: %s", reference_image_path)
            sys.exit(1)
        reference_image = cv2.imread(reference_image_path)
        if reference_image is None:
            log.error("Could not read reference image: %s", reference_image_path)
            sys.exit(1)
        log.info("Reference image: %s shape=%s mode=%s",
                 reference_image_path, reference_image.shape,
                 "global" if no_temporal else "first-timepoint-only")
    else:
        log.info("Temporal alignment: %s", "disabled" if no_temporal else "enabled")

    log.info("ALIGN START | input=%s output=%s tp=%d-%d zs=%d-%d step=%d base=%d",
             os.path.abspath(workingfolder), os.path.abspath(output_folder),
             mintimepoint, maxtimepoint, minzstack, maxzstack, zstackincrement, base)

    for timepoint in range(mintimepoint, maxtimepoint + 1):
        log.info("Processing timepoint %d:", timepoint)

        # Reference image alignment
        if reference_image is not None:
            if no_temporal:
                curr_base_img = cv2.imread(
                    os.path.join(workingfolder, f"{timepoint}_zs+{base}.png"))
                if curr_base_img is not None:
                    log.info("  Aligning tp=%d base z=%d to global reference", timepoint, base)
                    align(reference_image, curr_base_img, timepoint, base, output_folder)
                else:
                    log.error("  Could not read base image for timepoint %d", timepoint)
            elif timepoint == mintimepoint:
                curr_base_img = cv2.imread(
                    os.path.join(workingfolder, f"{timepoint}_zs+{base}.png"))
                if curr_base_img is not None:
                    log.info("  Aligning first tp=%d base z=%d to reference", timepoint, base)
                    align(reference_image, curr_base_img, timepoint, base, output_folder)
                else:
                    log.error("  Could not read first timepoint base image")

        # Temporal alignment
        if not no_temporal and timepoint > mintimepoint and (reference_image is None or not no_temporal):
            prev_base_aligned_path = os.path.join(output_folder, f"{timepoint-1}_{base}.png")
            if os.path.exists(prev_base_aligned_path):
                prev_base_img = cv2.imread(prev_base_aligned_path)
            else:
                prev_base_img = cv2.imread(
                    os.path.join(workingfolder, f"{timepoint-1}_zs+{base}.png"))

            curr_base_img = cv2.imread(
                os.path.join(workingfolder, f"{timepoint}_zs+{base}.png"))

            if prev_base_img is not None and curr_base_img is not None:
                log.info("  Temporal align tp=%d base z=%d to tp=%d",
                         timepoint, base, timepoint - 1)
                align(prev_base_img, curr_base_img, timepoint, base, output_folder)
            else:
                log.warning("  Could not align base images for timepoint %d", timepoint)

        # Copy base image if not already aligned
        base_output_path = os.path.join(output_folder, f"{timepoint}_{base}.png")
        if not os.path.exists(base_output_path):
            base_source_path = os.path.join(workingfolder, f"{timepoint}_zs+{base}.png")
            if os.path.exists(base_source_path):
                log.info("  Copying base image tp=%d z=%d to output", timepoint, base)
                shutil.copy(base_source_path, base_output_path)
            else:
                log.error("  Base image not found: %s", base_source_path)
                continue

        # Align all other z-stacks to the base
        for i in range(minzstack, maxzstack + 1, zstackincrement):
            if i == base:
                log.info("  Skipping base z=%d (already processed)", base)
                continue

            im1 = cv2.imread(base_output_path)
            im2_path = os.path.join(workingfolder, f"{timepoint}_zs+{i}.png")
            im2 = cv2.imread(im2_path)

            if im1 is None:
                log.error("  Could not read base image %s, skipping z=%d", base_output_path, i)
                continue
            if im2 is None:
                log.warning("  Could not read %s, skipping z=%d", im2_path, i)
                continue

            log.info("  Aligning z=%d to base z=%d for tp=%d", i, base, timepoint)
            align(im1, im2, timepoint, i, output_folder)

    log.info("ALIGN END | status=complete output=%s", os.path.abspath(output_folder))


if __name__ == "__main__":
    main()