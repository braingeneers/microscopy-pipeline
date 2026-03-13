import cv2
import numpy as np
import argparse
import shutil
import os

def align(im1, im2, timepoint, num, output_folder):
    """
    Aligns im2 to im1 using ECC algorithm and saves the aligned image.

    Parameters:
        im1 (numpy.ndarray): Reference image to align to.
        im2 (numpy.ndarray): Image to be aligned.
        timepoint (int): Timepoint identifier for output naming.
        num (int): Z-stack or image number for output naming.
        output_folder (str): Directory where output image will be saved.

    Returns:
        None
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    
    # Find size of image1
    sz = im1.shape
    
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Specify the number of iterations.
    number_of_iterations = 5000
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    #save the aligned image
    output_path = os.path.join(output_folder, f"{timepoint}_{num}.png")
    cv2.imwrite(output_path, im2_aligned)
    print(f"Aligning image number {num} for timepoint {timepoint} and saving to {output_path}")

parser = argparse.ArgumentParser(
    description="Align photos with specified parameters.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
EXAMPLES:
  # Standard temporal alignment (default output to workingfolder/aligned)
  python alignphotos.py --mintimepoint 5 --maxtimepoint 10 --minzstack 1 --maxzstack 10 --zstackincrement 1 --base 5 --workingfolder ./images
  
  # Custom output folder
  python alignphotos.py --mintimepoint 5 --maxtimepoint 10 --minzstack 1 --maxzstack 10 --zstackincrement 1 --base 5 --workingfolder ./images --output-folder ./results/aligned_images
  
  # Disable temporal alignment with custom output
  python alignphotos.py --mintimepoint 5 --maxtimepoint 10 --minzstack 1 --maxzstack 10 --zstackincrement 1 --base 5 --workingfolder ./images --output-folder ./no_temporal_results --no-temporal
  
  # Use reference image with custom output folder
  python alignphotos.py --mintimepoint 5 --maxtimepoint 10 --minzstack 1 --maxzstack 10 --zstackincrement 1 --base 5 --workingfolder ./images --reference-image ./reference.png --output-folder ./ref_aligned
  
  # Global reference alignment with custom output
  python alignphotos.py --mintimepoint 5 --maxtimepoint 10 --minzstack 1 --maxzstack 10 --zstackincrement 1 --base 5 --workingfolder ./images --reference-image ./reference.png --no-temporal --output-folder ./global_aligned

ALIGNMENT MODES:
  - Default: Temporal alignment - each timepoint's base image aligned to previous timepoint
  - --no-temporal: No temporal alignment - only align z-stacks within each timepoint
  - --reference-image: First timepoint's base image aligned to reference, then temporal alignment continues
  - --reference-image + --no-temporal: All timepoints aligned to reference image (global alignment)
  
OUTPUT FOLDER:
  - Default: {workingfolder}/aligned/
  - Custom: Specify with --output-folder (can be relative or absolute path)
  - Output folder is created automatically if it doesn't exist
  
TEMPORAL ALIGNMENT BEHAVIOR:
  - Default: Base images form temporal chain (t1 → t2 → t3 → ...)
  - --no-temporal: Each timepoint independent (base images not aligned to each other)
  - Z-stack alignment within timepoints is always performed regardless of temporal setting
    """
)

parser.add_argument('--mintimepoint', type=int, required=True, help='Minimum timepoint')
parser.add_argument('--maxtimepoint', type=int, required=True, help='Maximum timepoint')
parser.add_argument('--minzstack', type=int, required=True, help='Minimum z-stack')
parser.add_argument('--maxzstack', type=int, required=True, help='Maximum z-stack')
parser.add_argument('--zstackincrement', type=int, required=True, help='Z-stack increment')
parser.add_argument('--base', type=int, required=True, help='Base z-stack value')
parser.add_argument('--workingfolder', type=str, required=True, help='Working folder for input images')
parser.add_argument('--output-folder', type=str, help='Output folder for aligned images (default: workingfolder/aligned)')
parser.add_argument('--reference-image', type=str, help='Path to reference image file to align first timepoint base image to')
parser.add_argument('--no-temporal', action='store_true', 
                   help='Disable temporal alignment between timepoints (only align z-stacks within each timepoint)')

args = parser.parse_args()

mintimepoint = args.mintimepoint
maxtimepoint = args.maxtimepoint
minzstack = args.minzstack
maxzstack = args.maxzstack
zstackincrement = args.zstackincrement
base = args.base
workingfolder = args.workingfolder
output_folder = args.output_folder if args.output_folder else os.path.join(workingfolder, "aligned")
reference_image_path = args.reference_image
no_temporal = args.no_temporal

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Load reference image if specified
reference_image = None
if reference_image_path:
    if not os.path.exists(reference_image_path):
        print(f"Error: Reference image file not found: {reference_image_path}")
        exit(1)
    
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"Error: Could not read reference image: {reference_image_path}")
        exit(1)
    
    if no_temporal:
        print(f"Using global reference image for all timepoints: {reference_image_path}")
    else:
        print(f"Using reference image for first timepoint: {reference_image_path}")
    print(f"Reference image shape: {reference_image.shape}")
else:
    if no_temporal:
        print("No temporal alignment - each timepoint processed independently")
    else:
        print("Using standard temporal alignment")

print(f"Working folder (input): {os.path.abspath(workingfolder)}")
print(f"Output folder: {os.path.abspath(output_folder)}")
print(f"Timepoint range: {mintimepoint} to {maxtimepoint}")
print(f"Z-stack range: {minzstack} to {maxzstack} (increment: {zstackincrement})")
print(f"Base z-stack: {base}")
print(f"Temporal alignment: {'Disabled' if no_temporal else 'Enabled'}")
print("-" * 60)

for timepoint in range(mintimepoint, maxtimepoint + 1):
    print(f"\nProcessing timepoint {timepoint}:")
    
    # Handle reference image alignment
    if reference_image is not None:
        if no_temporal:
            # Global alignment mode: align all timepoints to reference
            curr_base_img = cv2.imread(os.path.join(workingfolder, f"{timepoint}_zs+{base}.png"))
            if curr_base_img is not None:
                print(f"  Aligning timepoint {timepoint} base image (z={base}) to global reference")
                align(reference_image, curr_base_img, timepoint, base, output_folder)
            else:
                print(f"  Error: Could not read base image for timepoint {timepoint}")
        elif timepoint == mintimepoint:
            # Reference alignment for first timepoint only
            curr_base_img = cv2.imread(os.path.join(workingfolder, f"{timepoint}_zs+{base}.png"))
            if curr_base_img is not None:
                print(f"  Aligning first timepoint base image (z={base}) to reference image")
                align(reference_image, curr_base_img, timepoint, base, output_folder)
            else:
                print(f"  Error: Could not read first timepoint base image")
    
    # Handle temporal alignment (only if not disabled and no global reference)
    if not no_temporal and timepoint > mintimepoint and (reference_image is None or not no_temporal):
        # For temporal alignment, we need to read the previously aligned base image
        prev_base_aligned_path = os.path.join(output_folder, f"{timepoint-1}_{base}.png")
        if os.path.exists(prev_base_aligned_path):
            prev_base_img = cv2.imread(prev_base_aligned_path)
        else:
            # Fallback to original image if aligned version doesn't exist
            prev_base_img = cv2.imread(os.path.join(workingfolder, f"{timepoint-1}_zs+{base}.png"))
        
        curr_base_img = cv2.imread(os.path.join(workingfolder, f"{timepoint}_zs+{base}.png"))
        if prev_base_img is not None and curr_base_img is not None:
            print(f"  Aligning base images for timepoint {timepoint} with base {base} to previous timepoint {timepoint-1}")
            align(prev_base_img, curr_base_img, timepoint, base, output_folder)
        else:
            print(f"  Warning: Could not align base images for timepoint {timepoint}.")
    
    # Handle base image copying if not already aligned
    base_output_path = os.path.join(output_folder, f"{timepoint}_{base}.png")
    if not os.path.exists(base_output_path):
        base_source_path = os.path.join(workingfolder, f"{timepoint}_zs+{base}.png")
        if os.path.exists(base_source_path):
            print(f"  Copying base image for timepoint {timepoint} to output folder")
            shutil.copy(base_source_path, base_output_path)
        else:
            print(f"  Error: Base image not found at {base_source_path}")
            continue
    
    # Align all other z-stack images to the base image (always performed)
    for i in range(minzstack, maxzstack + 1, zstackincrement):
        if i == base:
            print(f"  Skipping base z-stack {base} (already processed)")
            continue
        
        im1 = cv2.imread(base_output_path)
        im2_path = os.path.join(workingfolder, f"{timepoint}_zs+{i}.png")
        im2 = cv2.imread(im2_path)
        
        if im1 is None:
            print(f"  Error: Could not read base image {base_output_path}. Skipping z-stack {i}.")
            continue
        if im2 is None:
            print(f"  Warning: Could not read image {im2_path}. Skipping z-stack {i}.")
            continue
        
        print(f"  Aligning z-stack {i} to base image (z={base})")
        align(im1, im2, timepoint, i, output_folder)

print(f"\nAlignment complete! Aligned images saved to: {os.path.abspath(output_folder)}")