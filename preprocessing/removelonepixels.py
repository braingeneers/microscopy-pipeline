import os
import sys
import time
import numpy as np
from PIL import Image
from collections import defaultdict

def group_images_by_prefix(folder):
    """
    Group image files by their number prefix (before first underscore).
    Returns a dictionary where keys are prefixes and values are lists of filenames.
    """
    if not os.path.exists(folder):
        print(f"Error: Input folder '{folder}' does not exist")
        return {}
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    groups = defaultdict(list)
    
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(image_extensions):
            # Extract prefix before first underscore
            if '_' in fname:
                prefix = fname.split('_')[0]
            else:
                prefix = fname.split('.')[0]  # Use filename without extension if no underscore
            
            groups[prefix].append(fname)
    
    print(f"Found {len(groups)} groups:")
    for prefix, files in groups.items():
        print(f"  Group '{prefix}': {len(files)} images")
    
    return groups

def load_images(folder, filenames):
    """
    Load specific image files from a folder and convert to grayscale numpy arrays.
    Returns images and their filenames.
    """
    images = []
    loaded_filenames = []
    
    for fname in filenames:
        try:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('L')
            images.append(np.array(img))
            loaded_filenames.append(fname)
            print(f"    Loaded: {fname}")
        except Exception as e:
            print(f"    Warning: Could not load {fname}: {e}")
    
    print(f"    Successfully loaded {len(images)} images")
    return images, loaded_filenames

def find_lone_pixels(images, threshold=0.2, min_fraction=0.2):
    """
    Find pixels that are consistently brighter than their neighbors across multiple images.
    
    Parameters:
    - images: List of numpy arrays
    - threshold: Minimum brightness difference (as fraction) to consider a pixel "lone"
    - min_fraction: Minimum fraction of images where pixel must be lone to be marked
    
    Returns:
    - lone_pixel_mask: Boolean array indicating lone pixels
    """
    if not images:
        return np.array([])
    
    h, w = images[0].shape
    n_images = len(images)
    
    print(f"    Analyzing {n_images} images of size {h}x{w}")
    print(f"    Threshold: {threshold:.1%}, Min fraction: {min_fraction:.1%}")
    
    # Stack all images into a 3D array for vectorized processing
    print("    Stacking images for vectorized processing...")
    image_stack = np.stack([img.astype(np.float32) for img in images], axis=0)  # Shape: (n_images, h, w)
    
    # Pad all images at once
    print("    Padding images...")
    padded_stack = np.pad(image_stack, ((0, 0), (1, 1), (1, 1)), mode='edge')  # Shape: (n_images, h+2, w+2)
    
    # Extract center pixels and all 8 neighbors for all images at once
    print("    Extracting neighbors for all images...")
    center_stack = padded_stack[:, 1:-1, 1:-1]  # Shape: (n_images, h, w)
    
    # Extract all 8 neighbors using array slicing - shape will be (8, n_images, h, w)
    neighbors_stack = np.stack([
        padded_stack[:, 0:-2, 1:-1],   # up
        padded_stack[:, 2:, 1:-1],     # down
        padded_stack[:, 1:-1, 0:-2],   # left
        padded_stack[:, 1:-1, 2:],     # right
        padded_stack[:, 0:-2, 0:-2],   # up-left
        padded_stack[:, 0:-2, 2:],     # up-right
        padded_stack[:, 2:, 0:-2],     # down-left
        padded_stack[:, 2:, 2:],       # down-right
    ], axis=0)  # Shape: (8, n_images, h, w)
    
    # Vectorized comparison across all images and neighbors simultaneously
    print("    Computing brightness comparisons...")
    threshold_neighbors = neighbors_stack * (1 + threshold)  # Shape: (8, n_images, h, w)
    
    # Check if center pixel is brighter than ALL neighbors for each image
    # center_stack[np.newaxis, :, :, :] broadcasts to (1, n_images, h, w)
    # Compare against (8, n_images, h, w) and reduce along neighbor axis (axis=0)
    is_brighter_than_all = np.all(center_stack[np.newaxis, :, :, :] > threshold_neighbors, axis=0)  # Shape: (n_images, h, w)
    
    # Count how many images each pixel is lone in
    print("    Counting lone pixels across images...")
    count = np.sum(is_brighter_than_all.astype(np.int32), axis=0)  # Shape: (h, w)
    
    # Mark pixels that are lone in at least min_fraction of images
    min_count = int(n_images * min_fraction)
    lone_pixel_mask = count >= min_count
    
    num_lone_pixels = np.sum(lone_pixel_mask)
    print(f"    Found {num_lone_pixels} lone pixels ({num_lone_pixels/(h*w):.3%} of total pixels)")
    
    return lone_pixel_mask

def correct_images(images, lone_pixel_mask, out_folder, filenames, group_prefix):
    """
    Correct lone pixels by replacing them with the average of their neighbors.
    
    Parameters:
    - images: List of input images
    - lone_pixel_mask: Boolean mask indicating which pixels to correct
    - out_folder: Output directory
    - filenames: Original filenames
    - group_prefix: Prefix for the current group
    """
    if not images:
        return
    
    h, w = images[0].shape
    
    # Create output folder
    os.makedirs(out_folder, exist_ok=True)
    print(f"    Saving corrected images to: {out_folder}")
    
    for idx, img in enumerate(images):
        print(f"    Correcting image {idx + 1}/{len(images)}: {filenames[idx]}")
        
        img_corr = img.copy()
        padded = np.pad(img_corr, 1, mode='edge')
        
        # Count corrections made
        corrections_made = 0
        
        for y in range(h):
            for x in range(w):
                if lone_pixel_mask[y, x]:
                    # Get 4-connected neighbors
                    neighbors = [
                        padded[y, x+1],     # up
                        padded[y+2, x+1],   # down
                        padded[y+1, x],     # left
                        padded[y+1, x+2],   # right
                        padded[y, x],       # up-left
                        padded[y, x+2],     # up-right
                        padded[y+2, x],     # down-left
                        padded[y+2, x+2],   # down-right
                    ]
                    
                    # Replace with average of neighbors
                    img_corr[y, x] = int(np.mean(neighbors))
                    corrections_made += 1
        
        print(f"      Made {corrections_made} pixel corrections")
        
        # Save corrected image
        out_img = Image.fromarray(img_corr.astype(np.uint8))
        out_path = os.path.join(out_folder, filenames[idx])
        out_img.save(out_path)

def process_group(input_folder, output_folder, group_prefix, filenames, threshold, min_fraction):
    """
    Process a single group of images.
    """
    print(f"\n=== Processing Group '{group_prefix}' ===")
    print(f"  {len(filenames)} images in group")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Load images for this group
        print("  Loading images...")
        load_start = time.time()
        images, loaded_filenames = load_images(input_folder, filenames)
        load_time = time.time() - load_start
        print(f"    Loading took: {load_time:.2f} seconds")
        
        if not images:
            print(f"  Warning: No images loaded for group '{group_prefix}'")
            return
        
        # Find lone pixels for this group
        print("  Finding lone pixels...")
        analysis_start = time.time()
        lone_pixel_mask = find_lone_pixels(images, threshold, min_fraction)
        analysis_time = time.time() - analysis_start
        print(f"    Analysis took: {analysis_time:.2f} seconds")
        
        # Correct images for this group
        print("  Correcting images...")
        correction_start = time.time()
        correct_images(images, lone_pixel_mask, output_folder, loaded_filenames, group_prefix)
        correction_time = time.time() - correction_start
        print(f"    Correction took: {correction_time:.2f} seconds")
        
        # Calculate and print total time
        total_time = time.time() - start_time
        print(f"  Group '{group_prefix}' processing complete!")
        print(f"  Total time for group '{group_prefix}': {total_time:.2f} seconds")
        print(f"    - Loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
        print(f"    - Analysis: {analysis_time:.2f}s ({analysis_time/total_time*100:.1f}%)")
        print(f"    - Correction: {correction_time:.2f}s ({correction_time/total_time*100:.1f}%)")
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"  Error processing group '{group_prefix}' after {total_time:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to handle command line arguments and execute lone pixel removal.
    """
    if len(sys.argv) < 3:
        print("Usage: python removelonepixels.py <input_folder> <output_folder> [threshold] [min_fraction] [start_prefix]")
        print()
        print("Parameters:")
        print("  input_folder   - Folder containing input images")
        print("  output_folder  - Folder to save corrected images")
        print("  threshold      - Brightness threshold (default: 0.2 = 20% brighter)")
        print("  min_fraction   - Minimum fraction of images where pixel must be lone (default: 0.2 = 20%)")
        print("  start_prefix   - Start processing from this prefix (default: process all)")
        print()
        print("Note: Images are grouped by number prefix before first underscore (_)")
        print("Each group is processed separately to find lone pixels within that group.")
        print()
        print("Examples:")
        print("  python removelonepixels.py ./images ./corrected")
        print("  python removelonepixels.py ./images ./corrected 0.3 0.1")
        print("  python removelonepixels.py ./images ./corrected 0.15 0.25 50")
        print("  python removelonepixels.py ./raw ./processed 0.2 0.2 100")
        sys.exit(1)
    
    # Parse command line arguments
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Optional parameters with defaults
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    min_fraction = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    start_prefix = sys.argv[5] if len(sys.argv) > 5 else None
    
    # Validate parameters
    if threshold <= 0 or threshold > 1:
        print("Error: Threshold must be between 0 and 1")
        sys.exit(1)
    
    if min_fraction <= 0 or min_fraction > 1:
        print("Error: Min fraction must be between 0 and 1")
        sys.exit(1)
    
    print("=== Lone Pixel Removal (Grouped Processing) ===")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Threshold: {threshold:.1%}")
    print(f"Min fraction: {min_fraction:.1%}")
    if start_prefix:
        print(f"Starting from prefix: {start_prefix}")
    print()
    
    try:
        # Group images by prefix
        print("Grouping images by prefix...")
        image_groups = group_images_by_prefix(input_folder)
        
        if not image_groups:
            print("Error: No image groups found")
            sys.exit(1)
        
        # Sort groups by prefix numerically
        def sort_key(prefix):
            try:
                return int(prefix)
            except ValueError:
                # If prefix is not a number, sort alphabetically at the end
                return float('inf')
        
        sorted_groups = sorted(image_groups.items(), key=lambda x: sort_key(x[0]))
        
        # Filter groups to start from specified prefix
        if start_prefix:
            # Convert start_prefix to int for comparison if possible
            try:
                start_prefix_int = int(start_prefix)
            except ValueError:
                start_prefix_int = None
            
            # Find starting position
            start_found = False
            filtered_groups = []
            
            for prefix, filenames in sorted_groups:
                try:
                    prefix_int = int(prefix)
                    # If both are integers, compare numerically
                    if start_prefix_int is not None:
                        if prefix_int >= start_prefix_int:
                            start_found = True
                    else:
                        # If start_prefix is not a number, compare as strings
                        if prefix == start_prefix:
                            start_found = True
                except ValueError:
                    # If prefix is not a number, compare as strings
                    if prefix == start_prefix:
                        start_found = True
                
                if start_found:
                    filtered_groups.append((prefix, filenames))
            
            if not filtered_groups:
                print(f"Error: Start prefix '{start_prefix}' not found or no groups after it. Available prefixes:")
                for prefix, _ in sorted_groups:
                    print(f"  {prefix}")
                sys.exit(1)
            
            sorted_groups = filtered_groups
            print(f"Starting from prefix '{start_prefix}', processing {len(sorted_groups)} groups")
        
        # Show processing order
        print("Processing order:")
        for i, (prefix, filenames) in enumerate(sorted_groups):
            print(f"  {i+1}. Group '{prefix}': {len(filenames)} images")
        
        total_images = sum(len(files) for prefix, files in sorted_groups)
        print(f"Total images to process: {total_images}")
        
        # Start overall timing
        overall_start_time = time.time()
        
        # Process each group in numerical order
        group_times = []
        for i, (group_prefix, filenames) in enumerate(sorted_groups):
            group_start = time.time()
            process_group(input_folder, output_folder, group_prefix, filenames, threshold, min_fraction)
            group_time = time.time() - group_start
            group_times.append((group_prefix, group_time, len(filenames)))
            
            # Show progress
            print(f"  Progress: {i+1}/{len(sorted_groups)} groups completed")
        
        # Calculate and print overall statistics
        overall_time = time.time() - overall_start_time
        
        print(f"\n=== All Groups Processing Complete ===")
        print(f"Processed {len(sorted_groups)} groups with {total_images} total images")
        print(f"Total processing time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
        print(f"Average time per group: {overall_time/len(sorted_groups):.2f} seconds")
        print(f"Average time per image: {overall_time/total_images:.2f} seconds")
        print(f"Corrected images saved to: {output_folder}")
        
        # Show timing summary for each group
        print(f"\n=== Timing Summary ===")
        for prefix, group_time, num_images in group_times:
            avg_per_image = group_time / num_images if num_images > 0 else 0
            print(f"Group '{prefix}': {group_time:.2f}s total, {avg_per_image:.2f}s per image ({num_images} images)")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()