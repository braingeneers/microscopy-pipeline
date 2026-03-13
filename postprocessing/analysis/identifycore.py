import cv2
import numpy as np
from skimage import measure, morphology, segmentation
from scipy import ndimage
import sys
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from pathlib import Path

def identify_core_regions(image_path, core_thresh_percentile=90, outer_thresh_percentile=40, save_contours=False, output_dir=None):
    """
    Identify central core and outer regions in an image.
    
    Core region: High brightness, single connected component, high convexity, filled internal voids
    Outer region: Moderate brightness, contains core, filled internal voids
    
    Args:
        image_path: Path to input image
        core_thresh_percentile: Brightness percentile for core threshold (default: 90)
        outer_thresh_percentile: Lower brightness percentile for outer threshold (default: 40)
        save_contours: Whether to save contour overlay images
        output_dir: Directory to save contour images (if save_contours=True)
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    # Normalize image
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply slight Gaussian blur to reduce noise for outer region processing
    img_smooth = cv2.GaussianBlur(img_norm, (3, 3), 0)
    
    # Apply more aggressive Gaussian blur specifically for core detection
    img_core_smooth = cv2.GaussianBlur(img_norm, (9, 9), 2.0)
    
    # Apply CLAHE for outer region detection only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_smooth)
    
    # === IDENTIFY CENTRAL CORE REGION (using aggressively smoothed image) ===
    # Use high intensity threshold for core on heavily blurred image
    core_thresh = np.percentile(img_core_smooth, core_thresh_percentile)
    core_mask = img_core_smooth > core_thresh
    
    # Remove small objects and clean up
    core_mask = morphology.remove_small_objects(core_mask, min_size=1000)
    
    # Fill holes VERY aggressively - multiple passes to ensure no holes remain
    for _ in range(3):  # Multiple passes
        core_mask = ndimage.binary_fill_holes(core_mask)
        # Also use morphological closing to fill any remaining gaps
        core_mask = morphology.binary_closing(core_mask, morphology.disk(15))
    
    # Additional aggressive hole filling using flood fill from edges
    filled_mask = core_mask.copy()
    # Create a padded version to ensure edge connectivity
    padded_mask = np.pad(~filled_mask, 1, mode='constant', constant_values=True)
    # Flood fill from edges (this fills all areas connected to edges)
    from scipy.ndimage import binary_fill_holes as fill_holes
    background = ndimage.binary_erosion(padded_mask, structure=np.ones((3,3)))
    background = ndimage.label(background)[0] == 1  # Keep only the background component connected to edges
    background = background[1:-1, 1:-1]  # Remove padding
    core_mask = ~background | core_mask  # Union with original to ensure no loss of core regions
    
    # Final aggressive hole filling
    core_mask = ndimage.binary_fill_holes(core_mask)
    
    # Apply closing to smooth boundaries and fill any remaining tiny holes
    core_mask = morphology.binary_closing(core_mask, morphology.disk(10))
    
    # FORCE SINGLE CONNECTED COMPONENT - select largest only
    core_label = measure.label(core_mask)
    core_props = measure.regionprops(core_label)
    
    if not core_props:
        print(f"Could not identify core region in {image_path}")
        return None
    
    # Select only the largest core region to ensure single component
    largest_core = max(core_props, key=lambda x: x.area)
    core_mask = core_label == largest_core.label
    
    # FINAL hole filling pass on the selected component
    core_mask = ndimage.binary_fill_holes(core_mask)
    
    # Check convexity and require higher standard
    current_convexity = largest_core.area / largest_core.convex_area
    
    # Require higher convexity (80% instead of 70%)
    if current_convexity < 0.8:
        print(f"  Core convexity {current_convexity:.3f} below threshold, applying iterative smoothing...")
        
        # Apply iterative smoothing until convexity is reached
        smoothed_mask = core_mask.copy()
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            # Calculate current convexity
            current_label = measure.label(smoothed_mask)
            current_props = measure.regionprops(current_label)
            
            if not current_props:
                break
                
            largest_current = max(current_props, key=lambda x: x.area)
            current_conv = largest_current.area / largest_current.convex_area
            
            print(f"    Iteration {iteration + 1}: convexity = {current_conv:.3f}")
            
            # Check if we've reached target convexity
            if current_conv >= 0.8:
                print(f"    Target convexity reached after {iteration + 1} iterations")
                break
            
            # Apply progressive smoothing based on how far we are from target
            convexity_deficit = 0.8 - current_conv
            
            if convexity_deficit > 0.15:
                # Large deficit: more aggressive smoothing
                smoothed_mask = morphology.binary_closing(smoothed_mask, morphology.disk(20))
                smoothed_mask = morphology.binary_opening(smoothed_mask, morphology.disk(5))
            elif convexity_deficit > 0.08:
                # Medium deficit: moderately aggressive smoothing
                smoothed_mask = morphology.binary_closing(smoothed_mask, morphology.disk(15))
                smoothed_mask = morphology.binary_opening(smoothed_mask, morphology.disk(3))
            else:
                # Small deficit: gentle but more effective smoothing
                smoothed_mask = morphology.binary_closing(smoothed_mask, morphology.disk(8))
            
            # Ensure we still have a single component
            smoothed_label = measure.label(smoothed_mask)
            smoothed_props = measure.regionprops(smoothed_label)
            
            if smoothed_props:
                largest_smoothed = max(smoothed_props, key=lambda x: x.area)
                smoothed_mask = smoothed_label == largest_smoothed.label
            
            # Check if area expanded too much (>50% growth from original)
            original_area = largest_core.area
            current_area = np.sum(smoothed_mask)
            area_expansion = current_area / original_area
            
            if area_expansion > 1.5:
                print(f"    Area expansion {area_expansion:.2f}x too large, stopping smoothing")
                break
            
            iteration += 1
        
        # Use smoothed result if it improved convexity without excessive area growth
        final_label = measure.label(smoothed_mask)
        final_props = measure.regionprops(final_label)
        
        if final_props:
            largest_final = max(final_props, key=lambda x: x.area)
            final_convexity = largest_final.area / largest_final.convex_area
            final_area = largest_final.area
            final_expansion = final_area / largest_core.area
            
            # Accept smoothed result if convexity improved and area didn't expand too much
            if final_convexity > current_convexity and final_expansion <= 1.5:
                core_mask = smoothed_mask
                print(f"    Applied smoothing: convexity improved from {current_convexity:.3f} to {final_convexity:.3f}")
            else:
                print(f"    Smoothing unsuccessful, keeping original mask")
        
        # If smoothing failed to reach target, try convex hull as fallback
        final_label = measure.label(core_mask)
        final_props = measure.regionprops(final_label)
        
        if final_props:
            largest_final = max(final_props, key=lambda x: x.area)
            final_convexity = largest_final.area / largest_final.convex_area
            
            if final_convexity < 0.75:  # Lower threshold for convex hull fallback
                print(f"    Smoothing insufficient (convexity {final_convexity:.3f}), trying convex hull fallback...")
                core_coords = largest_final.coords
                
                if len(core_coords) > 3:
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(core_coords)
                        # Create mask from convex hull
                        core_convex_mask = np.zeros_like(core_mask)
                        hull_coords = core_coords[hull.vertices]
                        # Fill the convex hull polygon
                        from skimage.draw import polygon
                        rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], core_convex_mask.shape)
                        valid_idx = (rr >= 0) & (rr < core_convex_mask.shape[0]) & (cc >= 0) & (cc < core_convex_mask.shape[1])
                        core_convex_mask[rr[valid_idx], cc[valid_idx]] = True
                        
                        # Only use convex hull if it doesn't expand the area too much
                        convex_area = np.sum(core_convex_mask)
                        original_area = largest_core.area
                        area_expansion = convex_area / original_area
                        
                        if area_expansion < 1.3:  # Restrictive - don't expand by more than 30%
                            core_mask = core_convex_mask
                            print(f"    Applied convex hull fallback with {area_expansion:.2f}x area expansion")
                        else:
                            print(f"    Convex hull expansion {area_expansion:.2f}x too large, keeping smoothed result")
                    except Exception as e:
                        print(f"    Convex hull fallback failed: {e}")
    else:
        # Light smoothing only for already-convex shapes
        core_mask = morphology.binary_closing(core_mask, morphology.disk(5))
        print(f"  Core convexity {current_convexity:.3f} acceptable, applying light smoothing only")
    
    # Verify single component after processing
    final_core_label = measure.label(core_mask)
    final_core_components = len(measure.regionprops(final_core_label))
    if final_core_components > 1:
        final_core_props = measure.regionprops(final_core_label)
        largest_final = max(final_core_props, key=lambda x: x.area)
        core_mask = final_core_label == largest_final.label

    # === IDENTIFY OUTER REGION (using CLAHE-enhanced image) ===
    # Use moderate intensity threshold on CLAHE-enhanced image
    outer_low_thresh = np.percentile(img_clahe, outer_thresh_percentile)
    outer_high_thresh = np.percentile(img_clahe, 80)
    
    outer_mask = (img_clahe > outer_low_thresh) & (img_clahe < outer_high_thresh)
    
    # Remove small objects
    outer_mask = morphology.remove_small_objects(outer_mask, min_size=5000)
    
    # Fill ALL holes VERY aggressively - multiple passes
    for _ in range(5):  # Even more passes for outer region
        outer_mask = ndimage.binary_fill_holes(outer_mask)
        # Use larger closing operations to fill bigger gaps
        outer_mask = morphology.binary_closing(outer_mask, morphology.disk(25))
    
    # Additional aggressive hole filling using flood fill approach
    filled_outer = outer_mask.copy()
    # Create padded version
    padded_outer = np.pad(~filled_outer, 1, mode='constant', constant_values=True)
    # Flood fill from edges
    outer_background = ndimage.binary_erosion(padded_outer, structure=np.ones((3,3)))
    outer_background = ndimage.label(outer_background)[0] == 1
    outer_background = outer_background[1:-1, 1:-1]  # Remove padding
    outer_mask = ~outer_background | outer_mask
    
    # Final comprehensive hole filling
    outer_mask = ndimage.binary_fill_holes(outer_mask)
    
    # Apply large closing to create smooth, filled regions with no holes
    outer_mask = morphology.binary_closing(outer_mask, morphology.disk(20))
    
    # Additional dilation to ensure it contains the core
    outer_mask = morphology.binary_dilation(outer_mask, morphology.disk(5))
    
    # FINAL hole filling pass
    outer_mask = ndimage.binary_fill_holes(outer_mask)
    
    # Ensure outer region contains core region
    combined_mask = outer_mask | core_mask
    
    # Label regions and find the one that contains the core
    combined_label = measure.label(combined_mask)
    combined_props = measure.regionprops(combined_label)
    
    # Find which region contains the core centroid
    core_centroid = measure.regionprops(measure.label(core_mask))[0].centroid
    
    # Find the combined region that contains the core
    containing_region = None
    for prop in combined_props:
        # Check if core centroid is within this region
        region_mask = combined_label == prop.label
        if region_mask[int(core_centroid[0]), int(core_centroid[1])]:
            containing_region = prop
            break
    
    if containing_region is None:
        print(f"Could not find outer region containing core in {image_path}")
        return None
    
    # Update outer mask to be the containing region minus the core
    containing_region_mask = combined_label == containing_region.label
    
    # Fill holes in the containing region BEFORE subtracting core
    filled_containing_region = ndimage.binary_fill_holes(containing_region_mask)
    
    # Now subtract core to create the outer ring (this creates the intended hole)
    final_outer_mask = filled_containing_region & (~core_mask)
    
    # DO NOT fill holes here - we want to preserve the core hole
    # The core subtraction creates a single hole (the core area) which is intended
    
    # However, if there are OTHER holes (not from core subtraction), we should address them
    # Check if there are holes besides the core hole
    temp_filled = ndimage.binary_fill_holes(final_outer_mask)
    
    # If filling holes creates additional area beyond just the core, there were other holes
    core_area_pixels = np.sum(core_mask)
    additional_filled_area = np.sum(temp_filled) - np.sum(final_outer_mask)
    
    # If additional filled area is significantly larger than core area, 
    # there are other holes we should fill
    if additional_filled_area > core_area_pixels * 1.1:  # 10% tolerance
        print(f"  Detected additional holes beyond core subtraction, filling them...")
        
        # Create a mask of just the core hole
        core_hole_mask = filled_containing_region & core_mask
        
        # Fill all holes in the outer mask
        temp_filled_outer = ndimage.binary_fill_holes(final_outer_mask)
        
        # Then subtract only the core hole back out
        final_outer_mask = temp_filled_outer & (~core_hole_mask)
    
    # The final result should be a ring with only the core hole, no other holes
    
    # === CALCULATE STATISTICS ===
    core_props = measure.regionprops(measure.label(core_mask))[0]
    outer_props = measure.regionprops(measure.label(final_outer_mask))[0]
    
    # Calculate summed intensities from original (non-normalized) image
    core_sum_intensity = np.sum(img[core_mask])
    outer_only_sum_intensity = np.sum(img[final_outer_mask])
    total_sum_intensity = core_sum_intensity + outer_only_sum_intensity
    whole_image_sum = np.sum(img)
    
    # Calculate areas
    total_pixels = img.shape[0] * img.shape[1]
    core_area = core_props.area
    outer_area = outer_props.area
    
    # === SAVE CONTOUR OVERLAY IF REQUESTED ===
    if save_contours and output_dir:
        # Create base image for overlay (convert to RGB)
        overlay = np.stack([img_norm]*3, axis=-1)
        
        # Find contours for core and outer regions
        core_contours = measure.find_contours(core_mask.astype(float), 0.5)
        outer_contours = measure.find_contours(final_outer_mask.astype(float), 0.5)
        
        # Draw thick, visible contours
        def draw_thick_contour(overlay, contours, color, thickness=3):
            for contour in contours:
                contour_coords = contour.astype(int)
                for i in range(len(contour_coords)):
                    y, x = contour_coords[i]
                    # Draw thick contour by drawing neighboring pixels
                    for dy in range(-thickness, thickness+1):
                        for dx in range(-thickness, thickness+1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < overlay.shape[0] and 0 <= nx < overlay.shape[1]:
                                overlay[ny, nx] = color
        
        # Draw contours
        draw_thick_contour(overlay, outer_contours, [0, 255, 0], thickness=3)     # Green outer (first)
        draw_thick_contour(overlay, core_contours, [255, 0, 0], thickness=3)      # Red core (second, on top)
        
        # Save contour overlay
        filename = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{filename}_contours.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f"Contour Overlay: {filename}\n(Red: Core, Green: Outer)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # === DEBUG: SHOW INTERMEDIATE STEPS ===
    # Uncomment the following lines to save intermediate masks for debugging
    """
    debug_dir = output_dir / "debug_masks"
    debug_dir.mkdir(exist_ok=True)
    
    # Save core mask
    core_mask_path = debug_dir / f"{Path(image_path).stem}_core_mask.png"
    cv2.imwrite(str(core_mask_path), (core_mask.astype(np.uint8) * 255))
    
    # Save outer mask
    outer_mask_path = debug_dir / f"{Path(image_path).stem}_outer_mask.png"
    cv2.imwrite(str(outer_mask_path), (final_outer_mask.astype(np.uint8) * 255))
    
    print(f"  Debug masks saved: {core_mask_path}, {outer_mask_path}")
    """
    
    return {
        'filename': Path(image_path).name,
        'core_sum_intensity': core_sum_intensity,
        'outer_sum_intensity': outer_only_sum_intensity,
        'total_image_sum_intensity': whole_image_sum,
        'core_area': core_area,
        'outer_area': outer_area,
        'total_pixels': total_pixels,
        'core_mask': core_mask,
        'outer_mask': final_outer_mask,
        'core_props': core_props,
        'outer_props': outer_props,
        'original_image': img,
        'normalized_image': img_norm
    }

def process_folder(folder_path, core_thresh_percentile=90, outer_thresh_percentile=40):
    """
    Process all images in a folder and save results to CSV.
    """
    folder_path = Path(folder_path)
    
    # Create contours subfolder
    contours_dir = folder_path / "contours"
    contours_dir.mkdir(exist_ok=True)
    
    # Create outer regions subfolder
    outer_regions_dir = folder_path / "outer_regions"
    outer_regions_dir.mkdir(exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}
    
    # Find all image files (avoid duplicates)
    image_files = set()
    for ext in image_extensions:
        image_files.update(folder_path.glob(f"*{ext}"))
        image_files.update(folder_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(list(image_files))  # Convert back to sorted list
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} image files in {folder_path}")
    
    # Setup CSV file
    csv_path = folder_path / "core_analysis_results.csv"
    
    # Create CSV header if file doesn't exist
    if not csv_path.exists():
        header_df = pd.DataFrame(columns=[
            'filename', 'core_sum_intensity', 'outer_sum_intensity', 
            'total_image_sum_intensity', 'core_area', 'outer_area', 'total_pixels',
            'core_area_fraction', 'outer_area_fraction', 
            'core_intensity_fraction', 'outer_intensity_fraction'
        ])
        header_df.to_csv(csv_path, index=False)
        print(f"Created CSV file: {csv_path}")
    
    # Process each image
    successful_count = 0
    failed_files = []
    
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            # Process with both contours and outer region export
            result = identify_core_regions(
                str(image_file), 
                core_thresh_percentile, 
                outer_thresh_percentile,
                save_contours=True,
                output_dir=str(contours_dir)
            )
            
            # Also save outer region to separate directory
            if result:
                # Save outer region image to dedicated folder
                filename = Path(image_file).stem
                outer_only_img = result['normalized_image'].copy()
                outer_only_img[~result['outer_mask']] = 0
                
                # Find bounding box and crop
                outer_coords = np.where(result['outer_mask'])
                if len(outer_coords[0]) > 0:
                    min_row, max_row = outer_coords[0].min(), outer_coords[0].max()
                    min_col, max_col = outer_coords[1].min(), outer_coords[1].max()
                    
                    padding = 10
                    min_row = max(0, min_row - padding)
                    max_row = min(result['normalized_image'].shape[0], max_row + padding + 1)
                    min_col = max(0, min_col - padding)
                    max_col = min(result['normalized_image'].shape[1], max_col + padding + 1)
                    
                    cropped_outer = outer_only_img[min_row:max_row, min_col:max_col]
                    outer_region_path = outer_regions_dir / f"{filename}_outer_region.png"
                    cv2.imwrite(str(outer_region_path), cropped_outer)
            
            if result:
                # Create single row DataFrame
                row_data = {
                    'filename': result['filename'],
                    'core_sum_intensity': result['core_sum_intensity'],
                    'outer_sum_intensity': result['outer_sum_intensity'],
                    'total_image_sum_intensity': result['total_image_sum_intensity'],
                    'core_area': result['core_area'],
                    'outer_area': result['outer_area'],
                    'total_pixels': result['total_pixels'],
                    'core_area_fraction': result['core_area'] / result['total_pixels'],
                    'outer_area_fraction': result['outer_area'] / result['total_pixels'],
                    'core_intensity_fraction': result['core_sum_intensity'] / result['total_image_sum_intensity'],
                    'outer_intensity_fraction': result['outer_sum_intensity'] / result['total_image_sum_intensity']
                }
                
                # Append to CSV immediately
                row_df = pd.DataFrame([row_data])
                row_df.to_csv(csv_path, mode='a', header=False, index=False)
                
                successful_count += 1
                print(f"  ✓ Results appended to CSV")
                
            else:
                print(f"  ✗ Failed to identify regions in {image_file.name}")
                failed_files.append(image_file.name)
                
        except Exception as e:
            print(f"  ✗ Error processing {image_file.name}: {e}")
            failed_files.append(image_file.name)
    
    # Read final CSV for summary statistics
    if successful_count > 0:
        df = pd.read_csv(csv_path)
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"Successfully processed: {successful_count}/{len(image_files)} images")
        print(f"Results saved to: {csv_path}")
        print(f"Contour overlays saved to: {contours_dir}")
        print(f"Outer region images saved to: {outer_regions_dir}")
        
        if failed_files:
            print(f"\nFailed files:")
            for filename in failed_files:
                print(f"  - {filename}")
        
        # Print summary statistics
        print(f"\n=== Summary Statistics ===")
        print(f"Core sum intensity - Mean: {df['core_sum_intensity'].mean():,.0f}, Std: {df['core_sum_intensity'].std():,.0f}")
        print(f"Outer sum intensity - Mean: {df['outer_sum_intensity'].mean():,.0f}, Std: {df['outer_sum_intensity'].std():,.0f}")
        print(f"Core area fraction - Mean: {df['core_area_fraction'].mean():.3%}, Std: {df['core_area_fraction'].std():.3%}")
        print(f"Outer area fraction - Mean: {df['outer_area_fraction'].mean():.3%}, Std: {df['outer_area_fraction'].std():.3%}")
        
    else:
        print("No images were successfully processed.")

def main():
    """
    Main function to handle command line arguments.
    """
    parser = argparse.ArgumentParser(description='Identify bright central core and moderate outer regions.')
    
    # Make image and folder mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='Path to single input image')
    group.add_argument('--folder', help='Path to folder containing images')
    
    parser.add_argument('--core-threshold', type=float, default=90, 
                       help='Percentile threshold for core brightness (default: 90)')
    parser.add_argument('--outer-threshold', type=float, default=40,
                       help='Lower percentile threshold for outer brightness (default: 40)')
    
    args = parser.parse_args()
    
    # Validate thresholds
    if not (0 <= args.core_threshold <= 100):
        print("Error: Core threshold must be between 0 and 100")
        sys.exit(1)
    if not (0 <= args.outer_threshold <= 100):
        print("Error: Outer threshold must be between 0 and 100")
        sys.exit(1)
    if args.outer_threshold >= args.core_threshold:
        print("Error: Outer threshold must be less than core threshold")
        sys.exit(1)
    
    print(f"Core threshold: {args.core_threshold}th percentile")
    print(f"Outer threshold: {args.outer_threshold}th percentile")
    
    try:
        if args.image:
            # Process single image
            print(f"Processing single image: {args.image}")
            result = identify_core_regions(args.image, args.core_threshold, args.outer_threshold)
            
            if result:
                # Display detailed results for single image
                print(f"\n=== Region Analysis ===")
                print(f"Core region:")
                print(f"  Area: {result['core_area']} pixels ({result['core_area']/result['total_pixels']:.3%} of image)")
                print(f"  Summed intensity: {result['core_sum_intensity']:,}")
                print(f"  Intensity fraction: {result['core_sum_intensity']/result['total_image_sum_intensity']:.3%}")
                
                print(f"\nOuter region:")
                print(f"  Area: {result['outer_area']} pixels ({result['outer_area']/result['total_pixels']:.3%} of image)")
                print(f"  Summed intensity: {result['outer_sum_intensity']:,}")
                print(f"  Intensity fraction: {result['outer_sum_intensity']/result['total_image_sum_intensity']:.3%}")
                
                print(f"\nTotal image summed intensity: {result['total_image_sum_intensity']:,}")
                
                # Show visualization
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(result['normalized_image'], cmap='gray')
                plt.title("Original Image")
                plt.axis('off')
                
                # Identified regions (filled)
                plt.subplot(1, 3, 2)
                overlay = np.zeros((result['normalized_image'].shape[0], result['normalized_image'].shape[1], 3))
                overlay[result['outer_mask']] = [0, 0.6, 0]  # Dark green for outer
                overlay[result['core_mask']] = [1, 0, 0]     # Red for core
                plt.imshow(overlay)
                plt.title("Identified Regions\n(Red: Core, Green: Outer)")
                plt.axis('off')
                
                # Contour overlay
                plt.subplot(1, 3, 3)
                overlay_contour = np.stack([result['normalized_image']]*3, axis=-1)
                core_contours = measure.find_contours(result['core_mask'].astype(float), 0.5)
                outer_contours = measure.find_contours(result['outer_mask'].astype(float), 0.5)
                
                def draw_thick_contour(overlay, contours, color, thickness=3):
                    for contour in contours:
                        contour_coords = contour.astype(int)
                        for i in range(len(contour_coords)):
                            y, x = contour_coords[i]
                            for dy in range(-thickness, thickness+1):
                                for dx in range(-thickness, thickness+1):
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < overlay.shape[0] and 0 <= nx < overlay.shape[1]:
                                        overlay[ny, nx] = color
                
                # Draw contours
                draw_thick_contour(overlay_contour, outer_contours, [0, 255, 0], thickness=3)  # Green outer (first)
                draw_thick_contour(overlay_contour, core_contours, [255, 0, 0], thickness=3)   # Red core (second, on top)
                
                plt.imshow(overlay_contour)
                plt.title("Contour Overlay\n(Red: Core, Green: Outer)")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
            else:
                print("Failed to identify regions")
                
        elif args.folder:
            # Process folder
            print(f"Processing folder: {args.folder}")
            process_folder(args.folder, args.core_threshold, args.outer_threshold)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()