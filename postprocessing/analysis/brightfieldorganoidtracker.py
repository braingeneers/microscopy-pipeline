import cv2
import numpy as np
import sys
import os
from pathlib import Path
import argparse
from PIL import Image
import re

def find_organoid_contour(image, min_area=1000, gaussian_blur=5, threshold_method='otsu', 
                         threshold_value=127, morphology_kernel=5, morphology_iterations=2,
                         clahe_clip_limit=2.0, clahe_tile_grid_size=(8,8), otsu_adjustment=10):
    """
    Find the main organoid contour in a grayscale image using largest connected component.
    
    Args:
        image: Grayscale image as numpy array
        min_area: Minimum contour area to consider
        gaussian_blur: Kernel size for Gaussian blur (0 to disable)
        threshold_method: 'otsu', 'adaptive', or 'manual'
        threshold_value: Threshold value for manual thresholding
        morphology_kernel: Kernel size for morphological operations
        morphology_iterations: Number of iterations for morphological operations
        clahe_clip_limit: CLAHE clip limit for contrast enhancement
        clahe_tile_grid_size: CLAHE tile grid size
        otsu_adjustment: Amount to reduce Otsu threshold for more permissive detection
    
    Returns:
        tuple: (largest_contour, contour_area, binary_image)
    """
    
    # Preprocessing
    processed = image.copy()
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    processed = clahe.apply(processed)
    
    # Apply Gaussian blur to reduce noise
    if gaussian_blur > 0:
        processed = cv2.GaussianBlur(processed, (gaussian_blur, gaussian_blur), 0)
    
    # Thresholding - INVERTED for dark organoids on bright background
    if threshold_method == 'otsu':
        # Otsu's automatic thresholding - INVERTED and more permissive
        otsu_thresh, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Make threshold more permissive by reducing it by the specified amount
        permissive_thresh = max(0, otsu_thresh - otsu_adjustment)
        _, binary = cv2.threshold(processed, permissive_thresh, 255, cv2.THRESH_BINARY_INV)
        print(f"    Otsu threshold: {otsu_thresh}, Using permissive: {permissive_thresh} (adjustment: -{otsu_adjustment})")
    elif threshold_method == 'adaptive':
        # Adaptive thresholding - INVERTED
        binary = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    else:  # manual
        # Manual thresholding - INVERTED
        _, binary = cv2.threshold(processed, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up the binary image
    if morphology_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (morphology_kernel, morphology_kernel))
        # Opening (erosion followed by dilation) to remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 
                                iterations=morphology_iterations)
        # Closing (dilation followed by erosion) to fill gaps
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, 
                                iterations=morphology_iterations)
    
    # Find connected components to get the largest one
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter out background (label 0) and find the largest component
    if num_labels <= 1:  # Only background found
        return None, 0, binary
    
    # Get areas of all components (excluding background at index 0)
    component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
    component_indices = np.arange(1, num_labels)  # Component labels (1 to num_labels-1)
    
    # Filter by minimum area
    valid_mask = component_areas >= min_area
    if not np.any(valid_mask):
        return None, 0, binary
    
    valid_areas = component_areas[valid_mask]
    valid_indices = component_indices[valid_mask]
    
    # Find the largest valid component
    largest_idx = valid_indices[np.argmax(valid_areas)]
    largest_area = np.max(valid_areas)
    
    # Create a binary mask containing only the largest component
    largest_component_mask = (labels == largest_idx).astype(np.uint8) * 255
    
    # Find contour of the largest component
    contours, _ = cv2.findContours(largest_component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0, binary
    
    # Should only be one contour, but take the largest just in case
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    
    # Update binary image to show only the largest component
    binary_final = largest_component_mask
    
    return largest_contour, contour_area, binary_final

def process_single_image(input_path, output_path, contour_color=(0, 255, 0), 
                        contour_thickness=2, **detection_params):
    """
    Process a single TIFF image to detect organoid contour.
    
    Args:
        input_path: Path to input TIFF file
        output_path: Path to save output image with contour
        contour_color: RGB color for contour outline
        contour_thickness: Thickness of contour line
        **detection_params: Parameters for contour detection
    
    Returns:
        dict: Results including area, contour info, etc.
    """
    
    # Load image
    try:
        # Try loading with PIL first to handle different TIFF formats
        with Image.open(input_path) as img:
            original_mode = img.mode
            print(f"Original image mode: {original_mode}")
            
            # Convert to grayscale if needed
            # if img.mode != 'L':
            #    img = img.convert('L')
            
            # Convert to numpy array
            image = np.array(img)
            
            # Store the original unprocessed image for darkness calculations
            original_image = image.copy()
            
            # Check if image is 16-bit and convert to 8-bit
            if image.dtype == np.uint16 or original_mode in ['I;16', 'I;16B', 'I;16L', 'I']:
                print(f"16-bit image detected, converting to 8-bit")
                print(f"16-bit range: {np.min(image)} - {np.max(image)}")
                
                # Simple division method (faster, preserves relative intensities)
                image_8bit = (image // 256).astype(np.uint8)
                
                image = image_8bit
                print(f"8-bit range after conversion: {np.min(image)} - {np.max(image)}")
            else:
                print(f"8-bit image, no conversion needed")
            
            # Scale image brightness to full 0-255 range
            img_min = np.min(image)
            img_max = np.max(image)
            print(f"Original brightness range: {img_min} - {img_max}")
            
            if img_min == img_max:
                # All pixels have the same value - create uniform gray image
                print("Warning: Image has uniform pixel values, setting to middle gray")
                image = np.full_like(image, 128, dtype=np.uint8)
            else:
                # Scale to full 0-255 range
                image_float = image.astype(np.float32)
                image_normalized = (image_float - img_min) / (img_max - img_min)
                image = (image_normalized * 255).astype(np.uint8)
                
            print(f"Scaled brightness range: {np.min(image)} - {np.max(image)}")
        
    except Exception as e:
        raise ValueError(f"Failed to load image '{input_path}': {e}")
    
    # Find organoid contour
    contour, area, binary_image = find_organoid_contour(image, **detection_params)
    
    if contour is None:
        print(f"Warning: No contour found meeting criteria")
        return {
            'area_pixels': 0,
            'area_um2': 0,
            'contour_found': False,
            'perimeter': 0,
            'centroid': (0, 0),
            'bounding_box': (0, 0, 0, 0),
            'width': 0,
            'height': 0,
            'aspect_ratio': 0,
            'circularity': 0,
            'total_darkness': 0,
            'average_darkness': 0,
            'min_darkness': 0,
            'max_darkness': 0
        }
    
    # Calculate additional metrics
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M['m00'] != 0:
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])
        centroid = (centroid_x, centroid_y)
    else:
        centroid = (0, 0)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    bounding_box = (x, y, w, h)
    
    # Calculate darkness metrics from original unprocessed image
    # Create mask for the contour area
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    # Extract pixel values within the contour from original image
    enclosed_pixels = original_image[mask > 0]
    
    if len(enclosed_pixels) > 0:
        # Calculate darkness metrics (lower values = darker pixels)
        total_darkness = np.sum(255 - enclosed_pixels)  # Invert so higher = darker
        average_darkness = np.mean(255 - enclosed_pixels)
        min_darkness = np.min(255 - enclosed_pixels)  # Darkest pixel (highest darkness value)
        max_darkness = np.max(255 - enclosed_pixels)  # Lightest pixel (lowest darkness value)
        
        # Also calculate raw intensity statistics for reference
        raw_intensity_mean = np.mean(enclosed_pixels)
        raw_intensity_min = np.min(enclosed_pixels)
        raw_intensity_max = np.max(enclosed_pixels)
        
        print(f"    Darkness analysis: avg={average_darkness:.1f}, total={total_darkness:.0f}")
        print(f"    Raw intensity: mean={raw_intensity_mean:.1f}, range={raw_intensity_min}-{raw_intensity_max}")
    else:
        total_darkness = 0
        average_darkness = 0
        min_darkness = 0
        max_darkness = 0
        raw_intensity_mean = 0
        raw_intensity_min = 0
        raw_intensity_max = 0
    
    # Create output image with contour overlay
    if len(image.shape) == 2:
        # Convert grayscale to RGB for colored contour
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        output_image = image.copy()
    
    # Draw contour
    cv2.drawContours(output_image, [contour], -1, contour_color, contour_thickness)
    
    # Draw centroid
    cv2.circle(output_image, centroid, 5, (255, 0, 0), -1)  # Red dot for centroid
    
    # Draw bounding box
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 0), 1)  # Yellow box
    
    # Save output image
    output_pil = Image.fromarray(output_image)
    output_pil.save(output_path)
    
    # Prepare results
    results = {
        'area_pixels': float(area),
        'perimeter': float(perimeter),
        'contour_found': True,
        'centroid': centroid,
        'bounding_box': bounding_box,
        'width': w,
        'height': h,
        'aspect_ratio': w / h if h > 0 else 0,
        'circularity': (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0,
        'total_darkness': float(total_darkness),
        'average_darkness': float(average_darkness),
        'min_darkness': float(min_darkness),
        'max_darkness': float(max_darkness),
        'raw_intensity_mean': float(raw_intensity_mean),
        'raw_intensity_min': float(raw_intensity_min),
        'raw_intensity_max': float(raw_intensity_max)
    }
    
    return results

def extract_timepoint_from_filename(filename):
    """
    Extract timepoint number from filename.
    Supports various formats like: image_001.tif, timepoint_5.tiff, t10.tif, etc.
    
    Args:
        filename: Image filename
    
    Returns:
        int: Timepoint number, or 0 if no number found
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Look for various patterns of numbers in filename
    patterns = [
        r'_(\d+)$',           # ending with _123
        r't(\d+)',            # t123 or T123
        r'timepoint[_\s]*(\d+)', # timepoint_123 or timepoint 123
        r'tp(\d+)',           # tp123
        r'(\d+)_',            # 123_something
        r'(\d+)',             # any number in the filename
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # If no number found, return 0
    return 0

def batch_process_images(input_dir, output_dir, pixel_size_um=None, 
                        contour_color=(0, 255, 0), contour_thickness=2, **detection_params):
    """
    Batch process all TIFF images in a directory.
    
    Args:
        input_dir: Directory containing input TIFF files
        output_dir: Directory for output images and CSV
        pixel_size_um: Pixel size in micrometers (for area conversion)
        contour_color: RGB color for contour outline
        contour_thickness: Thickness of contour line
        **detection_params: Parameters for contour detection
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input directory
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find TIFF files
    tiff_extensions = {'.tif', '.tiff'}
    image_files = []
    for ext in tiff_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # Sort files by timepoint number extracted from filename
    def sort_key(file_path):
        timepoint = extract_timepoint_from_filename(file_path.name)
        return (timepoint, file_path.name)  # Secondary sort by filename for ties
    
    image_files = sorted(image_files, key=sort_key)
    
    if not image_files:
        print(f"No TIFF files found in '{input_dir}'")
        return
    
    print(f"Found {len(image_files)} TIFF files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if pixel_size_um:
        print(f"Pixel size: {pixel_size_um} μm/pixel")
    
    # Print CLAHE settings
    print(f"CLAHE settings: clip_limit={detection_params.get('clahe_clip_limit', 2.0)}, "
          f"tile_grid={detection_params.get('clahe_tile_grid_size', (8,8))}")
    print("Note: 16-bit images will be automatically converted to 8-bit")
    print("Files will be processed in timepoint order")
    
    # Show first few filenames and their extracted timepoints for verification
    print("\nTimepoint order (first 5 files):")
    for i, img_file in enumerate(image_files[:5]):
        timepoint = extract_timepoint_from_filename(img_file.name)
        print(f"  {img_file.name} -> timepoint {timepoint}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more files")
    
    print("-" * 60)
    
    # Process images and collect results
    all_results = []
    processed = 0
    failed = 0
    bit_depth_stats = {'8-bit': 0, '16-bit': 0, 'other': 0}
    
    for i, img_file in enumerate(image_files, 1):
        output_file = output_path / f"{img_file.stem}_contour{img_file.suffix}"
        timepoint = extract_timepoint_from_filename(img_file.name)
        
        print(f"[{i:3d}/{len(image_files)}] Processing: {img_file.name} (timepoint {timepoint})")
        
        try:
            # Check bit depth for statistics
            with Image.open(img_file) as img_check:
                if img_check.mode in ['I;16', 'I;16B', 'I;16L', 'I']:
                    bit_depth_stats['16-bit'] += 1
                elif img_check.mode == 'L':
                    bit_depth_stats['8-bit'] += 1
                else:
                    bit_depth_stats['other'] += 1
            
            results = process_single_image(
                str(img_file), str(output_file), 
                contour_color, contour_thickness, **detection_params
            )
            
            # Add filename, timepoint, and calculated areas
            results['filename'] = img_file.name
            results['timepoint'] = timepoint
            results['output_filename'] = output_file.name
            
            if pixel_size_um and results['area_pixels'] > 0:
                results['area_um2'] = results['area_pixels'] * (pixel_size_um ** 2)
                results['perimeter_um'] = results['perimeter'] * pixel_size_um
            else:
                results['area_um2'] = 0
                results['perimeter_um'] = 0
            
            all_results.append(results)
            
            if results['contour_found']:
                print(f"    ✓ Contour found: {results['area_pixels']:.0f} pixels")
                if pixel_size_um:
                    print(f"    ✓ Area: {results['area_um2']:.2f} μm²")
                print(f"    ✓ Centroid: {results['centroid']}")
                print(f"    ✓ Saved: {output_file.name}")
            else:
                print(f"    ⚠ No contour found")
            
            processed += 1
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1
        
        print()
    
    # Save results to CSV (already in timepoint order)
    if all_results:
        import csv
        csv_path = output_path / "organoid_analysis_results.csv"
        
        fieldnames = ['timepoint', 'filename', 'output_filename', 'contour_found', 'area_pixels', 'area_um2', 
                     'perimeter', 'perimeter_um', 'centroid', 'bounding_box', 'width', 'height', 
                     'aspect_ratio', 'circularity', 'total_darkness', 'average_darkness', 'min_darkness', 
                     'max_darkness', 'raw_intensity_mean', 'raw_intensity_min', 'raw_intensity_max']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
        
        print(f"Results saved to: {csv_path}")
        
        # Also create a summary CSV with just timepoint and key metrics
        summary_csv_path = output_path / "organoid_areas_summary.csv"
        with open(summary_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timepoint', 'filename', 'area_pixels', 'area_um2', 'total_darkness', 
                           'average_darkness', 'raw_intensity_mean', 'contour_found'])
            for result in all_results:
                writer.writerow([
                    result['timepoint'], 
                    result['filename'], 
                    result['area_pixels'], 
                    result['area_um2'],
                    result['total_darkness'],
                    result['average_darkness'],
                    result['raw_intensity_mean'],
                    result['contour_found']
                ])
        
        print(f"Area summary saved to: {summary_csv_path}")
    
    # Summary
    print("-" * 60)
    print(f"Batch processing complete:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Total files: {len(image_files)}")
    print(f"  Success rate: {(processed/len(image_files)*100):.1f}%")
    
    # Bit depth statistics
    print(f"\nBit depth statistics:")
    print(f"  8-bit images: {bit_depth_stats['8-bit']}")
    print(f"  16-bit images: {bit_depth_stats['16-bit']} (converted to 8-bit)")
    print(f"  Other formats: {bit_depth_stats['other']} (converted to 8-bit)")
    
    # Timepoint range
    if all_results:
        timepoints = [r['timepoint'] for r in all_results]
        print(f"\nTimepoint range:")
        print(f"  First timepoint: {min(timepoints)}")
        print(f"  Last timepoint: {max(timepoints)}")
        print(f"  Total timepoints: {len(set(timepoints))}")
    
    # Statistics for successfully processed images
    successful_results = [r for r in all_results if r['contour_found']]
    if successful_results:
        areas = [r['area_pixels'] for r in successful_results]
        darkness_values = [r['average_darkness'] for r in successful_results]
        intensity_values = [r['raw_intensity_mean'] for r in successful_results]
        
        print(f"\nArea statistics (pixels):")
        print(f"  Mean: {np.mean(areas):.1f}")
        print(f"  Median: {np.median(areas):.1f}")
        print(f"  Min: {np.min(areas):.1f}")
        print(f"  Max: {np.max(areas):.1f}")
        
        print(f"\nDarkness statistics (higher = darker):")
        print(f"  Mean: {np.mean(darkness_values):.1f}")
        print(f"  Median: {np.median(darkness_values):.1f}")
        print(f"  Min: {np.min(darkness_values):.1f}")
        print(f"  Max: {np.max(darkness_values):.1f}")
        
        print(f"\nRaw intensity statistics (higher = brighter):")
        print(f"  Mean: {np.mean(intensity_values):.1f}")
        print(f"  Median: {np.median(intensity_values):.1f}")
        print(f"  Min: {np.min(intensity_values):.1f}")
        print(f"  Max: {np.max(intensity_values):.1f}")
        
        if pixel_size_um:
            areas_um2 = [r['area_um2'] for r in successful_results]
            print(f"\nArea statistics (μm²):")
            print(f"  Mean: {np.mean(areas_um2):.1f}")
            print(f"  Median: {np.median(areas_um2):.1f}")
            print(f"  Min: {np.min(areas_um2):.1f}")
            print(f"  Max: {np.max(areas_um2):.1f}")

def main():
    parser = argparse.ArgumentParser(
        description="Detect organoid contours in grayscale TIFF images with CLAHE enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image (automatically handles 16-bit conversion)
  python brightfieldorganoidtracker.py input.tif output.tif

  # Batch processing
  python brightfieldorganoidtracker.py ./images ./results --batch

  # With pixel size for area calculation
  python brightfieldorganoidtracker.py input.tif output.tif --pixel-size 0.5

  # Custom detection parameters with CLAHE
  python brightfieldorganoidtracker.py input.tif output.tif --min-area 2000 --clahe-clip 3.0

  # More permissive Otsu thresholding
  python brightfieldorganoidtracker.py input.tif output.tif --otsu-adjustment 15

Output:
  - Images with contour overlays (green contour, red centroid, yellow bounding box)
  - CSV file with analysis results (batch mode only)
  - Area measurements in pixels and μm² (if pixel size provided)
  - Automatic 16-bit to 8-bit conversion for better processing

Bit Depth Handling:
  - 16-bit images are automatically converted to 8-bit by dividing by 256
  - This preserves relative intensities while enabling proper thresholding
  - Original bit depth is reported in processing output

Detection Parameters:
  --min-area: Minimum contour area to consider (default: 10000)
  --threshold: Thresholding method (otsu, adaptive, manual)
  --threshold-value: Manual threshold value (0-255, used with manual method)
  --otsu-adjustment: Amount to reduce Otsu threshold for more permissive detection (default: 10)
  --blur: Gaussian blur kernel size (0 to disable, default: 5)
  --morphology-kernel: Morphological operation kernel size (default: 5)
  --morphology-iterations: Number of morphological iterations (default: 2)

CLAHE Parameters:
  --clahe-clip: CLAHE clip limit for contrast enhancement (default: 2.0)
  --clahe-grid: CLAHE tile grid size (default: 8)
        """)
    
    parser.add_argument('input', help='Input TIFF file or directory')
    parser.add_argument('output', help='Output TIFF file or directory')
    parser.add_argument('--batch', action='store_true', 
                       help='Batch process all TIFF files in input directory')
    parser.add_argument('--pixel-size', type=float, 
                       help='Pixel size in micrometers for area calculation')
    
    # Detection parameters
    parser.add_argument('--min-area', type=int, default=10000,
                       help='Minimum contour area in pixels (default: 10000)')
    parser.add_argument('--threshold', choices=['otsu', 'adaptive', 'manual'], 
                       default='otsu', help='Thresholding method (default: otsu)')
    parser.add_argument('--threshold-value', type=int, default=127,
                       help='Manual threshold value 0-255 (default: 127)')
    parser.add_argument('--otsu-adjustment', type=int, default=10,
                       help='Amount to reduce Otsu threshold for more permissive detection (default: 10)')
    parser.add_argument('--blur', type=int, default=5,
                       help='Gaussian blur kernel size, 0 to disable (default: 5)')
    parser.add_argument('--morphology-kernel', type=int, default=5,
                       help='Morphological operation kernel size (default: 5)')
    parser.add_argument('--morphology-iterations', type=int, default=2,
                       help='Morphological operation iterations (default: 2)')
    
    # CLAHE parameters
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                       help='CLAHE clip limit for contrast enhancement (default: 2.0)')
    parser.add_argument('--clahe-grid', type=int, default=8,
                       help='CLAHE tile grid size (creates NxN grid, default: 8)')
    
    # Visualization parameters
    parser.add_argument('--contour-color', type=str, default='0,255,0',
                       help='Contour color as R,G,B (default: 0,255,0 for green)')
    parser.add_argument('--contour-thickness', type=int, default=2,
                       help='Contour line thickness (default: 2)')
    
    args = parser.parse_args()
    
    # Parse contour color
    try:
        color_values = [int(x.strip()) for x in args.contour_color.split(',')]
        if len(color_values) != 3 or any(not (0 <= x <= 255) for x in color_values):
            raise ValueError("Color values must be 0-255")
        contour_color = tuple(color_values)
    except ValueError:
        print("Error: Invalid contour color. Use format 'R,G,B' with values 0-255")
        sys.exit(1)
    
    # Validate parameters
    if args.min_area < 1:
        print("Error: Minimum area must be positive")
        sys.exit(1)
    
    if not (0 <= args.threshold_value <= 255):
        print("Error: Threshold value must be 0-255")
        sys.exit(1)
    
    if args.blur < 0:
        print("Error: Blur kernel size must be non-negative")
        sys.exit(1)
    
    if args.clahe_clip <= 0:
        print("Error: CLAHE clip limit must be positive")
        sys.exit(1)
    
    if args.clahe_grid < 1:
        print("Error: CLAHE grid size must be positive")
        sys.exit(1)
    
    # Replace with validation for reasonable range:
    if not (-50 <= args.otsu_adjustment <= 50):
        print("Error: Otsu adjustment must be between -50 and 50")
        sys.exit(1)
    
    # Prepare detection parameters
    detection_params = {
        'min_area': args.min_area,
        'gaussian_blur': args.blur,
        'threshold_method': args.threshold,
        'threshold_value': args.threshold_value,
        'morphology_kernel': args.morphology_kernel,
        'morphology_iterations': args.morphology_iterations,
        'clahe_clip_limit': args.clahe_clip,
        'clahe_tile_grid_size': (args.clahe_grid, args.clahe_grid),
        'otsu_adjustment': args.otsu_adjustment
    }
    
    try:
        if args.batch:
            # Batch processing
            batch_process_images(
                args.input, args.output, args.pixel_size,
                contour_color, args.contour_thickness, **detection_params
            )
        else:
            # Single file processing
            print(f"Processing single image: {args.input}")
            print(f"CLAHE settings: clip_limit={args.clahe_clip}, tile_grid={args.clahe_grid}x{args.clahe_grid}")
            print(f"Otsu adjustment: -{args.otsu_adjustment}")
            
            results = process_single_image(
                args.input, args.output, 
                contour_color, args.contour_thickness, **detection_params
            )
            
            if results['contour_found']:
                print(f"✓ Organoid contour detected")
                print(f"  Area: {results['area_pixels']:.0f} pixels")
                if args.pixel_size:
                    area_um2 = results['area_pixels'] * (args.pixel_size ** 2)
                    print(f"  Area: {area_um2:.2f} μm²")
                print(f"  Perimeter: {results['perimeter']:.1f} pixels")
                print(f"  Centroid: {results['centroid']}")
                print(f"  Bounding box: {results['bounding_box']}")
                print(f"  Aspect ratio: {results['aspect_ratio']:.2f}")
                print(f"  Circularity: {results['circularity']:.3f}")
                print(f"✓ Output saved to: {args.output}")
            else:
                print("⚠ No organoid contour found")
                print("Try adjusting detection parameters:")
                print("  - Lower --min-area for smaller organoids")
                print("  - Change --threshold method (otsu/adaptive/manual)")
                print("  - Adjust --otsu-adjustment for more/less permissive Otsu thresholding")
                print("  - Adjust --clahe-clip for contrast enhancement")
                print("  - Adjust --blur for noise reduction")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()