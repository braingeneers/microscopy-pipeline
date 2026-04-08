import cv2
import sys
import os
from pathlib import Path
import numpy as np

def apply_clahe_single(input_path, output_path, clip_limit=2.0, tile_grid_size=(8,8), channel_only=None):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a single image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: (8,8))
        channel_only: If specified, apply CLAHE only to that channel ('red', 'green', 'blue', or None for grayscale)
    """
    # Read image
    if channel_only is not None:
        # Read as color image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error: Unable to read image '{input_path}'")
            return False
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Channel mapping
        channel_map = {'red': 0, 'green': 1, 'blue': 2}
        channel_index = channel_map[channel_only]
        
        # Extract specified channel
        target_channel = img_rgb[:, :, channel_index]
        
        # Create CLAHE object and apply to target channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_channel = clahe.apply(target_channel)
        
        # Replace target channel with CLAHE-processed version
        img_rgb_clahe = img_rgb.copy()
        img_rgb_clahe[:, :, channel_index] = enhanced_channel
        
        # Convert back to BGR for saving
        img_output = cv2.cvtColor(img_rgb_clahe, cv2.COLOR_RGB2BGR)
        
    else:
        # Read image in grayscale (original behavior)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to read image '{input_path}'")
            return False
        
        # Create CLAHE object and apply to entire image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_output = clahe.apply(img)

    # Save result
    success = cv2.imwrite(output_path, img_output)
    if not success:
        print(f"Error: Failed to save image to '{output_path}'")
        return False
    
    return True

def batch_process_folder(input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8,8), channel_only=None):
    """
    Apply CLAHE to all images in a folder.
    
    Args:
        input_dir: Path to input directory containing images
        output_dir: Path to output directory for processed images
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        channel_only: If specified, apply CLAHE only to that channel
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.jp2'}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No supported image files found in '{input_dir}'")
        print(f"Supported extensions: {', '.join(sorted(image_extensions))}")
        return
    
    # Process images
    print(f"Found {len(image_files)} image files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    if channel_only:
        print(f"Processing mode: {channel_only} channel only")
    else:
        print(f"Processing mode: grayscale")
    
    print(f"CLAHE parameters: grid={tile_grid_size}, clip_limit={clip_limit}")
    print("-" * 60)
    
    processed = 0
    failed = 0
    
    for i, img_file in enumerate(image_files, 1):
        # Create output filename (preserve extension)
        output_file = output_path / img_file.name
        
        print(f"[{i:3d}/{len(image_files)}] Processing: {img_file.name}")
        
        try:
            # Get image info for reporting
            if channel_only:
                img_temp = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
                if img_temp is not None:
                    print(f"    RGB image: {img_temp.shape}")
            else:
                img_temp = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img_temp is not None:
                    print(f"    Grayscale image: {img_temp.shape}")
            
            # Apply CLAHE
            success = apply_clahe_single(
                str(img_file), 
                str(output_file), 
                clip_limit, 
                tile_grid_size, 
                channel_only
            )
            
            if success:
                print(f"    ✓ Saved to: {output_file.name}")
                processed += 1
            else:
                print(f"    ✗ Failed to process")
                failed += 1
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1
        
        print()  # Empty line between files
    
    # Summary
    print("-" * 60)
    print(f"Batch processing complete:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Total files: {len(image_files)}")
    print(f"  Success rate: {(processed/len(image_files)*100):.1f}%")
    print(f"  Output directory: {output_dir}")

def apply_clahe(input_path, output_path, clip_limit=2.0, tile_grid_size=(8,8), channel_only=None):
    """
    Apply CLAHE to a single image or batch process a folder.
    Automatically detects if input is a file or directory.
    """
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # Single file processing
        print(f"Processing single image: {input_path}")
        if channel_only:
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is not None:
                print(f"RGB image: {img.shape}")
        else:
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print(f"Grayscale image: {img.shape}")
        
        success = apply_clahe_single(input_path, output_path, clip_limit, tile_grid_size, channel_only)
        
        if success:
            if channel_only:
                print(f"CLAHE applied to {channel_only} channel only with grid size {tile_grid_size}")
            else:
                print(f"CLAHE applied to grayscale image with grid size {tile_grid_size}")
            print(f"Result saved to '{output_path}'")
        
    elif input_path_obj.is_dir():
        # Batch processing
        print(f"Batch processing directory: {input_path}")
        batch_process_folder(input_path, output_path, clip_limit, tile_grid_size, channel_only)
        
    else:
        print(f"Error: '{input_path}' is neither a file nor a directory")

def print_usage():
    print("Usage: python clahe.py <input> <output> [options]")
    print()
    print("Arguments:")
    print("  input            Input image file OR directory containing images")
    print("  output           Output image file OR directory for batch processing")
    print()
    print("Options:")
    print("  --grid SIZE      Grid size for CLAHE tiles (default: 8)")
    print("                   Creates SIZE x SIZE grid")
    print("  --clip LIMIT     Clip limit for contrast limiting (default: 2.0)")
    print("  --red-channel    Apply CLAHE only to red channel of RGB image")
    print("  --green-channel  Apply CLAHE only to green channel of RGB image")
    print("  --blue-channel   Apply CLAHE only to blue channel of RGB image")
    print("                   (default: process as grayscale)")
    print()
    print("Single Image Examples:")
    print("  # Basic grayscale CLAHE")
    print("  python clahe.py input.jpg output.jpg")
    print()
    print("  # Apply CLAHE to red channel only")
    print("  python clahe.py input.jpg output.jpg --red-channel")
    print()
    print("  # Custom parameters")
    print("  python clahe.py input.jpg output.jpg --green-channel --grid 16 --clip 3.0")
    print()
    print("Batch Processing Examples:")
    print("  # Process all images in folder (grayscale)")
    print("  python clahe.py ./input_folder ./output_folder")
    print()
    print("  # Batch process red channel enhancement")
    print("  python clahe.py ./microscopy_images ./enhanced_red --red-channel")
    print()
    print("  # Batch process with custom parameters")
    print("  python clahe.py ./raw_images ./processed --green-channel --grid 12 --clip 2.5")
    print()
    print("Supported formats: JPG, JPEG, PNG, TIFF, TIF, BMP, WEBP, JP2")
    print()
    print("Channel-specific applications:")
    print("  Red channel:   RFP, mCherry, tdTomato fluorescent proteins")
    print("  Green channel: GFP, EGFP, FITC fluorescent markers")
    print("  Blue channel:  DAPI, Hoechst nuclear stains, CFP proteins")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Default parameters
    grid_size = 8
    clip_limit = 2.0
    channel_only = None
    
    # Parse command line arguments
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--grid' and i + 1 < len(sys.argv):
            try:
                grid_size = int(sys.argv[i + 1])
                if grid_size <= 0:
                    print("Error: Grid size must be positive")
                    sys.exit(1)
                i += 2
            except ValueError:
                print(f"Error: Grid size '{sys.argv[i + 1]}' is not a valid integer")
                sys.exit(1)
                
        elif arg == '--clip' and i + 1 < len(sys.argv):
            try:
                clip_limit = float(sys.argv[i + 1])
                if clip_limit <= 0:
                    print("Error: Clip limit must be positive")
                    sys.exit(1)
                i += 2
            except ValueError:
                print(f"Error: Clip limit '{sys.argv[i + 1]}' is not a valid number")
                sys.exit(1)
                
        elif arg == '--red-channel':
            if channel_only is not None:
                print("Error: Only one channel option can be specified")
                sys.exit(1)
            channel_only = 'red'
            i += 1
            
        elif arg == '--green-channel':
            if channel_only is not None:
                print("Error: Only one channel option can be specified")
                sys.exit(1)
            channel_only = 'green'
            i += 1
            
        elif arg == '--blue-channel':
            if channel_only is not None:
                print("Error: Only one channel option can be specified")
                sys.exit(1)
            channel_only = 'blue'
            i += 1
            
        else:
            print(f"Error: Unknown argument '{arg}'")
            print_usage()
            sys.exit(1)
    
    # Apply CLAHE with specified parameters
    tile_grid_size = (grid_size, grid_size)
    apply_clahe(input_path, output_path, clip_limit, tile_grid_size, channel_only)