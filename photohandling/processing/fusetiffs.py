import sys
import argparse
import re
import glob
import numpy as np
from PIL import Image
from pathlib import Path

def parse_stack_filename(filename):
    """
    Parse filename in format stack_X.tiff and return X as integer.
    Returns None if filename doesn't match the expected format.
    """
    match = re.match(r'stack_(\d+)\.tiff?$', filename, re.IGNORECASE)
    if match:
        x = int(match.group(1))
        return x
    return None

def get_stack_files_sorted(folder_path):
    """
    Get stack_X.tiff files from folder, sorted by X value.
    Returns list of (X, filepath) tuples sorted by X.
    """
    tiff_files = []
    
    # Look for both .tiff and .tif extensions
    for ext in ['*.tiff', '*.tif']:
        tiff_files.extend(glob.glob(str(Path(folder_path) / ext)))
    
    stack_files = []
    
    for tiff_file in tiff_files:
        filename = Path(tiff_file).name
        x_value = parse_stack_filename(filename)
        
        if x_value is not None:
            stack_files.append((x_value, tiff_file))
        else:
            print(f"Warning: Skipping file '{filename}' - doesn't match stack_X.tiff format")
    
    # Sort by X value
    stack_files.sort(key=lambda item: item[0])
    
    return stack_files

def process_single_file(input_path, output_path=None, brightfield_mode=False):
    """
    Process a single TIFF stack file, supporting both 8-bit and 16-bit images.
    """
    if not Path(input_path).exists():
        print(f"Error: Input file '{input_path}' does not exist")
        return False
    
    # Create default output filename with _superimposed suffix
    if output_path is None:
        input_file = Path(input_path)
        suffix = "_brightfield_superimposed" if brightfield_mode else "_superimposed"
        output_path = input_file.parent / f"{input_file.stem}{suffix}{input_file.suffix}"
    
    print(f"Processing: {Path(input_path).name}")
    if brightfield_mode:
        print("Brightfield mode: Scaling to maximum dynamic range")
    
    # Read the tiff stack using PIL
    stack = []
    is_16bit = False
    original_mode = None
    
    with Image.open(input_path) as img:
        print(f"Input image mode: {img.mode}")
        original_mode = img.mode
        
        # Detect if input is 16-bit - explicitly check for I;16 modes
        is_16bit = img.mode in ['I;16', 'I;16B', 'I;16L', 'I']
        
        if is_16bit:
            print("Detected 16-bit input image")
            max_value = 65535  # Explicitly set to 65535 for all 16-bit modes including I;16
            dtype_processing = np.uint64  # Use uint64 for accumulation to prevent overflow
            dtype_output = np.uint16
        else:
            print("Detected 8-bit input image")
            max_value = 255
            dtype_processing = np.uint32  # Use uint32 for accumulation
            dtype_output = np.uint8
        
        try:
            while True:
                # Handle different input modes
                if img.mode in ['I;16', 'I;16B', 'I;16L']:
                    # 16-bit grayscale - keep as is, max value is 65535
                    frame = np.array(img, dtype=np.uint16)
                    # Ensure we're using 16-bit parameters
                    is_16bit = True
                    max_value = 65535
                    dtype_processing = np.uint64
                    dtype_output = np.uint16
                elif img.mode == 'I':
                    # 32-bit mode, but likely 16-bit data
                    frame_array = np.array(img)
                    # Check if values are actually 16-bit range
                    if frame_array.max() <= 65535:
                        frame = frame_array.astype(np.uint16)
                        is_16bit = True
                        max_value = 65535
                        dtype_processing = np.uint64
                        dtype_output = np.uint16
                    else:
                        # True 32-bit data, scale down to 16-bit
                        frame = (frame_array / (frame_array.max() / 65535)).astype(np.uint16)
                        is_16bit = True
                        max_value = 65535
                        dtype_processing = np.uint64
                        dtype_output = np.uint16
                elif img.mode == 'L':
                    # 8-bit grayscale - keep as is
                    frame = np.array(img, dtype=np.uint8)
                else:
                    # Convert color to grayscale while preserving bit depth
                    if is_16bit:
                        # For 16-bit color images, convert to grayscale maintaining 16-bit depth
                        rgb_img = img.convert('RGB')
                        rgb_array = np.array(rgb_img, dtype=np.float32)
                        # Luminance formula for RGB to grayscale
                        gray_array = 0.299 * rgb_array[:,:,0] + 0.587 * rgb_array[:,:,1] + 0.114 * rgb_array[:,:,2]
                        # Scale to 16-bit with max value 65535
                        frame = (gray_array * (65535 / 255)).astype(np.uint16)
                        max_value = 65535
                        dtype_processing = np.uint64
                        dtype_output = np.uint16
                    else:
                        # For 8-bit color images
                        frame = np.array(img.convert('L'), dtype=np.uint8)
                
                stack.append(frame)
                img.seek(img.tell() + 1)
        except EOFError:
            pass  # End of sequence
    
    if len(stack) <= 1:
        print("Input is a single image, nothing to stack.")
        return False
    
    print(f"Loaded {len(stack)} images from stack")
    print(f"Processing in {'16-bit' if is_16bit else '8-bit'} mode")
    print(f"Max value: {max_value}")
    
    # Ensure all frames have the same dtype
    if is_16bit:
        stack_processed = [frame.astype(np.uint16) for frame in stack]
    else:
        stack_processed = [frame.astype(np.uint8) for frame in stack]
    
    # Sum all frames using appropriate accumulation dtype to prevent overflow
    summed = np.zeros_like(stack_processed[0], dtype=dtype_processing)
    for frame in stack_processed:
        summed += frame.astype(dtype_processing)
    
    # Divide by number of images to get average
    averaged = summed / len(stack)
    
    # Apply brightfield scaling if requested
    if brightfield_mode and is_16bit:
        # Find current min and max values in the averaged image
        current_min = np.min(averaged)
        current_max = np.max(averaged)
        
        print(f"Original value range: {current_min:.1f} - {current_max:.1f}")
        
        if current_max > current_min:
            # Scale to full 16-bit range (0-65535) - explicitly use 65535
            scaled = ((averaged - current_min) / (current_max - current_min)) * 65535
            averaged_final = np.clip(scaled, 0, 65535).astype(dtype_output)
            print(f"Scaled to full 16-bit range: 0 - 65535")
        else:
            # All pixels have the same value, no scaling needed
            averaged_final = np.clip(averaged, 0, 65535).astype(dtype_output)  # Use 65535 explicitly
            print("No scaling applied (uniform image)")
    
    elif brightfield_mode and not is_16bit:
        # For 8-bit images, scale to full 8-bit range
        current_min = np.min(averaged)
        current_max = np.max(averaged)
        
        print(f"Original value range: {current_min:.1f} - {current_max:.1f}")
        
        if current_max > current_min:
            # Scale to full 8-bit range (0-255)
            scaled = ((averaged - current_min) / (current_max - current_min)) * 255
            averaged_final = np.clip(scaled, 0, 255).astype(dtype_output)
            print(f"Scaled to full 8-bit range: 0 - 255")
        else:
            # All pixels have the same value, no scaling needed
            averaged_final = np.clip(averaged, 0, 255).astype(dtype_output)  # Use 255 explicitly
            print("No scaling applied (uniform image)")
    
    else:
        # Normal mode: Convert back to appropriate bit depth without scaling
        if is_16bit:
            averaged_final = np.clip(averaged, 0, 65535).astype(dtype_output)  # Use 65535 explicitly for 16-bit
        else:
            averaged_final = np.clip(averaged, 0, 255).astype(dtype_output)   # Use 255 explicitly for 8-bit
    
    # Save using PIL with appropriate mode
    if is_16bit:
        # For 16-bit output, always use I;16 mode to preserve full 16-bit range
        result_img = Image.fromarray(averaged_final, mode='I;16')
        print(f"Saving as I;16 mode with max value 65535")
    else:
        result_img = Image.fromarray(averaged_final, mode='L')
        print(f"Saving as L mode with max value 255")
    
    # Try to preserve TIFF metadata if available
    try:
        with Image.open(input_path) as original_img:
            # Copy some metadata from the original
            if hasattr(original_img, 'tag'):
                result_img.save(output_path, tiffinfo=original_img.tag)
            else:
                result_img.save(output_path)
    except Exception as e:
        print(f"Warning: Could not preserve metadata: {e}")
        result_img.save(output_path)
    
    mode_info = "brightfield " if brightfield_mode else ""
    print(f"Averaged {mode_info}{'16-bit' if is_16bit else '8-bit'} image saved to {output_path}")
    
    return True

def process_folder(folder_path, output_dir=None, brightfield_mode=False):
    """
    Process all stack_X.tiff files in a folder, in numerical order of X.
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return False
    
    # Set default output directory
    if output_dir is None:
        suffix = "brightfield_superimposed" if brightfield_mode else "superimposed"
        output_dir = folder_path / suffix
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing folder: {folder_path}")
    print(f"Output directory: {output_dir}")
    if brightfield_mode:
        print("Brightfield mode: Scaling all images to maximum dynamic range")
    
    # Get stack files sorted by X value
    stack_files = get_stack_files_sorted(folder_path)
    
    if not stack_files:
        print("No stack_X.tiff files found in folder")
        return False
    
    print(f"\nFound {len(stack_files)} stack files:")
    for x_value, filepath in stack_files:
        filename = Path(filepath).name
        print(f"  X={x_value}: {filename}")
    
    # Process each file in order
    successful_count = 0
    
    for i, (x_value, filepath) in enumerate(stack_files, 1):
        print(f"\n=== Processing {i}/{len(stack_files)}: X={x_value} ===")
        
        # Create output filename
        input_file = Path(filepath)
        suffix = "_brightfield_superimposed" if brightfield_mode else "_superimposed"
        output_filename = f"{input_file.stem}{suffix}{input_file.suffix}"
        output_path = output_dir / output_filename
        
        if process_single_file(filepath, output_path, brightfield_mode):
            successful_count += 1
        else:
            print(f"Failed to process X={x_value}")
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed {successful_count}/{len(stack_files)} files")
    if brightfield_mode:
        print("All images scaled to maximum dynamic range")
    print(f"Output directory: {output_dir}")
    
    return successful_count > 0

def main():
    parser = argparse.ArgumentParser(
        description="Process TIFF stack files by averaging frames. Supports both 8-bit and 16-bit images with optional brightfield scaling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single 8-bit TIFF stack
  python fusetiffs.py --inputfile stack_1.tiff
  
  # Process a single 16-bit TIFF stack with brightfield scaling
  python fusetiffs.py --inputfile stack_16bit.tiff --brightfield
  
  # Process a single TIFF stack with custom output
  python fusetiffs.py --inputfile stack_1.tiff --output averaged_result.tiff
  
  # Process all stack_X.tiff files in a folder (in numerical order)
  python fusetiffs.py --inputfolder ./stacks
  
  # Process folder with brightfield scaling and custom output directory
  python fusetiffs.py --inputfolder ./stacks --output ./results --brightfield

Bit Depth Support:
  - 8-bit images: Processed and output as 8-bit (0-255 range)
  - 16-bit images: Processed and output as 16-bit (0-65535 range)
  - Automatic detection: Script detects input bit depth and preserves it
  - Color images: Converted to grayscale while maintaining bit depth
  - Overflow protection: Uses larger data types during averaging to prevent overflow

Brightfield Mode (--brightfield):
  - Scales the averaged image to use the full dynamic range
  - 16-bit images: Scaled to 0-65535 range for maximum contrast
  - 8-bit images: Scaled to 0-255 range for maximum contrast
  - Useful for brightfield microscopy where contrast enhancement is desired
  - Preserves relative intensity differences while maximizing visibility
  - Applied after averaging all frames in the stack

Normal Mode (default):
  - Preserves original intensity values and dynamic range
  - No contrast enhancement applied
  - Suitable for fluorescence microscopy where absolute intensities matter
  - Maintains quantitative relationships between pixel values

Expected format for folder processing:
  - TIFF stacks: stack_X.tiff (e.g., stack_1.tiff, stack_2.tiff, stack_10.tiff)
  - Files will be processed in numerical order of X value
  - Mixed bit depths supported (each file processed according to its bit depth)
  - Brightfield scaling applied individually to each processed stack

Supported Input Modes:
  - L: 8-bit grayscale
  - I;16, I;16B, I;16L: 16-bit grayscale
  - I: 32-bit (usually 16-bit data, automatically detected)
  - RGB, RGBA: Color images (converted to grayscale, bit depth preserved)

Output Naming:
  - Normal mode: filename_superimposed.tiff
  - Brightfield mode: filename_brightfield_superimposed.tiff
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inputfile', help='Process a single TIFF stack file (8-bit or 16-bit)')
    group.add_argument('--inputfolder', help='Process all stack_X.tiff files in folder (mixed bit depths supported)')
    
    parser.add_argument('--output', help='Output file (for single file) or output directory (for folder)')
    parser.add_argument('--brightfield', action='store_true', 
                       help='Enable brightfield mode: scale averaged image to full dynamic range (0-65535 for 16-bit, 0-255 for 8-bit)')
    
    args = parser.parse_args()
    
    try:
        if args.inputfile:
            # Process single TIFF file
            mode_desc = "Brightfield " if args.brightfield else ""
            print(f"=== Single TIFF Stack {mode_desc}Processor (8-bit/16-bit) ===")
            success = process_single_file(args.inputfile, args.output, args.brightfield)
            
        elif args.inputfolder:
            # Process TIFF stack folder
            mode_desc = "Brightfield " if args.brightfield else ""
            print(f"=== TIFF Stack Folder {mode_desc}Processor (8-bit/16-bit) ===")
            success = process_folder(args.inputfolder, args.output, args.brightfield)
        
        if success:
            print("\nProcess completed successfully!")
        else:
            print("\nProcess failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()