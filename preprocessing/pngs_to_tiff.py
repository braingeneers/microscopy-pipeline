import os
import sys
import glob
import re
from PIL import Image
from PIL.TiffTags import TAGS
from collections import defaultdict

def parse_filename(filename):
    """
    Parse filename in format X_Y.png and return X, Y as integers.
    Returns None if filename doesn't match the expected format.
    """
    match = re.match(r'(\d+)_(-?\d+)\.png$', filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    return None

def group_files_by_x(png_dir):
    """
    Group PNG files by their X value.
    Returns a dictionary where keys are X values and values are lists of (Y, filepath) tuples.
    """
    png_files = glob.glob(os.path.join(png_dir, '*.png'))
    
    files_by_x = defaultdict(list)
    
    for png_file in png_files:
        filename = os.path.basename(png_file)
        parsed = parse_filename(filename)
        
        if parsed:
            x, y = parsed
            files_by_x[x].append((y, png_file))
        else:
            print(f"Warning: Skipping file '{filename}' - doesn't match X_Y.png format")
    
    # Sort Y values for each X
    for x in files_by_x:
        files_by_x[x].sort(key=lambda item: item[0])  # Sort by Y value
    
    return files_by_x

def create_imagej_metadata(num_images, width, height, bit_depth=8):
    """
    Create ImageJ-compatible metadata for TIFF stack.
    Returns the ImageJ metadata string and TIFF tags.
    """
    # Set min/max values based on bit depth
    if bit_depth == 16:
        max_val = 65535.0
        mode = "grayscale"
    else:
        max_val = 255.0
        mode = "grayscale"
    
    # ImageJ metadata string
    imagej_info = f"ImageJ=1.53t\nimages={num_images}\nchannels=1\nslices={num_images}\nframes=1\nhyperstack=false\nmode={mode}\nloop=false\nmin=0.0\nmax={max_val}\n"
    
    # TIFF tags for ImageJ compatibility
    tiff_tags = {
        # ImageJ metadata tag (50839)
        50839: imagej_info.encode('utf-8'),
        # Software tag
        305: "ImageJ",
        # Document name
        269: f"Stack ({num_images} slices)",
        # Image description with ImageJ format
        270: f"ImageJ=1.53t\nimages={num_images}\nslices={num_images}\nchannels=1\nframes=1\nhyperstack=false\nmode={mode}\nloop=false\n"
    }
    
    return tiff_tags

def get_bit_depth(img):
    """
    Determine the bit depth of an image.
    Returns 8 or 16 based on the image mode and format.
    """
    if img.mode == 'I;16' or img.mode == 'I;16B' or img.mode == 'I;16L':
        return 16
    elif img.mode == 'I' and hasattr(img, 'tag') and img.tag.get(258):  # BitsPerSample tag
        bits_per_sample = img.tag[258]
        if isinstance(bits_per_sample, (list, tuple)):
            return bits_per_sample[0]
        else:
            return bits_per_sample
    elif img.mode == 'L':
        return 8
    elif img.mode == 'RGB' or img.mode == 'RGBA':
        return 8
    else:
        # Default assumption
        return 8

def create_stack_for_x(x_value, y_files, output_dir):
    """
    Create an ImageJ-compatible TIFF stack for a specific X value.
    y_files: List of (Y, filepath) tuples sorted by Y value.
    """
    if not y_files:
        print(f"Warning: No files found for X={x_value}")
        return False
    
    print(f"\nCreating stack for X={x_value} with {len(y_files)} files:")
    
    # Extract file paths in Y order
    png_files = [filepath for y, filepath in y_files]
    y_values = [y for y, filepath in y_files]
    
    print(f"  Y range: {min(y_values)} to {max(y_values)}")
    
    # Load all PNG images and determine bit depth
    tiff_images = []
    bit_depth = None
    
    for i, (y, png_file) in enumerate(y_files):
        print(f"    Loading slice {i+1}/{len(y_files)}: Y={y} from {os.path.basename(png_file)}")
        
        img = Image.open(png_file)
        
        # Determine bit depth from first image
        if bit_depth is None:
            bit_depth = get_bit_depth(img)
            print(f"    Detected bit depth: {bit_depth}-bit")
        
        # Handle conversion based on bit depth
        if bit_depth == 16:
            # Preserve 16-bit data
            if img.mode in ['I;16', 'I;16B', 'I;16L']:
                # Already 16-bit, keep as is
                converted_img = img
            elif img.mode == 'I':
                # 32-bit integer, convert to 16-bit
                converted_img = img.point(lambda x: min(x, 65535)).convert('I;16')
            elif img.mode == 'L':
                # 8-bit, scale up to 16-bit
                converted_img = img.point(lambda x: x * 257).convert('I;16')  # 257 = 65535/255
            elif img.mode in ['RGB', 'RGBA']:
                # Convert color to grayscale first, then to 16-bit
                gray_img = img.convert('L')
                converted_img = gray_img.point(lambda x: x * 257).convert('I;16')
            else:
                # Fallback: convert to 8-bit first, then scale to 16-bit
                gray_img = img.convert('L')
                converted_img = gray_img.point(lambda x: x * 257).convert('I;16')
        else:
            # Convert to 8-bit grayscale
            if img.mode in ['I;16', 'I;16B', 'I;16L']:
                # 16-bit to 8-bit, scale down
                converted_img = img.point(lambda x: x // 257).convert('L')  # 257 = 65535/255
            elif img.mode == 'I':
                # 32-bit integer, convert to 8-bit
                converted_img = img.point(lambda x: min(x // 257, 255)).convert('L')
            elif img.mode != 'L':
                # Convert any other mode to grayscale
                converted_img = img.convert('L')
            else:
                # Already 8-bit grayscale
                converted_img = img
        
        tiff_images.append(converted_img)
    
    # Get image dimensions
    width, height = tiff_images[0].size
    num_images = len(tiff_images)
    
    # Create output filename
    output_filename = f"stack_{x_value}.tiff"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"  Creating ImageJ-compatible stack: {output_filename}")
    print(f"  Output bit depth: {bit_depth}-bit")
    
    # Create ImageJ metadata with appropriate bit depth
    imagej_tags = create_imagej_metadata(num_images, width, height, bit_depth)
    
    try:
        # Save as multi-page TIFF with ImageJ metadata
        if bit_depth == 16:
            # For 16-bit images, don't use LZW compression as it can cause issues
            tiff_images[0].save(
                output_path,
                save_all=True,
                append_images=tiff_images[1:],
                compression=None,  # No compression for 16-bit to ensure compatibility
                tiffinfo=imagej_tags,
                resolution_unit=1,
                resolution=(72.0, 72.0)
            )
        else:
            # For 8-bit images, LZW compression is fine
            tiff_images[0].save(
                output_path,
                save_all=True,
                append_images=tiff_images[1:],
                compression='tiff_lzw',
                tiffinfo=imagej_tags,
                resolution_unit=1,
                resolution=(72.0, 72.0)
            )
        
        print(f"  ✓ Successfully saved: {output_path}")
        print(f"  Stack contains {num_images} slices of size {width}x{height} ({bit_depth}-bit)")
        return True
        
    except Exception as e:
        print(f"  ✗ Error creating stack for X={x_value}: {e}")
        return False

def create_multiple_stacks(png_dir, output_dir):
    """
    Create multiple TIFF stacks, one for each X value found in the PNG directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group files by X value
    print("Analyzing files...")
    files_by_x = group_files_by_x(png_dir)
    
    if not files_by_x:
        print("Error: No valid X_Y.png files found")
        return False
    
    x_values = sorted(files_by_x.keys())
    print(f"Found files for X values: {x_values}")
    
    successful_stacks = 0
    total_stacks = len(x_values)
    
    # Process each X value
    for i, x_value in enumerate(x_values, 1):
        print(f"\n=== Processing X={x_value} ({i}/{total_stacks}) ===")
        
        y_files = files_by_x[x_value]
        if create_stack_for_x(x_value, y_files, output_dir):
            successful_stacks += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully created {successful_stacks}/{total_stacks} stacks")
    print(f"Output directory: {output_dir}")
    
    return successful_stacks > 0

def analyze_existing_files(png_dir):
    """
    Analyze and report on existing PNG files in the directory.
    """
    print(f"Analyzing PNG files in: {png_dir}")
    
    png_files = glob.glob(os.path.join(png_dir, '*.png'))
    
    if not png_files:
        print("No PNG files found")
        return
    
    print(f"Found {len(png_files)} PNG files")
    
    files_by_x = group_files_by_x(png_dir)
    
    if not files_by_x:
        print("No files match the X_Y.png format")
        return
    
    # Analyze bit depths
    bit_depths = {}
    sample_images = []
    
    for x in sorted(files_by_x.keys()):
        y_files = files_by_x[x]
        if y_files:
            # Check first file of each X group
            sample_file = y_files[0][1]
            try:
                with Image.open(sample_file) as img:
                    bit_depth = get_bit_depth(img)
                    bit_depths[x] = bit_depth
                    sample_images.append((x, img.mode, bit_depth, img.size))
            except Exception as e:
                print(f"Warning: Could not analyze {sample_file}: {e}")
    
    print(f"\nFile groups by X value:")
    for x in sorted(files_by_x.keys()):
        y_values = [y for y, filepath in files_by_x[x]]
        bit_info = f", {bit_depths.get(x, 'unknown')}-bit" if x in bit_depths else ""
        print(f"  X={x}: {len(y_values)} files, Y range: {min(y_values)} to {max(y_values)}{bit_info}")
    
    if sample_images:
        print(f"\nImage properties (sample from each X group):")
        for x, mode, bit_depth, size in sample_images:
            print(f"  X={x}: Mode={mode}, Bit depth={bit_depth}, Size={size[0]}x{size[1]}")
    
    total_valid = sum(len(files) for files in files_by_x.values())
    print(f"\nTotal valid files: {total_valid}/{len(png_files)}")
    
    # Report bit depth distribution
    if bit_depths:
        bit_depth_counts = {}
        for bd in bit_depths.values():
            bit_depth_counts[bd] = bit_depth_counts.get(bd, 0) + 1
        
        print(f"\nBit depth distribution:")
        for bd, count in sorted(bit_depth_counts.items()):
            print(f"  {bd}-bit: {count} groups")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pngs_to_tiff.py <png_directory> [output_directory] [--analyze]")
        print("\nOptions:")
        print("  png_directory    - Directory containing PNG files named X_Y.png")
        print("  output_directory - Directory to save TIFF stacks (default: same as PNG directory)")
        print("  --analyze        - Only analyze files, don't create stacks")
        print("\nExamples:")
        print("  python pngs_to_tiff.py ./pngs")
        print("  python pngs_to_tiff.py ./pngs ./stacks")
        print("  python pngs_to_tiff.py ./pngs --analyze")
        print("\nExpected filename format: X_Y.png where:")
        print("  X = positive integer (stack group)")
        print("  Y = positive or negative integer (slice order)")
        print("  Examples: 1_0.png, 1_5.png, 2_-10.png, 2_15.png")
        sys.exit(1)
    
    png_dir = sys.argv[1]
    
    # Check for analyze flag
    if len(sys.argv) > 2 and sys.argv[2] == '--analyze':
        analyze_existing_files(png_dir)
        sys.exit(0)
    
    # Optional output directory (default: same as PNG directory)
    if len(sys.argv) > 2 and sys.argv[2] != '--analyze':
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.join(png_dir, "stacks")
    
    # Check if PNG directory exists
    if not os.path.exists(png_dir):
        print(f"Error: PNG directory '{png_dir}' does not exist")
        sys.exit(1)
    
    print("=== PNG to ImageJ TIFF Stack Converter ===")
    print(f"Input directory: {png_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        success = create_multiple_stacks(png_dir, output_dir)
        if success:
            print("\nProcess completed successfully!")
            print("The TIFF stacks should now open correctly in ImageJ/FIJI as multi-slice stacks.")
        else:
            print("\nProcess failed - no stacks were created")
            sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)