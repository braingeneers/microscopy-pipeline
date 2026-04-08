import sys
from pathlib import Path
import numpy as np
from PIL import Image

def apply_gamma_correction(rgb_arr, gamma, max_val):
    """Apply gamma correction to RGB array"""
    if gamma == 1.0:
        return rgb_arr  # No correction needed
    
    # Normalize to 0-1 range
    normalized = rgb_arr.astype(np.float64) / max_val
    
    # Apply gamma correction
    gamma_corrected = np.power(normalized, gamma)
    
    # Scale back to original range
    corrected = (gamma_corrected * max_val).astype(rgb_arr.dtype)
    
    return corrected

def grey_to_color(input_path, output_path, color_channel, gamma=1.0):
    """Convert grayscale image to colored version by setting one channel to full brightness"""
    with Image.open(input_path) as img:
        print(f"Processing {input_path.name}: mode={img.mode}, size={img.size}")
        
        # Detect bit depth and convert appropriately
        if img.mode == 'L':
            # 8-bit grayscale
            arr = np.array(img, dtype=np.uint8)
            max_val = 255
            output_dtype = np.uint8
            bit_depth = "8-bit"
        elif img.mode in ['I;16', 'I;16B', 'I;16L']:
            # 16-bit grayscale
            arr = np.array(img, dtype=np.uint16)
            max_val = 65535
            output_dtype = np.uint16
            bit_depth = "16-bit"
        elif img.mode == 'I':
            # 32-bit integer mode, but check if it's actually 16-bit data
            arr_temp = np.array(img)
            if arr_temp.max() <= 65535:
                # Treat as 16-bit
                arr = arr_temp.astype(np.uint16)
                max_val = 65535
                output_dtype = np.uint16
                bit_depth = "16-bit (from I mode)"
            else:
                # True 32-bit, scale down to 16-bit
                arr = (arr_temp * (65535 / arr_temp.max())).astype(np.uint16)
                max_val = 65535
                output_dtype = np.uint16
                bit_depth = "16-bit (scaled from 32-bit)"
        else:
            # Convert other modes to grayscale first
            img_gray = img.convert("L")
            arr = np.array(img_gray, dtype=np.uint8)
            max_val = 255
            output_dtype = np.uint8
            bit_depth = "8-bit (converted from color)"

        print(f"  Detected: {bit_depth}, max_val={max_val}")

        # Create RGB array with appropriate bit depth
        if output_dtype == np.uint16:
            rgb_arr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint16)
        else:
            rgb_arr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

        channels = {'red': 0, 'green': 1, 'blue': 2}
        if color_channel not in channels:
            raise ValueError(f"Invalid color '{color_channel}'. Choose from red, green, blue.")

        # Set the selected channel to full brightness, others to 0%
        for ch_name, ch_idx in channels.items():
            if ch_name == color_channel:
                rgb_arr[..., ch_idx] = arr
            else:
                rgb_arr[..., ch_idx] = (arr * 0).astype(output_dtype)

        # Apply gamma correction after color conversion
        if gamma != 1.0:
            rgb_arr = apply_gamma_correction(rgb_arr, gamma, max_val)
            print(f"  Applied gamma correction: γ={gamma}")

        # Create output image with appropriate mode
        if output_dtype == np.uint16:
            # Convert to 8-bit for RGB mode (scaling down)
            rgb_arr_8bit = (rgb_arr / (max_val / 255)).astype(np.uint8)
            color_img = Image.fromarray(rgb_arr_8bit, mode="RGB")
            
            # For TIFF files, try to preserve 16-bit by saving as separate approach
            if output_path.suffix.lower() in ['.tif', '.tiff']:
                try:
                    color_img.save(output_path)
                    print(f"  Note: 16-bit input converted to 8-bit RGB for compatibility")
                except Exception as e:
                    print(f"  Warning: Could not save as 16-bit, saving as 8-bit: {e}")
                    color_img.save(output_path)
            else:
                color_img.save(output_path)
        else:
            # 8-bit output
            color_img = Image.fromarray(rgb_arr, mode="RGB")
            color_img.save(output_path)

        gamma_suffix = f", γ={gamma}" if gamma != 1.0 else ""
        print(f"  Converted → {output_path.name} ({color_channel}{gamma_suffix})")

def batch_process(input_dir, output_dir, color_channel, gamma=1.0):
    """Process all PNG and TIFF files in input directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all supported image files
    image_extensions = ["*.png", "*.PNG", "*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No PNG or TIFF files found in {input_dir}")
        return

    print(f"Found {len(image_files)} image files to process")
    print(f"Output directory: {output_dir}")
    print(f"Color channel: {color_channel}")
    if gamma != 1.0:
        print(f"Gamma correction: {gamma}")
    print("-" * 50)
    
    processed = 0
    failed = 0
    bit_depth_stats = {'8-bit': 0, '16-bit': 0, 'converted': 0}
    
    for img_path in image_files:
        try:
            # Keep original extension for output
            output_path = output_dir / img_path.name
            
            # Track bit depth statistics
            with Image.open(img_path) as img:
                if img.mode == 'L':
                    bit_depth_stats['8-bit'] += 1
                elif img.mode in ['I;16', 'I;16B', 'I;16L']:
                    bit_depth_stats['16-bit'] += 1
                elif img.mode == 'I':
                    arr_temp = np.array(img)
                    if arr_temp.max() <= 65535:
                        bit_depth_stats['16-bit'] += 1
                    else:
                        bit_depth_stats['converted'] += 1
                else:
                    bit_depth_stats['converted'] += 1
            
            grey_to_color(img_path, output_path, color_channel, gamma)
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            failed += 1

    print("-" * 50)
    print(f"Processing complete:")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"\nBit depth statistics:")
    print(f"  8-bit images: {bit_depth_stats['8-bit']}")
    print(f"  16-bit images: {bit_depth_stats['16-bit']}")
    print(f"  Converted images: {bit_depth_stats['converted']}")
    
    if bit_depth_stats['16-bit'] > 0 or bit_depth_stats['converted'] > 0:
        print(f"\nNote: 16-bit images are converted to 8-bit RGB for color output")
        print(f"This is due to PIL's RGB mode limitation (8-bit per channel only)")
    
    if gamma != 1.0:
        print(f"\nGamma correction applied:")
        if gamma < 1.0:
            print(f"  γ={gamma} brightens dark areas (gamma correction)")
        else:
            print(f"  γ={gamma} darkens bright areas (gamma correction)")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python grey_to_color.py <input_dir> <output_dir> <color> [--gamma value]")
        print()
        print("Arguments:")
        print("  input_dir   : Directory containing grayscale images")
        print("  output_dir  : Directory to save colored images")
        print("  color       : Target color channel (red, green, blue)")
        print()
        print("Options:")
        print("  --gamma VALUE : Gamma correction value (default: 1.0, no correction)")
        print("                  < 1.0 brightens dark areas (e.g., 0.5)")
        print("                  > 1.0 darkens bright areas (e.g., 2.0)")
        print()
        print("Supported formats:")
        print("  Input:  PNG, TIFF (8-bit and 16-bit, case-insensitive)")
        print("  Output: RGB images (8-bit per channel due to PIL limitations)")
        print()
        print("Bit depth handling:")
        print("  8-bit grayscale  → 8-bit RGB")
        print("  16-bit grayscale → 8-bit RGB (scaled)")
        print("  Other modes      → 8-bit RGB (converted)")
        print()
        print("Gamma correction:")
        print("  Applied after color conversion")
        print("  γ < 1.0: Brightens image (more details in shadows)")
        print("  γ > 1.0: Darkens image (more contrast)")
        print("  γ = 1.0: No correction (linear, default)")
        print()
        print("Examples:")
        print("  python grey_to_color.py ./grayscale ./red_output red")
        print("  python grey_to_color.py ./images ./green_output green --gamma 0.7")
        print("  python grey_to_color.py ./16bit_tiffs ./blue_output blue --gamma 2.2")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    color_channel = sys.argv[3].lower()
    
    # Default gamma value
    gamma = 1.0
    
    # Parse optional gamma argument
    if len(sys.argv) > 4:
        i = 4
        while i < len(sys.argv):
            if sys.argv[i] == '--gamma' and i + 1 < len(sys.argv):
                try:
                    gamma = float(sys.argv[i + 1])
                    if gamma <= 0:
                        print("Error: Gamma value must be positive")
                        sys.exit(1)
                    i += 2
                except ValueError:
                    print(f"Error: Invalid gamma value '{sys.argv[i + 1]}'. Must be a positive number.")
                    sys.exit(1)
            else:
                print(f"Error: Unknown argument '{sys.argv[i]}'")
                sys.exit(1)

    if color_channel not in ['red', 'green', 'blue']:
        print(f"Error: Invalid color '{color_channel}'. Choose from red, green, blue.")
        sys.exit(1)

    batch_process(input_dir, output_dir, color_channel, gamma)