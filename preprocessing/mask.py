import argparse
import os
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

def is_tiff_stack(image_path):
    """Check if a TIFF file is a multi-page stack"""
    try:
        with Image.open(image_path) as img:
            # Try to seek to second frame
            img.seek(1)
            return True
    except (EOFError, AttributeError):
        # EOFError: no second frame, AttributeError: seek not supported
        return False
    except Exception:
        return False

def feather_mask(mask, feather_radius):
    """Apply feathering (Gaussian blur) to mask edges for smooth transitions"""
    if feather_radius <= 0:
        return mask
    
    # Convert mask to numpy array for processing
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    
    # Apply Gaussian blur using PIL's filter
    mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
    blurred_mask = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    
    return blurred_mask

def apply_mask_to_frame(image, mask, replace_color=(255, 0, 0), feather_radius=0):
    """Apply a mask to a single image frame using vectorized operations with optional feathering"""
    
    # Apply feathering to mask if specified
    if feather_radius > 0:
        mask = feather_mask(mask, feather_radius)
        print(f"    Applied feathering with radius {feather_radius}")
    
    # Detect if image is grayscale or color and its bit depth
    is_grayscale = image.mode in ['L', 'LA', 'I;16', 'I;16B', 'I;16L']
    is_16bit = image.mode in ['I;16', 'I;16B', 'I;16L']
    
    # Convert images to numpy arrays
    if is_grayscale:
        # For grayscale images, preserve bit depth
        if is_16bit:
            # 16-bit grayscale
            if image.mode != 'I;16':
                image = image.convert('I;16')
            
            # Convert to numpy array with appropriate dtype
            image_array = np.asarray(image, dtype=np.float32)
            
            # Check if input color is already a grayscale value (R=G=B)
            r, g, b = replace_color
            if r == g == b:
                # Direct grayscale value provided - scale to 16-bit if needed
                if 0 <= r <= 255:
                    # Assume 8-bit value, scale to 16-bit
                    replace_value = r * 257  # 257 = 65535/255
                else:
                    # Already 16-bit value or out of range
                    replace_value = min(max(r, 0), 65535)
            else:
                # Convert color replacement to grayscale value using luminance formula
                gray_8bit = int(0.299 * r + 0.587 * g + 0.114 * b)
                replace_value = gray_8bit * 257  # Scale to 16-bit
            
            replace_value = np.float32(replace_value)
        else:
            # 8-bit grayscale
            image = image.convert('L')
            image_array = np.asarray(image, dtype=np.float32)
            
            # Check if input color is already a grayscale value (R=G=B)
            r, g, b = replace_color
            if r == g == b:
                # Direct grayscale value provided
                if 0 <= r <= 255:
                    replace_value = r
                elif r > 255:
                    # Assume 16-bit value, scale down to 8-bit
                    replace_value = min(r // 257, 255)
                else:
                    replace_value = 0
            else:
                # Convert color replacement to grayscale value using luminance formula
                replace_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            
            replace_value = np.float32(replace_value)
    else:
        # Color images
        image = image.convert('RGB')
        image_array = np.asarray(image, dtype=np.float32)
        replace_value = np.array(replace_color, dtype=np.float32)

    # Ensure images are the same size
    if image.size != mask.size:
        raise ValueError(f"Image and mask must be the same size. "
                        f"Image: {image.size}, Mask: {mask.size}")

    # Convert mask to numpy array (0-1 range for blending)
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    
    # For feathered masks, use alpha blending
    if feather_radius > 0:
        # Alpha blending: result = original * (1 - alpha) + replacement * alpha
        # Where alpha comes from the inverted mask (black areas = 1, white areas = 0)
        alpha = 1.0 - mask_array  # Invert mask so black areas have alpha=1
        
        if is_grayscale:
            # For grayscale images, apply blending
            blended_array = image_array * (1.0 - alpha) + replace_value * alpha
        else:
            # For color images, use broadcasting for RGB blending
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            blended_array = image_array * (1.0 - alpha_3d) + replace_value * alpha_3d
        
        # Count pixels that are significantly affected (alpha > 0.1)
        pixels_replaced = np.count_nonzero(alpha > 0.1)
        
        # Convert back to original data type
        if is_16bit:
            final_array = np.clip(blended_array, 0, 65535).astype(np.uint16)
            masked_image = Image.fromarray(final_array, mode='I;16')
        elif is_grayscale:
            final_array = np.clip(blended_array, 0, 255).astype(np.uint8)
            masked_image = Image.fromarray(final_array, mode='L')
        else:
            final_array = np.clip(blended_array, 0, 255).astype(np.uint8)
            masked_image = Image.fromarray(final_array, mode='RGB')
    
    else:
        # Sharp mask: binary replacement (original behavior)
        # Create boolean mask where mask pixels are black (value 0)
        mask_bool = mask_array < 0.5  # Threshold for binary mask
        
        # Count pixels to be replaced
        pixels_replaced = np.count_nonzero(mask_bool)
        
        # Apply mask using vectorized operations
        if is_grayscale:
            # For grayscale images, directly assign the replacement value
            image_array[mask_bool] = replace_value
        else:
            # For color images, use broadcasting to assign RGB values
            image_array[mask_bool] = replace_value
        
        # Convert back to PIL Image
        if is_16bit:
            # For 16-bit images, convert back to the appropriate mode
            masked_image = Image.fromarray(image_array.astype(np.uint16), mode='I;16')
        elif is_grayscale:
            # For 8-bit grayscale
            masked_image = Image.fromarray(image_array.astype(np.uint8), mode='L')
        else:
            # For color images
            masked_image = Image.fromarray(image_array.astype(np.uint8), mode='RGB')

    return masked_image, pixels_replaced

def apply_mask_to_stack(image_path, mask_path, output_path, replace_color=(255, 0, 0), feather_radius=0):
    """Apply a mask to all frames in a TIFF stack using vectorized operations with optional feathering"""
    try:
        # Load the mask once
        mask = Image.open(mask_path).convert('L')
        
        # Apply feathering once if specified
        if feather_radius > 0:
            print(f"  Applying feathering to mask with radius {feather_radius}...")
            mask = feather_mask(mask, feather_radius)
        
        with Image.open(image_path) as img:
            masked_frames = []
            frame_count = 0
            total_pixels_replaced = 0
            is_grayscale = None
            is_16bit = None
            
            # Process each frame in the stack
            try:
                while True:
                    print(f"  Processing frame {frame_count + 1}...")
                    
                    # Get current frame
                    current_frame = img.copy()
                    
                    # Detect grayscale and bit depth on first frame
                    if is_grayscale is None:
                        is_grayscale = current_frame.mode in ['L', 'LA', 'I;16', 'I;16B', 'I;16L']
                        is_16bit = current_frame.mode in ['I;16', 'I;16B', 'I;16L']
                        
                        if is_grayscale:
                            bit_depth = "16-bit" if is_16bit else "8-bit"
                            print(f"    Stack contains {bit_depth} grayscale images")
                        else:
                            print(f"    Stack contains color images")
                    
                    # Apply mask to current frame (feathering already applied to mask)
                    masked_frame, pixels_replaced = apply_mask_to_frame(
                        current_frame, mask, replace_color, feather_radius=0  # Don't re-feather
                    )
                    masked_frames.append(masked_frame)
                    total_pixels_replaced += pixels_replaced
                    
                    frame_count += 1
                    
                    # Move to next frame
                    img.seek(img.tell() + 1)
                    
            except EOFError:
                # End of stack reached
                pass
            
            if not masked_frames:
                print(f"  No frames found in stack: {image_path}")
                return 0, False
            
            print(f"  Found {frame_count} frames in stack")
            
            # Save the masked stack
            # Preserve original metadata if possible
            save_kwargs = {
                'save_all': True,
                'append_images': masked_frames[1:] if len(masked_frames) > 1 else []
            }
            
            # Try to preserve compression and other metadata
            try:
                if hasattr(img, 'tag'):
                    # Get compression from original if available
                    compression_tag = img.tag.get(259)  # Compression tag
                    if compression_tag == 5:  # LZW compression
                        save_kwargs['compression'] = 'tiff_lzw'
                    elif compression_tag == 1:  # No compression
                        save_kwargs['compression'] = None
                
                # For 16-bit images, avoid compression that might cause issues
                if is_16bit:
                    save_kwargs['compression'] = None
                
                # Preserve ImageJ metadata if present
                imagej_tag = img.tag.get(50839)  # ImageJ metadata tag
                if imagej_tag:
                    # Update frame count in ImageJ metadata
                    imagej_info = imagej_tag.decode('utf-8', errors='ignore')
                    if 'images=' in imagej_info:
                        # Update the image count
                        lines = imagej_info.split('\n')
                        updated_lines = []
                        for line in lines:
                            if line.startswith('images='):
                                updated_lines.append(f'images={frame_count}')
                            elif line.startswith('slices='):
                                updated_lines.append(f'slices={frame_count}')
                            else:
                                updated_lines.append(line)
                        updated_imagej_info = '\n'.join(updated_lines)
                        
                        save_kwargs['tiffinfo'] = {
                            50839: updated_imagej_info.encode('utf-8'),
                            270: updated_imagej_info.encode('utf-8')  # ImageDescription
                        }
                        
            except Exception as e:
                print(f"  Warning: Could not preserve all metadata: {e}")
            
            # Save the masked stack
            masked_frames[0].save(output_path, **save_kwargs)
            
            print(f"  Saved masked stack with {frame_count} frames")
            print(f"  Total pixels replaced across all frames: {total_pixels_replaced}")
            if feather_radius > 0:
                print(f"  Applied feathering with radius {feather_radius} pixels")
            return total_pixels_replaced, is_grayscale
            
    except Exception as e:
        print(f"Error processing TIFF stack {image_path}: {e}")
        return 0, False

def apply_mask(image_path, mask_path, output_path, replace_color=(255, 0, 0), feather_radius=0):
    """Apply a mask to an image or TIFF stack, replacing masked areas with a specified color"""
    
    # Check if it's a TIFF stack
    if image_path.lower().endswith(('.tif', '.tiff')) and is_tiff_stack(image_path):
        print(f"Detected TIFF stack: {image_path}")
        return apply_mask_to_stack(image_path, mask_path, output_path, replace_color, feather_radius)
    else:
        # Regular single image processing
        # Open the image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
        
        # Detect bit depth and type
        is_grayscale = image.mode in ['L', 'LA', 'I;16', 'I;16B', 'I;16L']
        is_16bit = image.mode in ['I;16', 'I;16B', 'I;16L']
        
        print(f"  Image mode: {image.mode}")
        if is_grayscale:
            bit_depth = "16-bit" if is_16bit else "8-bit"
            print(f"  Detected {bit_depth} grayscale image")
        else:
            print(f"  Detected color image")
        
        if feather_radius > 0:
            print(f"  Applying feathering with radius {feather_radius}")
        
        # Apply mask to the single frame using vectorized operations
        masked_image, pixels_replaced = apply_mask_to_frame(image, mask, replace_color, feather_radius)
        
        # Save the result
        masked_image.save(output_path)
        return pixels_replaced, is_grayscale

def apply_mask_folder(input_folder, mask_path, output_folder, replace_color=(255, 0, 0), feather_radius=0):
    """Apply a mask to all images in a folder"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    
    # Find all image files
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    print(f"Using mask: {mask_path}")
    if feather_radius > 0:
        print(f"Feathering radius: {feather_radius} pixels")
    
    successful = 0
    failed = 0
    grayscale_count = 0
    color_count = 0
    stacks_processed = 0
    
    for image_file in sorted(image_files):
        output_file = output_path / image_file.name
        
        try:
            # Check if it's a TIFF stack
            if str(image_file).lower().endswith(('.tif', '.tiff')) and is_tiff_stack(str(image_file)):
                stacks_processed += 1
                file_type = "TIFF stack"
            else:
                file_type = "single image"
            
            pixels_replaced, was_grayscale = apply_mask(str(image_file), mask_path, str(output_file), replace_color, feather_radius)
            
            if was_grayscale:
                grayscale_count += 1
                image_type = "grayscale"
            else:
                color_count += 1
                image_type = "color"
            
            feather_info = f", feathered {feather_radius}px" if feather_radius > 0 else ""
            print(f"Processed {image_file.name} -> {output_file.name} ({pixels_replaced} pixels replaced, {image_type} {file_type}{feather_info})")
            successful += 1
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {successful}")
    print(f"    - Color images: {color_count}")
    print(f"    - Grayscale images: {grayscale_count}")
    print(f"    - TIFF stacks: {stacks_processed}")
    if feather_radius > 0:
        print(f"    - Feathering applied: {feather_radius} pixels")
    print(f"  Failed: {failed}")
    print(f"  Output folder: {output_path}")

def parse_color(color_str):
    """Parse color string in format 'R,G,B', hex codes, grayscale values, or common color names"""
    color_names = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
        # Add some grayscale-friendly options
        'lightgray': (211, 211, 211),
        'lightgrey': (211, 211, 211),
        'darkgray': (64, 64, 64),
        'darkgrey': (64, 64, 64),
        # 16-bit grayscale presets
        'white16': (65535, 65535, 65535),
        'lightgray16': (49151, 49151, 49151),  # 75% of 65535
        'gray16': (32767, 32767, 32767),       # 50% of 65535
        'darkgray16': (16383, 16383, 16383),   # 25% of 65535
        'black16': (0, 0, 0)
    }
    
    # Check if it's a named color
    if color_str.lower() in color_names:
        return color_names[color_str.lower()]
    
    # Check if it's a hex color code
    if color_str.startswith('#'):
        try:
            # Remove the '#' and convert to RGB
            hex_color = color_str[1:]
            
            # Support both #RGB and #RRGGBB formats
            if len(hex_color) == 3:
                # Convert #RGB to #RRGGBB
                hex_color = ''.join([c*2 for c in hex_color])
            elif len(hex_color) != 6:
                raise ValueError("Invalid hex length")
            
            # Convert hex to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            return (r, g, b)
            
        except ValueError:
            raise ValueError(f"Invalid hex color format: '{color_str}'. "
                            f"Use #RRGGBB (e.g., '#FF0000') or #RGB (e.g., '#F00') format")
    
    # Check if it's a single grayscale value (0-255 for 8-bit, 0-65535 for 16-bit)
    if ',' not in color_str:
        try:
            gray_value = int(color_str)
            if not 0 <= gray_value <= 65535:
                raise ValueError("Grayscale value must be between 0 and 65535 (0-255 for 8-bit, 0-65535 for 16-bit)")
            # Return as RGB tuple for consistency (will be used directly for grayscale images)
            return (gray_value, gray_value, gray_value)
        except ValueError:
            pass  # Not a valid single number, try R,G,B format
    
    # Try to parse as R,G,B
    try:
        r, g, b = map(int, color_str.split(','))
        if not all(0 <= val <= 65535 for val in [r, g, b]):
            raise ValueError("RGB values must be between 0 and 65535 (use 0-255 for standard 8-bit colors)")
        return (r, g, b)
    except ValueError:
        raise ValueError(f"Invalid color format: '{color_str}'. "
                        f"Use single grayscale value (0-255 for 8-bit, 0-65535 for 16-bit), "
                        f"'R,G,B' format (e.g., '255,0,0' for 8-bit or '65535,0,0' for 16-bit), "
                        f"hex codes (e.g., '#FF0000', '#F00'), "
                        f"or color names ({', '.join(sorted([k for k in color_names.keys() if not k.endswith('16')]))} for 8-bit, "
                        f"{', '.join(sorted([k for k in color_names.keys() if k.endswith('16')]))} for 16-bit)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply a mask to images, replacing masked areas with a specified color. Supports single images, TIFF stacks, and batch processing. Handles both 8-bit and 16-bit grayscale images. Includes feathering for smooth mask transitions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Apply mask to single image with red replacement
  python mask.py image.jpg mask.png output.jpg
  
  # Apply mask with feathering for smooth edges
  python mask.py image.jpg mask.png output.jpg --feather 5
  
  # Apply feathered mask to 8-bit grayscale image
  python mask.py grayscale_8bit.png mask.png output.png --color 128 --feather 3
  
  # Apply feathered mask to 16-bit grayscale image
  python mask.py grayscale_16bit.tiff mask.png output.tiff --color 32767 --feather 8
  
  # Apply feathered mask to TIFF stack (all frames will be masked)
  python mask.py time_series.tiff mask.png masked_series.tiff --feather 4
  
  # Use custom color with feathering (RGB format)
  python mask.py image.jpg mask.png output.jpg --color 0,255,0 --feather 6
  
  # Use 16-bit grayscale values with heavy feathering
  python mask.py 16bit_stack.tiff mask.png masked_stack.tiff --color 65535 --feather 10
  
  # Process all images in a folder with feathering
  python mask.py --input-folder ./images --mask mask.png --output-folder ./masked --color 150 --feather 5

FEATHERING SUPPORT:
  - Gaussian blur applied to mask edges for smooth transitions
  - Feather radius specified in pixels (0 = no feathering, sharp edges)
  - Alpha blending used for smooth color transitions
  - Works with all image types: 8-bit, 16-bit, grayscale, color
  - Applied once to mask, then used for all frames in stacks
  - Larger radius = softer, more gradual transition
  - Typical values: 2-10 pixels for most applications

PERFORMANCE OPTIMIZATIONS:
  - Uses numpy vectorized operations for fast pixel manipulation
  - Processes entire images at once instead of pixel-by-pixel
  - Precomputes feathered masks for stack processing
  - Optimized memory usage with appropriate data types
  - Significant speed improvement for large images and stacks

BIT DEPTH SUPPORT:
  - 8-bit grayscale: Values 0-255, standard grayscale images
  - 16-bit grayscale: Values 0-65535, scientific/medical imaging
  - Automatic detection: Script detects bit depth and handles conversion
  - Preservation: Output maintains same bit depth as input
  - Scaling: 8-bit values automatically scaled to 16-bit when needed
  - Feathering preserves bit depth accuracy

TIFF STACK SUPPORT:
  - Automatically detects multi-frame TIFF files
  - Applies feathered mask to all frames in the stack
  - Preserves ImageJ metadata when possible
  - Maintains original compression settings
  - Reports total pixels replaced across all frames
  - Handles mixed bit depths within stacks

IMAGE TYPES:
  - Color images: Processed in RGB mode, output as RGB
  - 8-bit grayscale: Processed in L mode, output as L
  - 16-bit grayscale: Processed in I;16 mode, output as I;16
  - TIFF stacks: Each frame processed according to its type
  - Automatic bit depth detection and appropriate value scaling
  - Feathering works with all image types

MASK FORMAT:
  - Mask should be a grayscale image
  - Black pixels (value 0) in mask will be replaced with the specified color/gray value
  - White/gray pixels in mask will leave original image unchanged
  - Mask and image must be the same dimensions
  - Same feathered mask applied to all frames in TIFF stacks
  - Feathering creates smooth gradients at mask boundaries

FEATHERING BEHAVIOR:
  - 0 pixels: Sharp, binary mask (original behavior)
  - 1-3 pixels: Subtle softening, good for fine details
  - 4-8 pixels: Moderate feathering, natural transitions
  - 9-15 pixels: Heavy feathering, very soft edges
  - 16+ pixels: Extreme feathering, very gradual transitions

COLOR FORMATS:
  - 8-bit grayscale: '0' to '255' (e.g., '128', '255', '0')
  - 16-bit grayscale: '0' to '65535' (e.g., '32767', '65535', '16383')
  - RGB (8-bit): '255,0,0' (red), '0,255,0' (green), '0,0,255' (blue)
  - RGB (16-bit): '65535,0,0' (red), '0,65535,0' (green), '0,0,65535' (blue)
  - Hex codes: '#FF0000' (red), '#00FF00' (green), '#0000FF' (blue)
  - Short hex: '#F00' (red), '#0F0' (green), '#00F' (blue)
  - Named colors (8-bit): red, green, blue, black, white, yellow, cyan, magenta, 
                          orange, purple, pink, gray, grey, lightgray, darkgray
  - Named colors (16-bit): white16, lightgray16, gray16, darkgray16, black16

VALUE SCALING:
  - 8-bit to 16-bit: Values multiplied by 257 (65535/255)
  - 16-bit to 8-bit: Values divided by 257
  - RGB to grayscale: Luminance formula (0.299*R + 0.587*G + 0.114*B)
  - Automatic range checking and clamping
  - Feathering preserves value scaling accuracy

SUPPORTED FORMATS:
  Input: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .gif (single images or stacks)
  Output: Same format as input (preserves single/stack structure and bit depth)
        """
    )
    
    # Make input and folder mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("image", nargs='?', help="Path to input image or TIFF stack (for single file mode)")
    input_group.add_argument("--input-folder", help="Path to folder containing images (for batch mode)")
    
    # Mask argument
    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument("mask", nargs='?', help="Path to mask image (for single file mode)")
    mask_group.add_argument("--mask", dest="mask_file", help="Path to mask image (for batch mode)")
    
    # Output arguments
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("output", nargs='?', help="Path to save masked image or stack (for single file mode)")
    output_group.add_argument("--output-folder", help="Path to folder for masked images (for batch mode)")
    
    # Color argument
    parser.add_argument("--color", default="red", 
                       help="Replacement color for masked areas. "
                            "Use single grayscale value (0-255 for 8-bit, 0-65535 for 16-bit), "
                            "'R,G,B' format (e.g., '255,0,0' for 8-bit or '65535,0,0' for 16-bit), "
                            "hex codes (e.g., '#FF0000', '#F00'), "
                            "or color names (red, blue, gray16, etc.). Default: red")
    
    # Feathering argument
    parser.add_argument("--feather", type=float, default=0, 
                       help="Feathering radius in pixels for soft mask edges. "
                            "0 = sharp edges (default), 1-3 = subtle, 4-8 = moderate, 9+ = heavy feathering")
    
    args = parser.parse_args()
    
    # Validate feather radius
    if args.feather < 0:
        parser.error("Feather radius must be non-negative")
    
    # Parse replacement color
    try:
        replace_color = parse_color(args.color)
    except ValueError as e:
        parser.error(str(e))
    
    # Validate arguments and process
    if args.input_folder:
        # Folder mode
        if not args.output_folder:
            parser.error("--output-folder is required when using --input-folder")
        
        if not args.mask_file:
            parser.error("--mask is required when using --input-folder")
        
        if not os.path.isdir(args.input_folder):
            parser.error(f"Input folder '{args.input_folder}' does not exist")
        
        if not os.path.isfile(args.mask_file):
            parser.error(f"Mask file '{args.mask_file}' does not exist")
        
        print(f"Processing folder: {args.input_folder}")
        print(f"Output folder: {args.output_folder}")
        print(f"Replacement color: {replace_color}")
        if args.feather > 0:
            print(f"Feathering radius: {args.feather} pixels")
        
        apply_mask_folder(args.input_folder, args.mask_file, args.output_folder, replace_color, args.feather)
    
    else:
        # Single file mode
        if not all([args.image, args.mask, args.output]):
            parser.error("image, mask, and output paths are required for single file mode")
        
        if not os.path.isfile(args.image):
            parser.error(f"Input image '{args.image}' does not exist")
        
        if not os.path.isfile(args.mask):
            parser.error(f"Mask file '{args.mask}' does not exist")
        
        print(f"Processing single file: {args.image}")
        print(f"Using mask: {args.mask}")
        print(f"Replacement color: {replace_color}")
        if args.feather > 0:
            print(f"Feathering radius: {args.feather} pixels")
        
        try:
            pixels_replaced, was_grayscale = apply_mask(args.image, args.mask, args.output, replace_color, args.feather)
            
            image_type = "grayscale" if was_grayscale else "color"
            if args.image.lower().endswith(('.tif', '.tiff')) and is_tiff_stack(args.image):
                file_type = "TIFF stack"
            else:
                file_type = "single image"
            
            feather_info = f" with {args.feather}px feathering" if args.feather > 0 else ""
            print(f"Mask applied successfully! {pixels_replaced} pixels replaced{feather_info}.")
            print(f"Processed {image_type} {file_type}.")
            print(f"Output saved to: {args.output}")
        except Exception as e:
            print(f"Error: {e}")