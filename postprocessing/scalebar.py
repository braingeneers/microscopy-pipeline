import cv2
import numpy as np
import sys
import os
from pathlib import Path
import argparse

def add_scale_bar(image_path, output_path, scale_length_um, pixel_size_um, bar_height=10, margin=30, color=(255,255,255)):
    """
    Add a scale bar to a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        scale_length_um: Length of scale bar in micrometers
        pixel_size_um: Size of one pixel in micrometers
        bar_height: Height of scale bar in pixels
        margin: Margin from image edge in pixels
        color: RGB color tuple for scale bar and text
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    # Calculate scale bar length in pixels
    scale_bar_length_px = int(scale_length_um / pixel_size_um)
    
    # Validate scale bar fits in image
    h, w = img.shape[:2]
    if scale_bar_length_px + 2 * margin > w:
        raise ValueError(f"Scale bar too long ({scale_bar_length_px}px) for image width ({w}px)")

    # Position: bottom left with margin
    x_start = margin
    y_start = h - margin - bar_height

    # Draw scale bar
    cv2.rectangle(img, (x_start, y_start), (x_start + scale_bar_length_px, y_start + bar_height), color, -1)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{scale_length_um} μm"
    text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
    text_x = x_start
    text_y = y_start - 10
    cv2.putText(img, text, (text_x, text_y), font, 0.8, color, 2, cv2.LINE_AA)

    # Save output
    success = cv2.imwrite(output_path, img)
    if not success:
        raise ValueError(f"Failed to save image to: {output_path}")
    
    return scale_bar_length_px

def batch_process_images(input_dir, output_dir, scale_length_um, pixel_size_um, bar_height=10, margin=30, color=(255,255,255)):
    """
    Add scale bars to all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        scale_length_um: Length of scale bar in micrometers
        pixel_size_um: Size of one pixel in micrometers
        bar_height: Height of scale bar in pixels
        margin: Margin from image edge in pixels
        color: RGB color tuple for scale bar and text
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input directory
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
    
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
    print(f"Scale bar: {scale_length_um} μm ({scale_length_um / pixel_size_um:.1f} pixels)")
    print(f"Pixel size: {pixel_size_um} μm/pixel")
    print(f"Bar parameters: height={bar_height}px, margin={margin}px, color={color}")
    print("-" * 60)
    
    processed = 0
    failed = 0
    
    for i, img_file in enumerate(image_files, 1):
        # Create output filename (preserve extension)
        output_file = output_path / img_file.name
        
        print(f"[{i:3d}/{len(image_files)}] Processing: {img_file.name}")
        
        try:
            # Get image dimensions for reporting
            img_temp = cv2.imread(str(img_file))
            if img_temp is not None:
                h, w = img_temp.shape[:2]
                print(f"    Image size: {w}x{h} pixels")
            
            # Add scale bar
            scale_bar_px = add_scale_bar(
                str(img_file),
                str(output_file),
                scale_length_um,
                pixel_size_um,
                bar_height,
                margin,
                color
            )
            
            print(f"    ✓ Scale bar added: {scale_bar_px} pixels")
            print(f"    ✓ Saved to: {output_file.name}")
            processed += 1
            
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

def parse_color(color_str):
    """Parse color string to RGB tuple"""
    if color_str.lower() == 'white':
        return (255, 255, 255)
    elif color_str.lower() == 'black':
        return (0, 0, 0)
    elif color_str.lower() == 'red':
        return (0, 0, 255)  # BGR format for OpenCV
    elif color_str.lower() == 'green':
        return (0, 255, 0)
    elif color_str.lower() == 'blue':
        return (255, 0, 0)
    elif color_str.lower() == 'yellow':
        return (0, 255, 255)
    elif color_str.lower() == 'cyan':
        return (255, 255, 0)
    elif color_str.lower() == 'magenta':
        return (255, 0, 255)
    else:
        # Try to parse as RGB values (e.g., "255,255,255")
        try:
            rgb_values = [int(x.strip()) for x in color_str.split(',')]
            if len(rgb_values) == 3 and all(0 <= x <= 255 for x in rgb_values):
                # Convert RGB to BGR for OpenCV
                return (rgb_values[2], rgb_values[1], rgb_values[0])
            else:
                raise ValueError("RGB values must be 0-255")
        except:
            raise ValueError(f"Invalid color: {color_str}")

def main():
    parser = argparse.ArgumentParser(
        description="Add scale bars to microscopy images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python scalebar.py input.jpg output.jpg --scale 100 --pixel-size 0.5

  # Batch processing
  python scalebar.py ./input_folder ./output_folder --scale 50 --pixel-size 0.25

  # Custom appearance
  python scalebar.py input.jpg output.jpg --scale 200 --pixel-size 1.0 --height 15 --margin 40 --color yellow

  # High magnification (small pixel size)
  python scalebar.py ./images ./scaled --scale 10 --pixel-size 0.1

Supported colors: white, black, red, green, blue, yellow, cyan, magenta, or RGB values (e.g., "255,128,0")
Supported formats: JPG, PNG, TIFF, BMP, WEBP
        """)
    
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('output', help='Output image file or directory')
    parser.add_argument('--scale', type=float, required=True, 
                       help='Scale bar length in micrometers')
    parser.add_argument('--pixel-size', type=float, required=True,
                       help='Pixel size in micrometers per pixel')
    parser.add_argument('--height', type=int, default=10,
                       help='Scale bar height in pixels (default: 10)')
    parser.add_argument('--margin', type=int, default=30,
                       help='Margin from image edge in pixels (default: 30)')
    parser.add_argument('--color', type=str, default='white',
                       help='Scale bar color (default: white)')
    
    args = parser.parse_args()
    
    # Parse color
    try:
        color = parse_color(args.color)
    except ValueError as e:
        print(f"Error: {e}")
        print("Supported colors: white, black, red, green, blue, yellow, cyan, magenta")
        print("Or RGB values like: 255,128,0")
        sys.exit(1)
    
    # Validate parameters
    if args.scale <= 0:
        print("Error: Scale length must be positive")
        sys.exit(1)
    
    if args.pixel_size <= 0:
        print("Error: Pixel size must be positive")
        sys.exit(1)
    
    if args.height <= 0:
        print("Error: Bar height must be positive")
        sys.exit(1)
    
    if args.margin < 0:
        print("Error: Margin must be non-negative")
        sys.exit(1)
    
    # Determine if input is file or directory
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Single file processing
            print(f"Processing single image: {args.input}")
            
            # Get image info
            img_temp = cv2.imread(args.input)
            if img_temp is not None:
                h, w = img_temp.shape[:2]
                scale_bar_px = args.scale / args.pixel_size
                print(f"Image size: {w}x{h} pixels")
                print(f"Scale bar: {args.scale} μm ({scale_bar_px:.1f} pixels)")
                print(f"Pixel size: {args.pixel_size} μm/pixel")
            
            scale_bar_px = add_scale_bar(
                args.input,
                args.output,
                args.scale,
                args.pixel_size,
                args.height,
                args.margin,
                color
            )
            
            print(f"✓ Scale bar added successfully")
            print(f"✓ Saved to: {args.output}")
            
        elif input_path.is_dir():
            # Batch processing
            print(f"Batch processing directory: {args.input}")
            batch_process_images(
                args.input,
                args.output,
                args.scale,
                args.pixel_size,
                args.height,
                args.margin,
                color
            )
            
        else:
            print(f"Error: '{args.input}' is neither a file nor a directory")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()