import argparse
import os
from pathlib import Path
from PIL import Image

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

def crop_tiff_stack(input_path, output_path, left, top, right, bottom):
    """Crop all frames in a TIFF stack"""
    try:
        with Image.open(input_path) as img:
            cropped_frames = []
            frame_count = 0
            
            # Process each frame in the stack
            try:
                while True:
                    print(f"  Processing frame {frame_count + 1}...")
                    
                    # Get current frame
                    current_frame = img.copy()
                    width, height = current_frame.size
                    
                    # Crop the frame
                    cropped_frame = current_frame.crop((left, top, width - right, height - bottom))
                    cropped_frames.append(cropped_frame)
                    
                    frame_count += 1
                    
                    # Move to next frame
                    img.seek(img.tell() + 1)
                    
            except EOFError:
                # End of stack reached
                pass
            
            if not cropped_frames:
                print(f"  No frames found in stack: {input_path}")
                return False
            
            print(f"  Found {frame_count} frames in stack")
            
            # Save the cropped stack
            # Preserve original metadata if possible
            save_kwargs = {
                'save_all': True,
                'append_images': cropped_frames[1:] if len(cropped_frames) > 1 else []
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
            
            # Save the cropped stack
            cropped_frames[0].save(output_path, **save_kwargs)
            
            print(f"  Saved cropped stack with {frame_count} frames")
            return True
            
    except Exception as e:
        print(f"Error processing TIFF stack {input_path}: {e}")
        return False

def crop_image(input_path, output_path, left, top, right, bottom):
    """Crop a single image or TIFF stack"""
    try:
        # Check if it's a TIFF stack
        if input_path.lower().endswith(('.tif', '.tiff')) and is_tiff_stack(input_path):
            print(f"Detected TIFF stack: {input_path}")
            return crop_tiff_stack(input_path, output_path, left, top, right, bottom)
        else:
            # Regular single image
            img = Image.open(input_path)
            width, height = img.size
            cropped = img.crop((left, top, width - right, height - bottom))
            cropped.save(output_path)
            print(f"Cropped {input_path} -> {output_path}")
            return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def crop_folder(input_folder, output_folder, left, top, right, bottom):
    """Crop all images in a folder"""
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
    
    successful = 0
    failed = 0
    stacks_processed = 0
    
    for image_file in sorted(image_files):
        output_file = output_path / image_file.name
        
        print(f"\nProcessing: {image_file.name}")
        
        # Check if it's a TIFF stack
        if str(image_file).lower().endswith(('.tif', '.tiff')) and is_tiff_stack(str(image_file)):
            stacks_processed += 1
        
        if crop_image(str(image_file), str(output_file), left, top, right, bottom):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    if stacks_processed > 0:
        print(f"  TIFF stacks processed: {stacks_processed}")
    print(f"  Output folder: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop images by specifying pixels to remove from each edge. Supports single images, folders, and TIFF stacks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Crop a single image
  python crop.py input.jpg output.jpg --left 50 --top 30 --right 50 --bottom 30
  
  # Crop a TIFF stack (all frames will be cropped)
  python crop.py stack.tiff cropped_stack.tiff --left 100 --right 100
  
  # Crop all images in a folder (including TIFF stacks)
  python crop.py --input-folder ./images --output-folder ./cropped --left 100 --right 100
  
  # Crop with different amounts from each edge
  python crop.py --input-folder ./raw --output-folder ./processed --left 50 --top 25 --right 75 --bottom 25

TIFF STACK SUPPORT:
  - Automatically detects multi-frame TIFF files
  - Crops all frames in the stack
  - Preserves ImageJ metadata when possible
  - Maintains original compression settings

SUPPORTED FORMATS:
  Input: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .gif
  Output: Same format as input
  Special: Multi-frame TIFF stacks are fully supported
        """
    )
    
    # Make input and folder mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("input", nargs='?', help="Path to input image or TIFF stack (for single file mode)")
    input_group.add_argument("--input-folder", help="Path to folder containing images (for batch mode)")
    
    # Output arguments
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("output", nargs='?', help="Path to save cropped image or stack (for single file mode)")
    output_group.add_argument("--output-folder", help="Path to folder for cropped images (for batch mode)")
    
    # Cropping parameters
    parser.add_argument("--left", type=int, default=0, help="Pixels to crop from left edge (default: 0)")
    parser.add_argument("--top", type=int, default=0, help="Pixels to crop from top edge (default: 0)")
    parser.add_argument("--right", type=int, default=0, help="Pixels to crop from right edge (default: 0)")
    parser.add_argument("--bottom", type=int, default=0, help="Pixels to crop from bottom edge (default: 0)")
    
    args = parser.parse_args()
    
    # Validate crop values
    if any(val < 0 for val in [args.left, args.top, args.right, args.bottom]):
        parser.error("Crop values must be non-negative")
    
    # Validate arguments
    if args.input_folder:
        # Folder mode
        if not args.output_folder:
            parser.error("--output-folder is required when using --input-folder")
        
        if not os.path.isdir(args.input_folder):
            parser.error(f"Input folder '{args.input_folder}' does not exist")
        
        print(f"Processing folder: {args.input_folder}")
        print(f"Output folder: {args.output_folder}")
        print(f"Crop settings: left={args.left}, top={args.top}, right={args.right}, bottom={args.bottom}")
        
        crop_folder(args.input_folder, args.output_folder, args.left, args.top, args.right, args.bottom)
    
    else:
        # Single file mode
        if not args.output:
            parser.error("output path is required when processing a single file")
        
        if not os.path.isfile(args.input):
            parser.error(f"Input file '{args.input}' does not exist")
        
        print(f"Processing single file: {args.input}")
        print(f"Crop settings: left={args.left}, top={args.top}, right={args.right}, bottom={args.bottom}")
        
        if crop_image(args.input, args.output, args.left, args.top, args.right, args.bottom):
            print("Processing completed successfully")
        else:
            print("Processing failed")