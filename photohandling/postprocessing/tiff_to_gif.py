import os
import re
from PIL import Image
import sys
import numpy as np

def normalize_16bit_to_8bit(image):
    """Convert 16-bit image to 8-bit by normalizing the intensity range"""
    if image.mode in ['I;16', 'I;16B', 'I;16L', 'I']:
        # Convert to numpy array
        arr = np.array(image)
        
        # Check if it's actually 16-bit data
        if arr.max() > 255:
            print(f"    16-bit image detected (max value: {arr.max()})")
            # Normalize to 0-255 range
            if arr.max() > arr.min():
                arr_normalized = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
            else:
                arr_normalized = np.zeros_like(arr, dtype=np.uint8)
            
            # Convert back to PIL Image
            return Image.fromarray(arr_normalized, mode='L')
        else:
            # Data is already in 8-bit range
            return image.convert('L')
    else:
        # Already 8-bit or other format
        return image

def get_tiff_files(folder):
    pattern = re.compile(r'stack_(\d+)_edf\.tif$', re.IGNORECASE)
    files = []
    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            n = int(match.group(1))
            files.append((n, fname))
    files.sort(key=lambda x: x[0])
    return [os.path.join(folder, fname) for _, fname in files]

def tiff_to_gif(folder, output_gif):
    tiff_files = get_tiff_files(folder)
    if not tiff_files:
        print("No matching TIFF files found.")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    images = []
    
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"Processing [{i}/{len(tiff_files)}]: {os.path.basename(tiff_file)}")
        
        try:
            # Open image
            img = Image.open(tiff_file)
            print(f"    Image mode: {img.mode}, size: {img.size}")
            
            # Handle 16-bit images
            if img.mode in ['I;16', 'I;16B', 'I;16L', 'I']:
                img_normalized = normalize_16bit_to_8bit(img)
                img_rgb = img_normalized.convert('RGB')
                print(f"    Normalized 16-bit to 8-bit RGB")
            else:
                # Convert other modes to RGB
                img_rgb = img.convert('RGB')
                print(f"    Converted to RGB")
            
            images.append(img_rgb)
            
        except Exception as e:
            print(f"    Error processing {tiff_file}: {e}")
            continue
    
    if not images:
        print("No images could be processed successfully.")
        return
    
    print(f"\nCreating animated GIF with {len(images)} frames...")
    
    # Save as animated GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=10,
        loop=0,
        optimize=True
    )
    
    print(f"Animated GIF saved as {output_gif}")
    print(f"Frame count: {len(images)}")
    print(f"Duration per frame: 10ms")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tiff_to_gif.py <tiff_folder> <output_gif>")
        print()
        print("Description:")
        print("  Converts a series of TIFF files to an animated GIF")
        print("  Looks for files matching pattern: stack_N_edf.tif")
        print("  Supports both 8-bit and 16-bit TIFF images")
        print()
        print("Examples:")
        print("  python tiff_to_gif.py ./tiff_stack animation.gif")
        print("  python tiff_to_gif.py ./microscopy_images timelapse.gif")
        print()
        print("Features:")
        print("  - Automatic 16-bit to 8-bit normalization")
        print("  - Natural sorting by stack number")
        print("  - Progress reporting")
        print("  - Error handling for individual files")
    else:
        tiff_to_gif(sys.argv[1], sys.argv[2])