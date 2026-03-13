import numpy as np
from scipy.signal import convolve2d
import pywt
from PIL import Image
import os
import argparse
import csv

def complex_wavelet_edf(images, wavelet='db3', levels=3, bit_depth=16, top_n=1):
    """
    Extended Depth of Field using Complex Wavelets.
    Args:
        images (list of np.ndarray): List of grayscale images (2D arrays) to fuse.
        wavelet (str): Wavelet type (default 'db3' - 6 coefficients).
        levels (int): Number of decomposition levels.
        bit_depth (int): Output bit depth (8 or 16).
        top_n (int): Number of top layers to average based on wavelet metric (default: 1).
    Returns:
        tuple: (fused_image, min_z_used, max_z_used)
    """
    
    # Determine max value based on bit depth
    max_val = 255 if bit_depth == 8 else 65535
    
    # Convert images to float for processing
    float_images = [img.astype(np.float64) for img in images]

    def complex_wavelet_decompose(img, wavelet, levels):
        coeffs = pywt.wavedec2(img, wavelet, level=levels)
        complex_coeffs = []
        for c in coeffs:
            if isinstance(c, tuple):
                cH, cV, cD = c
                complex_c = cH + 1j * cV
                complex_coeffs.append((complex_c, cD))
            else:
                complex_coeffs.append(c)
        return complex_coeffs

    def complex_wavelet_reconstruct(coeffs, wavelet):
        real_coeffs = []
        for c in coeffs:
            if isinstance(c, tuple):
                complex_c, cD = c
                cH = np.real(complex_c)
                cV = np.imag(complex_c)
                real_coeffs.append((cH, cV, cD))
            else:
                real_coeffs.append(c)
        return pywt.waverec2(real_coeffs, wavelet)

    # Track z-indices used across the entire image
    z_indices_used = []

    # Decompose all images
    decomposed = [complex_wavelet_decompose(img, wavelet, levels) for img in float_images]

    # Enhanced fusion rule with top-N selection
    fused_coeffs = []
    for level_idx, coeffs_per_level in enumerate(zip(*decomposed)):
        if isinstance(coeffs_per_level[0], tuple):
            complex_c_list, cD_list = zip(*coeffs_per_level)
            
            # Calculate magnitude maps without smoothing
            mags = [np.abs(c) for c in complex_c_list]
            
            # Stack magnitudes for easy sorting
            mag_stack = np.stack(mags, axis=0)  # Shape: (num_images, height, width)
            
            # Get indices of top N magnitudes at each pixel
            top_n_clamped = min(top_n, len(complex_c_list))  # Don't exceed number of images
            top_indices = np.argpartition(-mag_stack, top_n_clamped-1, axis=0)[:top_n_clamped]
            
            # Track z-indices used (only for the first level to avoid redundancy)
            if level_idx == 0:
                z_indices_used.extend(top_indices.flatten())
            
            # Create coefficient stacks for indexing
            complex_c_stack = np.stack(complex_c_list, axis=0)
            cD_stack = np.stack(cD_list, axis=0)
            
            # Initialize fused coefficients
            fused_complex_c = np.zeros_like(complex_c_list[0])
            fused_cD = np.zeros_like(cD_list[0])
            
            # Average top N coefficients at each pixel
            for i in range(top_n_clamped):
                # Get the i-th best coefficient at each pixel
                idx_map = top_indices[i]  # Shape: (height, width)
                
                # Use advanced indexing to select coefficients
                selected_complex_c = np.choose(idx_map, complex_c_stack)
                selected_cD = np.choose(idx_map, cD_stack)
                
                fused_complex_c += selected_complex_c
                fused_cD += selected_cD
            
            # Average the selected coefficients
            fused_complex_c /= top_n_clamped
            fused_cD /= top_n_clamped
            
            fused_coeffs.append((fused_complex_c, fused_cD))
        else:
            # Approximation coefficients: top-N simple average based on coefficient magnitude
            mags = [np.abs(coeff) for coeff in coeffs_per_level]
            
            # Stack magnitudes and coefficients for easy sorting
            mag_stack = np.stack(mags, axis=0)
            coeff_stack = np.stack(coeffs_per_level, axis=0)
            
            # Get indices of top N magnitudes at each pixel
            top_n_clamped = min(top_n, len(coeffs_per_level))
            top_indices = np.argpartition(-mag_stack, top_n_clamped-1, axis=0)[:top_n_clamped]
            
            # Track z-indices used (only for the first level to avoid redundancy)
            if level_idx == 0:
                z_indices_used.extend(top_indices.flatten())
            
            # Initialize fused approximation
            fused_approx = np.zeros_like(coeffs_per_level[0])
            
            # Simple average of top N coefficients
            for i in range(top_n_clamped):
                idx_map = top_indices[i]
                selected_coeff = np.choose(idx_map, coeff_stack)
                fused_approx += selected_coeff
            
            # Average the selected coefficients
            fused_approx /= top_n_clamped
            
            fused_coeffs.append(fused_approx)

    # Reconstruct fused image
    fused_img = complex_wavelet_reconstruct(fused_coeffs, wavelet)
    
    # Calculate overall min and max z-indices used
    if z_indices_used:
        min_z_used = int(np.min(z_indices_used))
        max_z_used = int(np.max(z_indices_used))
    else:
        min_z_used = 0
        max_z_used = len(images) - 1
    
    # Clip to appropriate range and convert to target bit depth
    return np.clip(fused_img, 0, max_val), min_z_used, max_z_used

def complex_wavelet_edf_from_file(input_path, output_path, wavelet='db3', levels=3, bit_depth=16, top_n=1):
    """
    Extended Depth of Field using Complex Wavelets from TIFF stack.
    Args:
        input_path (str): Path to input TIFF stack file.
        output_path (str): Path to output fused TIFF file.
        wavelet (str): Wavelet type (default 'db3' - 6 coefficients).
        levels (int): Number of decomposition levels.
        bit_depth (int): Output bit depth (8 or 16).
        top_n (int): Number of top layers to average based on wavelet metric (default: 1).
    Returns:
        tuple: (min_z_used, max_z_used)
    """
    
    # Read TIFF stack
    images = []
    original_mode = None
    input_bit_depth = 8  # Default assumption
    
    with Image.open(input_path) as img:
        original_mode = img.mode
        print(f"  Input image mode: {original_mode}")
        
        # Determine input bit depth
        if img.mode == 'I;16':
            input_bit_depth = 16
        elif img.mode == 'L':
            input_bit_depth = 8
            
        print(f"  Input bit depth: {input_bit_depth}, Output bit depth: {bit_depth}")
        
        try:
            while True:
                # Convert to grayscale if needed, preserve bit depth
                if img.mode == 'I;16':
                    # 16-bit grayscale
                    frame = img
                elif img.mode == 'L':
                    # 8-bit grayscale
                    frame = img
                else:
                    # Convert color to grayscale, preserve bit depth if possible
                    frame = img.convert('L')
                
                # Convert to numpy array and scale if needed
                img_array = np.array(frame)
                
                # If input is 8-bit and output is 16-bit, scale up
                if input_bit_depth == 8 and bit_depth == 16:
                    img_array = img_array.astype(np.uint16) * 257  # Scale 0-255 to 0-65535
                # If input is 16-bit and output is 8-bit, scale down
                elif input_bit_depth == 16 and bit_depth == 8:
                    img_array = (img_array.astype(np.float32) / 257).astype(np.uint8)  # Scale 0-65535 to 0-255
                
                images.append(img_array)
                img.seek(img.tell() + 1)
        except EOFError:
            pass  # End of sequence
    
    print(f"  Loaded {len(images)} images from stack")
    print(f"  Image data type: {images[0].dtype}, range: {images[0].min()}-{images[0].max()}")
    
    # Process with existing complex wavelet EDF function
    fused_image, min_z_used, max_z_used = complex_wavelet_edf(images, wavelet, levels, bit_depth, top_n)
    
    # Save result as TIFF with appropriate bit depth
    if bit_depth == 16:
        result_img = Image.fromarray(fused_image.astype(np.uint16), mode='I;16')
    else:
        result_img = Image.fromarray(fused_image.astype(np.uint8), mode='L')
    
    result_img.save(output_path)
    print(f"  Fused {bit_depth}-bit image saved")
    print(f"  Z-range used: {min_z_used} to {max_z_used} (out of {len(images)-1})")
    
    return min_z_used, max_z_used

def process_folder(input_folder, output_folder, wavelet='db3', levels=3, bit_depth=16, top_n=1):
    """
    Batch process all TIFF stack files in a folder.
    Args:
        input_folder (str): Path to folder containing TIFF stack files.
        output_folder (str): Path to folder for output fused images.
        wavelet (str): Wavelet type (default 'db3').
        levels (int): Number of decomposition levels.
        bit_depth (int): Output bit depth (8 or 16).
        top_n (int): Number of top layers to average.
    """
    input_path = os.path.abspath(input_folder)
    output_path = os.path.abspath(output_folder)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Supported TIFF extensions
    tiff_extensions = {'.tif', '.tiff'}
    
    # Find all TIFF files
    tiff_files = [f for f in os.listdir(input_path) 
                  if os.path.isfile(os.path.join(input_path, f)) and 
                  os.path.splitext(f.lower())[1] in tiff_extensions]
    
    if not tiff_files:
        print(f"No TIFF files found in {input_folder}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    print(f"Input folder: {input_path}")
    print(f"Output folder: {output_path}")
    print(f"Settings: wavelet={wavelet}, levels={levels}, bit_depth={bit_depth}, top_n={top_n}")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    # CSV file to record z-range data for all timepoints
    csv_path = os.path.join(output_path, "edf_z_ranges.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'timepoint', 'min_z_used', 'max_z_used', 'total_frames', 'z_range_used'])
        
        for i, tiff_file in enumerate(sorted(tiff_files), 1):
            input_file_path = os.path.join(input_path, tiff_file)
            
            # Create output filename (preserve name, ensure .tif extension)
            base_name = os.path.splitext(tiff_file)[0]
            output_filename = f"{base_name}_edf.tif"
            output_file_path = os.path.join(output_path, output_filename)
            
            print(f"[{i}/{len(tiff_files)}] Processing: {tiff_file}")
            
            try:
                # Check if this is actually a TIFF stack
                frame_count = 0
                with Image.open(input_file_path) as img:
                    try:
                        while True:
                            frame_count += 1
                            img.seek(img.tell() + 1)
                    except EOFError:
                        pass
                    
                    if frame_count < 2:
                        print(f"  Warning: {tiff_file} contains only {frame_count} frame(s), skipping")
                        continue
                    
                    print(f"  Found {frame_count} frames")
                
                # Process the TIFF stack
                min_z_used, max_z_used = complex_wavelet_edf_from_file(
                    input_file_path, output_file_path, wavelet, levels, bit_depth, top_n
                )
                
                # Extract timepoint from filename (assume format contains timepoint info)
                # You can modify this logic based on your specific filename format
                timepoint = base_name  # Use full base name as timepoint identifier
                
                # Write to CSV
                z_range_used = max_z_used - min_z_used + 1
                writer.writerow([
                    tiff_file, 
                    timepoint, 
                    min_z_used, 
                    max_z_used, 
                    frame_count, 
                    z_range_used
                ])
                
                print(f"  Success: {output_filename}")
                successful += 1
                
            except Exception as e:
                print(f"  Error processing {tiff_file}: {e}")
                failed += 1
            
            print()  # Add blank line between files
    
    print("=" * 60)
    print(f"Batch processing complete:")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output folder: {output_path}")
    print(f"  Z-range summary saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Complex Wavelet Extended Depth of Field for TIFF stacks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process single TIFF stack
  python complexedf.py input_stack.tif output_edf.tif
  
  # Process with custom settings
  python complexedf.py input.tif output.tif --wavelet db5 --levels 8 --bit-depth 8
  
  # Batch process all TIFF stacks in a folder
  python complexedf.py --input-folder ./stacks --output-folder ./fused
  
  # Batch process with top-3 averaging
  python complexedf.py --input-folder ./input --output-folder ./output --top-n 3

WAVELET OPTIONS:
  Common wavelets: db1, db2, db3, db4, db5, haar, bior2.2, bior4.4, coif2, coif4
  
BATCH PROCESSING:
  - Processes all .tif and .tiff files in input folder
  - Skips files with less than 2 frames
  - Output files named as: {original_name}_edf.tif
  - Creates output folder if it doesn't exist
  - Generates edf_z_ranges.csv with z-range summary for all timepoints

Z-RANGE TRACKING:
  - Records minimum and maximum z-indices used across entire image
  - Single CSV file for all timepoints: edf_z_ranges.csv
  - Columns: filename, timepoint, min_z_used, max_z_used, total_frames, z_range_used
        """
    )
    
    # Make input and folder mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('input', nargs='?', help='Input TIFF stack file path (for single file mode)')
    input_group.add_argument('--input-folder', help='Path to folder containing TIFF stack files (for batch mode)')
    
    # Output arguments
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('output', nargs='?', help='Output fused TIFF file path (for single file mode)')
    output_group.add_argument('--output-folder', help='Path to folder for output fused images (for batch mode)')
    
    # Processing parameters
    parser.add_argument('-w', '--wavelet', default='db3', 
                       help='Wavelet type (default: db3 - 6 coefficients)')
    parser.add_argument('-l', '--levels', type=int, default=11,
                       help='Number of decomposition levels (default: 11)')
    parser.add_argument('-b', '--bit-depth', type=int, choices=[8, 16], default=16,
                       help='Output bit depth: 8 or 16 (default: 16)')
    parser.add_argument('-n', '--top-n', type=int, default=1,
                       help='Number of top layers to average based on wavelet metric (default: 1)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_folder:
        # Batch mode
        if not args.output_folder:
            parser.error("--output-folder is required when using --input-folder")
        
        if not os.path.isdir(args.input_folder):
            parser.error(f"Input folder '{args.input_folder}' does not exist")
        
        # Process folder
        process_folder(args.input_folder, args.output_folder, args.wavelet, args.levels, args.bit_depth, args.top_n)
    
    else:
        # Single file mode
        if not args.output:
            parser.error("output path is required when processing a single file")
        
        if not os.path.exists(args.input):
            parser.error(f"Input file '{args.input}' not found")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process single file
        try:
            min_z, max_z = complex_wavelet_edf_from_file(args.input, args.output, args.wavelet, args.levels, args.bit_depth, args.top_n)
            print(f"Z-range used: {min_z} to {max_z}")
        except Exception as e:
            print(f"Error processing images: {e}")

# Example usage
if __name__ == "__main__":
    main()