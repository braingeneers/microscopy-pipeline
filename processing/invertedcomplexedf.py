import numpy as np
from scipy.signal import convolve2d
import pywt
from PIL import Image
import os
import argparse

def complex_wavelet_edf(images, wavelet='db3', levels=3, bit_depth=16, top_n=1):
    """
    Extended Depth of Field using Complex Wavelets - INVERTED to select least in focus.
    Args:
        images (list of np.ndarray): List of grayscale images (2D arrays) to fuse.
        wavelet (str): Wavelet type (default 'db3' - 6 coefficients).
        levels (int): Number of decomposition levels.
        bit_depth (int): Output bit depth (8 or 16).
        top_n (int): Number of least focused layers to average based on wavelet metric (default: 1).
    Returns:
        np.ndarray: Fused image.
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

    # Decompose all images
    decomposed = [complex_wavelet_decompose(img, wavelet, levels) for img in float_images]

    # Enhanced fusion rule with bottom-N selection (least in focus)
    fused_coeffs = []
    for level_idx, coeffs_per_level in enumerate(zip(*decomposed)):
        if isinstance(coeffs_per_level[0], tuple):
            complex_c_list, cD_list = zip(*coeffs_per_level)
            
            # Calculate magnitude maps without smoothing
            mags = [np.abs(c) for c in complex_c_list]
            
            # Stack magnitudes for easy sorting
            mag_stack = np.stack(mags, axis=0)  # Shape: (num_images, height, width)
            
            # Get indices of bottom N magnitudes at each pixel (LEAST in focus)
            top_n_clamped = min(top_n, len(complex_c_list))  # Don't exceed number of images
            # Use argpartition with positive values to get SMALLEST magnitudes
            bottom_indices = np.argpartition(mag_stack, top_n_clamped-1, axis=0)[:top_n_clamped]
            
            # Create coefficient stacks for indexing
            complex_c_stack = np.stack(complex_c_list, axis=0)
            cD_stack = np.stack(cD_list, axis=0)
            
            # Initialize fused coefficients
            fused_complex_c = np.zeros_like(complex_c_list[0])
            fused_cD = np.zeros_like(cD_list[0])
            
            # Average bottom N coefficients at each pixel (least in focus)
            for i in range(top_n_clamped):
                # Get the i-th worst coefficient at each pixel
                idx_map = bottom_indices[i]  # Shape: (height, width)
                
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
            # Approximation coefficients: bottom-N simple average based on coefficient magnitude
            mags = [np.abs(coeff) for coeff in coeffs_per_level]
            
            # Stack magnitudes and coefficients for easy sorting
            mag_stack = np.stack(mags, axis=0)
            coeff_stack = np.stack(coeffs_per_level, axis=0)
            
            # Get indices of bottom N magnitudes at each pixel (LEAST in focus)
            top_n_clamped = min(top_n, len(coeffs_per_level))
            # Use argpartition with positive values to get SMALLEST magnitudes
            bottom_indices = np.argpartition(mag_stack, top_n_clamped-1, axis=0)[:top_n_clamped]
            
            # Initialize fused approximation
            fused_approx = np.zeros_like(coeffs_per_level[0])
            
            # Simple average of bottom N coefficients (least in focus)
            for i in range(top_n_clamped):
                idx_map = bottom_indices[i]
                selected_coeff = np.choose(idx_map, coeff_stack)
                fused_approx += selected_coeff
            
            # Average the selected coefficients
            fused_approx /= top_n_clamped
            
            fused_coeffs.append(fused_approx)

    # Reconstruct fused image
    fused_img = complex_wavelet_reconstruct(fused_coeffs, wavelet)
    
    # Clip to appropriate range and convert to target bit depth
    return np.clip(fused_img, 0, max_val)

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
        None: Saves fused image to output_path.
    """
    
    # Read TIFF stack
    images = []
    original_mode = None
    input_bit_depth = 8  # Default assumption
    
    with Image.open(input_path) as img:
        original_mode = img.mode
        print(f"Input image mode: {original_mode}")
        
        # Determine input bit depth
        if img.mode == 'I;16':
            input_bit_depth = 16
        elif img.mode == 'L':
            input_bit_depth = 8
            
        print(f"Input bit depth: {input_bit_depth}, Output bit depth: {bit_depth}")
        
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
    
    print(f"Loaded {len(images)} images from {input_path}")
    print(f"Image data type: {images[0].dtype}, range: {images[0].min()}-{images[0].max()}")
    
    # Process with existing complex wavelet EDF function
    fused_image = complex_wavelet_edf(images, wavelet, levels, bit_depth, top_n)
    
    # Save result as TIFF with appropriate bit depth
    if bit_depth == 16:
        result_img = Image.fromarray(fused_image.astype(np.uint16), mode='I;16')
    else:
        result_img = Image.fromarray(fused_image.astype(np.uint8), mode='L')
    
    result_img.save(output_path)
    print(f"Fused {bit_depth}-bit image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Inverted Complex Wavelet Extended Depth of Field for TIFF stacks')
    parser.add_argument('input', help='Input TIFF stack file path')
    parser.add_argument('output', help='Output fused TIFF file path')
    parser.add_argument('-w', '--wavelet', default='db3', 
                       help='Wavelet type (default: db3 - 6 coefficients)')
    parser.add_argument('-l', '--levels', type=int, default=11,
                       help='Number of decomposition levels (default: 11)')
    parser.add_argument('-b', '--bit-depth', type=int, choices=[8, 16], default=16,
                       help='Output bit depth: 8 or 16 (default: 16)')
    parser.add_argument('-n', '--top-n', type=int, default=1,
                       help='Number of least focused layers to average based on wavelet metric (default: 1)')

    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the images
    try:
        complex_wavelet_edf_from_file(args.input, args.output, args.wavelet, args.levels, args.bit_depth, args.top_n)
    except Exception as e:
        print(f"Error processing images: {e}")

# Example usage
if __name__ == "__main__":
    main()