import os
import re
import csv
import sys
import argparse
from glob import glob
from PIL import Image
import numpy as np

def sum_image_brightness(image_path, brightfield=False):
    """Optimized brightness summation using vectorized operations with 16-bit support"""
    with Image.open(image_path) as img:
        # Convert to numpy array with optimal dtype
        arr = np.asarray(img, dtype=np.float64)
        
        if brightfield:
            # For brightfield: invert so black (0) becomes highest value, white becomes 0
            # Get max possible value based on image mode
            if img.mode == 'L':  # 8-bit grayscale
                max_val = 255
            elif img.mode in ['I;16', 'I;16B', 'I;16L']:  # 16-bit grayscale modes
                max_val = 65535
                print(f"  Detected 16-bit image mode '{img.mode}', using max_val=65535")
            elif img.mode == 'I':  # 32-bit integer (but check if it's actually 16-bit data)
                # Check if the actual data range suggests 16-bit
                actual_max = np.max(arr)
                if actual_max <= 65535:
                    max_val = 65535
                    print(f"  Mode 'I' with max value {actual_max:.0f}, treating as 16-bit (max_val=65535)")
                else:
                    max_val = 2**32 - 1
                    print(f"  Mode 'I' with max value {actual_max:.0f}, treating as 32-bit")
            elif img.mode == 'F':  # 32-bit float
                max_val = 1.0
            elif img.mode in ['RGB', 'RGBA']:  # Color images (8-bit per channel)
                max_val = 255
            else:
                # Default to 255 for unknown modes
                max_val = 255
                print(f"  Warning: Unknown image mode '{img.mode}', using max_val=255")
            
            # Invert: black pixels (0) become max_val, white pixels (max_val) become 0
            inverted_arr = max_val - arr
            return np.sum(inverted_arr)
        else:
            # Standard fluorescence: sum pixel values directly
            return np.sum(arr)

def load_timestamps(timestamp_file):
    """Load timestamps from CSV file, indexed by N column"""
    timestamps = {}
    try:
        with open(timestamp_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if required columns exist
            if 'N' not in reader.fieldnames or 'actual_time' not in reader.fieldnames:
                print(f"Error: Timestamp file must contain 'N' and 'actual_time' columns")
                print(f"Found columns: {reader.fieldnames}")
                return None
            
            for row in reader:
                try:
                    n_value = int(row['N'])
                    actual_time = row['actual_time']
                    timestamps[n_value] = actual_time
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row in timestamp file: {row}")
                    continue
        
        print(f"Loaded {len(timestamps)} timestamps from {timestamp_file}")
        return timestamps
    
    except FileNotFoundError:
        print(f"Error: Timestamp file '{timestamp_file}' not found")
        return None
    except Exception as e:
        print(f"Error reading timestamp file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Sum fluorescence brightness in images matching pattern *_N_*.tif(f). Supports 8-bit and 16-bit images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Basic usage - sum brightness for all matching images
  python sumfluorescence.py /path/to/images
  
  # Brightfield mode - invert pixel values (black = high, white = 0)
  python sumfluorescence.py /path/to/images --brightfield
  
  # Include timestamps from external CSV file
  python sumfluorescence.py /path/to/images --timestamps timestamps.csv
  
  # Brightfield with timestamps (works with 16-bit images)
  python sumfluorescence.py /path/to/images --brightfield --timestamps timing.csv

BRIGHTFIELD MODE:
  The --brightfield flag inverts pixel values for optical density measurements:
  
  8-bit images (mode 'L'):
  - Black pixels (value 0) become 255 (high optical density)
  - White pixels (value 255) become 0 (no optical density)
  
  16-bit images (modes 'I;16', 'I;16B', 'I;16L'):
  - Black pixels (value 0) become 65535 (high optical density)
  - White pixels (value 65535) become 0 (no optical density)
  
  Mode 'I' images:
  - Automatically detects if data is 16-bit range (max ≤ 65535) or true 32-bit
  - Uses appropriate max value for inversion
  
  Useful for analyzing organoid growth, cell density, optical density measurements

16-BIT IMAGE SUPPORT:
  - Automatically detects 16-bit images: I;16, I;16B, I;16L modes
  - Uses proper max value of 65535 for brightfield inversion
  - Handles mixed 8-bit and 16-bit images in same folder
  - Preserves full dynamic range for accurate measurements

TIMESTAMP CSV FORMAT:
  The timestamp CSV file must contain columns 'N' and 'actual_time':
  
  N,actual_time,other_columns...
  1,2024-08-20 10:30:15,data...
  2,2024-08-20 10:31:20,data...
  5,2024-08-20 10:35:42,data...
  
  - 'N' column: Integer values matching the N in image filenames
  - 'actual_time' column: Timestamp values (any format)
  - Missing N values will show as "No timestamp" in output

SUPPORTED IMAGE MODES:
  - L: 8-bit grayscale (max=255)
  - I;16, I;16B, I;16L: 16-bit grayscale (max=65535)
  - I: 32-bit integer (auto-detects 16-bit vs 32-bit data)
  - F: 32-bit float (max=1.0)
  - RGB, RGBA: Color images (max=255 per channel)
        """
    )
    
    parser.add_argument('folder', help='Folder containing image files to process (8-bit and 16-bit supported)')
    parser.add_argument('--timestamps', help='CSV file containing timestamps indexed by N column')
    parser.add_argument('--brightfield', action='store_true', 
                       help='Invert pixel values for brightfield analysis (8-bit: black=255, white=0; 16-bit: black=65535, white=0)')
    
    args = parser.parse_args()
    
    folder = args.folder
    
    # Validate folder
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist")
        sys.exit(1)
    
    # Load timestamps if provided
    timestamps = None
    if args.timestamps:
        timestamps = load_timestamps(args.timestamps)
        if timestamps is None:
            sys.exit(1)
    
    # Process images
    pattern = re.compile(r'.*_(\d+)_.*\.tiff?$', re.IGNORECASE)
    image_files = glob(os.path.join(folder, '*_*_*.tif')) + glob(os.path.join(folder, '*_*_*.tiff'))
    
    print(f"Found {len(image_files)} image files in folder: {folder}")
    if args.brightfield:
        print(f"Brightfield mode: Inverting pixel values (8-bit: black=255, white=0; 16-bit: black=65535, white=0)")
    
    # First pass: extract indices and create (index, filepath) pairs
    indexed_files = []
    for img_file in image_files:
        filename = os.path.basename(img_file)
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            indexed_files.append((index, img_file))
    
    # Sort by numerical index, not filename
    indexed_files.sort(key=lambda x: x[0])
    
    print(f"Pattern matched: {len(indexed_files)} files")
    if timestamps:
        print(f"Timestamp data available for correlation")
    print(f"Processing in numerical order of N...")
    print("-" * 60)
    
    results = []
    processed_count = 0
    skipped_count = 0
    bit_depth_counts = {'8-bit': 0, '16-bit': 0, '32-bit': 0, 'other': 0}

    for i, (index, img_file) in enumerate(indexed_files, 1):
        filename = os.path.basename(img_file)
        print(f"[{i}/{len(indexed_files)}] Processing: {filename} (N={index})")
        
        try:
            # Get file size for reporting
            file_size = os.path.getsize(img_file)
            file_size_mb = file_size / (1024 * 1024)
            
            # Open and check image properties
            with Image.open(img_file) as img:
                img_shape = img.size  # (width, height)
                img_mode = img.mode
                
                # Determine bit depth for statistics
                if img_mode == 'L':
                    bit_depth = '8-bit'
                elif img_mode in ['I;16', 'I;16B', 'I;16L']:
                    bit_depth = '16-bit'
                elif img_mode == 'I':
                    # Check actual data range for mode 'I'
                    sample_arr = np.asarray(img, dtype=np.float64)
                    if np.max(sample_arr) <= 65535:
                        bit_depth = '16-bit'
                    else:
                        bit_depth = '32-bit'
                else:
                    bit_depth = 'other'
                
                bit_depth_counts[bit_depth] += 1
            
            print(f"  Size: {img_shape[0]}x{img_shape[1]}, Mode: {img_mode} ({bit_depth}), File: {file_size_mb:.1f} MB")
            
            # Optimized brightness calculation with brightfield option
            brightness_sum = sum_image_brightness(img_file, brightfield=args.brightfield)
            
            # Format the brightness sum for readability
            if brightness_sum > 1e9:
                brightness_str = f"{brightness_sum:.2e}"
            else:
                brightness_str = f"{int(brightness_sum):,}"
            
            # Get timestamp if available
            timestamp = "No timestamp"
            if timestamps and index in timestamps:
                timestamp = timestamps[index]
            
            # Label the output based on mode
            measurement_type = "Optical density sum" if args.brightfield else "Brightness sum"
            print(f"  Index: {index}, {measurement_type}: {brightness_str}")
            if timestamps:
                print(f"  Timestamp: {timestamp}")
            
            # Store results (with or without timestamp)
            if timestamps:
                results.append((filename, index, brightness_sum, timestamp))
            else:
                results.append((filename, index, brightness_sum))
            processed_count += 1
            
        except Exception as e:
            print(f"  Error processing image: {e}")
            skipped_count += 1
        
        print()  # Add blank line between files

    # Report files that didn't match the pattern
    unmatched_files = [f for f in image_files if not any(f == indexed_file[1] for indexed_file in indexed_files)]
    if unmatched_files:
        print("Files that didn't match pattern:")
        for unmatched in unmatched_files:
            print(f"  Skipped: {os.path.basename(unmatched)} (no pattern match)")

    print("=" * 60)
    print(f"Processing Summary:")
    print(f"  Total files found: {len(image_files)}")
    print(f"  Pattern matched: {len(indexed_files)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Failed to process: {skipped_count}")
    print(f"  No pattern match: {len(unmatched_files)}")
    
    # Report bit depth statistics
    print(f"\nBit Depth Distribution:")
    for depth, count in bit_depth_counts.items():
        if count > 0:
            print(f"  {depth}: {count} images")
    
    if results:
        # Results are already in numerical order since we processed them that way
        # Calculate statistics
        brightness_values = [r[2] for r in results]
        min_brightness = min(brightness_values)
        max_brightness = max(brightness_values)
        avg_brightness = np.mean(brightness_values)
        
        measurement_label = "Optical Density" if args.brightfield else "Brightness"
        print(f"\n{measurement_label} Statistics:")
        print(f"  Minimum: {min_brightness:,.0f}")
        print(f"  Maximum: {max_brightness:,.0f}")
        print(f"  Average: {avg_brightness:,.0f}")
        
        # Check timestamp coverage if timestamps were provided
        if timestamps:
            matched_timestamps = sum(1 for r in results if len(r) > 3 and r[3] != "No timestamp")
            print(f"\nTimestamp Coverage:")
            print(f"  Images with timestamps: {matched_timestamps}/{processed_count}")
            print(f"  Images without timestamps: {processed_count - matched_timestamps}")
        
        # Save results to CSV (already in numerical order)
        csv_filename = 'sum_optical_density.csv' if args.brightfield else 'sum_brightness.csv'
        csv_path = os.path.join(folder, csv_filename)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header based on whether timestamps are included
            header_label = 'sum_optical_density' if args.brightfield else 'sum_brightness'
            if timestamps:
                writer.writerow(['filename', 'index', header_label, 'actual_time'])
            else:
                writer.writerow(['filename', 'index', header_label])
            
            writer.writerows(results)

        print(f"\nResults saved to: {csv_path}")
        print(f"Files processed in order: N={indexed_files[0][0]} to N={indexed_files[-1][0]}")
        
        if args.brightfield:
            print(f"Brightfield mode: Higher values indicate darker regions (higher optical density)")
            if bit_depth_counts['16-bit'] > 0:
                print(f"16-bit images: Used max_val=65535 for inversion")
            if bit_depth_counts['8-bit'] > 0:
                print(f"8-bit images: Used max_val=255 for inversion")
        
        if timestamps:
            print(f"Output includes timestamp correlation from: {args.timestamps}")
    else:
        print("\nNo valid images were processed!")

if __name__ == "__main__":
    main()