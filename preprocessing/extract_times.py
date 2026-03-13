import os
import re
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime

def extract_creation_times(directory, time_offset=0):
    pattern = re.compile(r'^(\d+)_zs.*')
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    n_to_files = {}

    # Group files by N
    for fname in files:
        match = pattern.match(fname)
        if match:
            n = int(match.group(1))
            n_to_files.setdefault(n, []).append(fname)

    # Prepare CSV data
    csv_data = []
    first_time = None
    
    # Iterate in numerical order of N
    for n in sorted(n_to_files.keys()):
        # Find the first occurring file for this N
        first_file = sorted(n_to_files[n])[0]
        file_path = Path(directory) / first_file
        # Get creation time (platform dependent)
        try:
            ctime = os.path.getctime(file_path)
            actual_time = datetime.fromtimestamp(ctime)
            
            # Store the first time as reference
            if first_time is None:
                first_time = ctime
                elapsed_seconds = 0.0
            else:
                elapsed_seconds = ctime - first_time
            
            # Add the time offset to elapsed seconds
            adjusted_elapsed_seconds = elapsed_seconds + time_offset
            
            # Format elapsed time in a readable format
            hours = int(adjusted_elapsed_seconds // 3600)
            minutes = int((adjusted_elapsed_seconds % 3600) // 60)
            seconds = adjusted_elapsed_seconds % 60
            elapsed_formatted = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
            
            # Add row to CSV data
            csv_data.append({
                'N': n,
                'filename': first_file,
                'actual_time': actual_time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_time': elapsed_formatted,
                'elapsed_seconds': round(adjusted_elapsed_seconds, 3),
                'raw_elapsed_seconds': round(elapsed_seconds, 3),
                'time_offset': time_offset,
                'raw_timestamp': ctime
            })
            
            print(f"N={n}, file={first_file}, actual_time={actual_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                  f"elapsed_time={elapsed_formatted}, raw_timestamp={ctime}")
                  
        except Exception as e:
            print(f"Error getting creation time for {file_path}: {e}")
            continue
    
    # Write to CSV file in the same directory as the images
    csv_filename = Path(directory) / 'file_timestamps.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['N', 'filename', 'actual_time', 'elapsed_time', 'elapsed_seconds', 'raw_elapsed_seconds', 'time_offset', 'raw_timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    if time_offset != 0:
        offset_hours = int(abs(time_offset) // 3600)
        offset_minutes = int((abs(time_offset) % 3600) // 60)
        offset_secs = abs(time_offset) % 60
        offset_sign = "+" if time_offset >= 0 else "-"
        print(f"\nTime offset applied: {offset_sign}{offset_hours:02d}:{offset_minutes:02d}:{offset_secs:06.3f} ({time_offset:+.3f} seconds)")
    
    print(f"Data saved to {csv_filename}")
    print(f"Total files processed: {len(csv_data)}")

def parse_time_offset(time_str):
    """Parse time offset from either seconds or HH:MM:SS.sss format"""
    try:
        # First try parsing as float (seconds)
        return float(time_str)
    except ValueError:
        pass
    
    # Try parsing HH:MM:SS.sss format
    try:
        # Handle formats like HH:MM:SS.sss, HH:MM:SS, MM:SS, etc.
        parts = time_str.split(':')
        
        if len(parts) == 1:
            # Just seconds
            return float(parts[0])
        elif len(parts) == 2:
            # MM:SS or MM:SS.sss
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            # HH:MM:SS or HH:MM:SS.sss
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            raise ValueError("Too many time components")
            
    except (ValueError, IndexError):
        raise ValueError(f"Invalid time format: '{time_str}'. Use seconds (e.g., '1800') or HH:MM:SS.sss format (e.g., '00:30:00')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract creation times from numbered image files and calculate elapsed times.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process files in current directory
  python extract_times.py
  
  # Process files in specific directory
  python extract_times.py --directory /path/to/images
  
  # Add 30 minutes using seconds
  python extract_times.py --time-offset 1800
  
  # Add 30 minutes using HH:MM:SS format
  python extract_times.py --time-offset 00:30:00
  
  # Subtract 10 minutes and 30 seconds
  python extract_times.py --time-offset -00:10:30
  
  # Add 2 hours, 15 minutes, 30 seconds
  python extract_times.py --time-offset 02:15:30 --directory ./data
  
  # Add 45.5 seconds using decimal format
  python extract_times.py --time-offset 00:00:45.500

TIME OFFSET FORMATS:
  - Seconds: 1800, -600, 45.5
  - MM:SS: 30:00, -10:30, 01:45.5
  - HH:MM:SS: 02:15:30, -00:10:30, 01:00:45.500
  - Negative values subtract time from elapsed counter
  - Positive values add time to elapsed counter

COMMON CONVERSIONS:
  - 1 hour = 3600 seconds = 01:00:00
  - 30 minutes = 1800 seconds = 00:30:00
  - 10 minutes = 600 seconds = 00:10:00
  - 1 minute = 60 seconds = 00:01:00

OUTPUT:
  Creates file_timestamps.csv in the target directory with columns:
  - N: File number extracted from filename
  - filename: Original filename
  - actual_time: File creation timestamp (YYYY-MM-DD HH:MM:SS)
  - elapsed_time: Adjusted elapsed time (HH:MM:SS.sss)
  - elapsed_seconds: Adjusted elapsed time in decimal seconds
  - raw_elapsed_seconds: Original elapsed time before offset
  - time_offset: Applied time offset in seconds
  - raw_timestamp: Unix timestamp

FILE PATTERN:
  Processes files matching pattern: {number}_zs*
  Examples: 001_zs_image.tiff, 125_zs_data.jpg, 999_zs_experiment.png
        """
    )
    
    parser.add_argument('--directory', '-d', 
                       default='.', 
                       help='Directory containing image files (default: current directory)')
    
    parser.add_argument('--time-offset', '-t',
                       type=str,
                       default='0',
                       help='Time offset to add to elapsed times. '
                            'Formats: seconds (1800), MM:SS (30:00), HH:MM:SS (02:15:30), '
                            'or with decimals (00:30:15.500). Can be negative.')
    
    args = parser.parse_args()
    
    # Validate directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)
    
    # Parse time offset
    try:
        time_offset = parse_time_offset(args.time_offset)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Processing files in directory: {os.path.abspath(args.directory)}")
    extract_creation_times(args.directory, time_offset)