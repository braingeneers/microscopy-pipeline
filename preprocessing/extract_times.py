import os
import re
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime

from pipeline_log import get_logger
log = get_logger("extract_times")


# Time parsing

def parse_time_offset(time_str):
    """Parse time offset from either seconds or HH:MM:SS.sss format."""
    try:
        return float(time_str)
    except ValueError:
        pass

    try:
        parts = time_str.split(':')
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            raise ValueError("Too many time components")
    except (ValueError, IndexError):
        raise ValueError(
            f"Invalid time format: '{time_str}'. "
            "Use seconds (e.g. '1800') or HH:MM:SS.sss (e.g. '00:30:00')"
        )


# Timestamp extraction — individual PNG/TIFF files


def extract_from_files(directory, time_offset=0):
    """
    Extract creation times from individually named image files
    matching the pattern <N>_zs*.
    Returns a list of row dicts.
    """
    pattern = re.compile(r'^(\d+)_zs.*')
    files = [f for f in os.listdir(directory) if pattern.match(f)]

    n_to_files = {}
    for fname in files:
        match = pattern.match(fname)
        if match:
            n = int(match.group(1))
            n_to_files.setdefault(n, []).append(fname)

    rows = []
    first_time = None

    for n in sorted(n_to_files.keys()):
        first_file = sorted(n_to_files[n])[0]
        file_path = Path(directory) / first_file
        try:
            ctime = os.path.getctime(file_path)
            actual_time = datetime.fromtimestamp(ctime)

            if first_time is None:
                first_time = ctime
                elapsed_seconds = 0.0
            else:
                elapsed_seconds = ctime - first_time

            adjusted = elapsed_seconds + time_offset
            elapsed_formatted = _format_elapsed(adjusted)

            rows.append({
                'N': n,
                'filename': first_file,
                'actual_time': actual_time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_time': elapsed_formatted,
                'elapsed_seconds': round(adjusted, 3),
                'raw_elapsed_seconds': round(elapsed_seconds, 3),
                'time_offset': time_offset,
                'raw_timestamp': ctime,
            })

            log.info("N=%d file=%s actual_time=%s elapsed=%s",
                     n, first_file, actual_time.strftime('%Y-%m-%d %H:%M:%S'),
                     elapsed_formatted)

        except Exception as e:
            log.warning("Could not get creation time for %s: %s", file_path, e)
            continue

    return rows


# Timestamp extraction — TIFF stacks


def extract_from_tiff_stacks(directory, time_offset=0):
    """
    Extract timestamps from TIFF stack files (stack_N.tiff / stack_N.tif).
    Reads acquisition timestamps from OME/ImageJ metadata where available;
    falls back to filesystem creation time.
    Returns a list of row dicts.
    """
    try:
        from PIL import Image
        import json
    except ImportError:
        log.error("Pillow is required for TIFF stack mode: pip install Pillow")
        sys.exit(1)

    tiff_pattern = re.compile(r'^stack_(\d+)\.tiff?$', re.IGNORECASE)
    stack_files = [f for f in os.listdir(directory) if tiff_pattern.match(f)]

    if not stack_files:
        log.warning("No stack_N.tiff files found in: %s", directory)
        return []

    n_to_file = {}
    for fname in stack_files:
        match = tiff_pattern.match(fname)
        if match:
            n_to_file[int(match.group(1))] = fname

    rows = []
    first_time = None

    for n in sorted(n_to_file.keys()):
        fname = n_to_file[n]
        file_path = Path(directory) / fname
        acquisition_time = None

        try:
            with Image.open(file_path) as img:
                # Try OME-XML in ImageDescription tag (tag 270)
                tag_data = img.tag_v2 if hasattr(img, 'tag_v2') else {}
                image_description = tag_data.get(270, '')
                if isinstance(image_description, bytes):
                    image_description = image_description.decode('utf-8', errors='ignore')

                # Look for AcquisitionDate in OME-XML
                ome_match = re.search(
                    r'AcquisitionDate[^>]*>([^<]+)<', image_description)
                if ome_match:
                    try:
                        acquisition_time = datetime.fromisoformat(
                            ome_match.group(1).strip().replace('Z', '+00:00')
                        ).timestamp()
                        log.info("  N=%d: OME AcquisitionDate found", n)
                    except ValueError:
                        pass

                # Fallback: ImageJ DeltaT in metadata (relative, not absolute)
                if acquisition_time is None:
                    ij_match = re.search(r'DeltaT=([0-9.]+)', image_description)
                    if ij_match:
                        log.info("  N=%d: ImageJ DeltaT found (relative only, using as elapsed)", n)

        except Exception as e:
            log.warning("Could not read TIFF metadata for %s: %s", fname, e)

        # Fall back to filesystem ctime if no metadata timestamp found
        if acquisition_time is None:
            acquisition_time = os.path.getctime(file_path)
            log.info("  N=%d: using filesystem ctime (no metadata timestamp)", n)

        actual_time = datetime.fromtimestamp(acquisition_time)

        if first_time is None:
            first_time = acquisition_time
            elapsed_seconds = 0.0
        else:
            elapsed_seconds = acquisition_time - first_time

        adjusted = elapsed_seconds + time_offset
        elapsed_formatted = _format_elapsed(adjusted)

        rows.append({
            'N': n,
            'filename': fname,
            'actual_time': actual_time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': elapsed_formatted,
            'elapsed_seconds': round(adjusted, 3),
            'raw_elapsed_seconds': round(elapsed_seconds, 3),
            'time_offset': time_offset,
            'raw_timestamp': acquisition_time,
        })

        log.info("N=%d file=%s actual_time=%s elapsed=%s",
                 n, fname, actual_time.strftime('%Y-%m-%d %H:%M:%S'),
                 elapsed_formatted)

    return rows


# Helpers
def _format_elapsed(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def write_csv(rows, output_path):
    fieldnames = [
        'N', 'filename', 'actual_time', 'elapsed_time',
        'elapsed_seconds', 'raw_elapsed_seconds', 'time_offset', 'raw_timestamp'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV saved | path=%s rows=%d", output_path, len(rows))


# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract creation times from image files and write elapsed times to CSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog= """
EXAMPLES
  # Individual image files in current directory
  python extract_times.py
 
  # Specific directory, custom output location
  python extract_times.py --directory /path/to/images --output /path/to/times.csv
 
  # TIFF stack mode
  python extract_times.py --directory ./stacks --tiff-stacks
 
  # Add 30-minute offset using HH:MM:SS
  python extract_times.py --time-offset 00:30:00
 
  # Subtract 10 minutes using seconds
  python extract_times.py --time-offset -600
 
TIME OFFSET FORMATS
  Seconds: 1800, -600, 45.5
  MM:SS:   30:00, -10:30
  HH:MM:SS: 02:15:30, -00:10:30, 01:00:45.500
 
OUTPUT COLUMNS
  N                   - Timepoint index from filename
  filename            - Source filename
  actual_time         - File creation timestamp (YYYY-MM-DD HH:MM:SS)
  elapsed_time        - Adjusted elapsed time (HH:MM:SS.sss)
  elapsed_seconds     - Adjusted elapsed time in decimal seconds
  raw_elapsed_seconds - Elapsed time before offset
  time_offset         - Applied offset in seconds
  raw_timestamp       - Unix timestamp
        """
    )

    parser.add_argument('--directory', '-d',
                        default='.',
                        help='Directory containing image files (default: current directory)')
    parser.add_argument('--output', '-o',
                        default=None,
                        help='Output CSV file path (default: <directory>/file_timestamps.csv)')
    parser.add_argument('--time-offset', '-t',
                        type=str,
                        default='0',
                        help='Time offset to add to elapsed times (default: 0)')
    parser.add_argument('--tiff-stacks',
                        action='store_true',
                        help='Read from TIFF stack files (stack_N.tiff) instead of individual images')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        log.error("Directory not found: %s", args.directory)
        sys.exit(1)

    try:
        time_offset = parse_time_offset(args.time_offset)
    except ValueError as e:
        log.error("%s", e)
        sys.exit(1)

    output_path = args.output or str(Path(args.directory) / 'file_timestamps.csv')

    log.info("EXTRACT START | directory=%s mode=%s output=%s offset=%s",
             os.path.abspath(args.directory),
             "tiff-stacks" if args.tiff_stacks else "individual-files",
             output_path,
             args.time_offset)

    if args.tiff_stacks:
        rows = extract_from_tiff_stacks(args.directory, time_offset)
    else:
        rows = extract_from_files(args.directory, time_offset)

    if not rows:
        log.warning("No data extracted — no output written")
        sys.exit(1)

    if time_offset != 0:
        log.info("Time offset applied: %+.3f seconds (%s)", time_offset,
                 _format_elapsed(abs(time_offset)))

    write_csv(rows, output_path)

    log.info("EXTRACT END | status=complete rows=%d output=%s", len(rows), output_path)