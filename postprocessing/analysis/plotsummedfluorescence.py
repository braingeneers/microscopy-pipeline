import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

def parse_timestamp(timestamp_str):
    """
    Parse timestamp string into datetime object.
    Supports various formats commonly used in scientific data.
    """
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%H:%M:%S',
        '%H:%M'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    # If no format matches, try to parse as float (seconds since epoch)
    try:
        return datetime.fromtimestamp(float(timestamp_str))
    except (ValueError, OSError):
        return None

def load_fluorescence_data(csv_path, brightfield=False):
    """
    Load fluorescence data from CSV file.
    
    Args:
        brightfield (bool): If True, look for 'sum_optical_density' column instead of 'sum_brightness'
    
    Returns:
        tuple: (indices, brightness_values, filenames, timestamps)
    """
    indices = []
    brightness_values = []
    filenames = []
    timestamps = []
    has_timestamps = False
    
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Check if actual_time column exists
        has_timestamps = 'actual_time' in reader.fieldnames
        
        # Determine which column to use based on brightfield mode
        brightness_column = 'sum_optical_density' if brightfield else 'sum_brightness'
        
        # Check if the expected column exists
        if brightness_column not in reader.fieldnames:
            available_columns = ', '.join(reader.fieldnames)
            mode_desc = "brightfield" if brightfield else "fluorescence"
            raise ValueError(f"Expected column '{brightness_column}' not found in CSV file for {mode_desc} mode. "
                           f"Available columns: {available_columns}")
        
        print(f"Using column '{brightness_column}' for data values")
        
        for row in reader:
            indices.append(int(row['index']))
            
            # Handle float brightness/optical density values by truncating to integer
            brightness_raw = row[brightness_column]
            try:
                # Try to parse as float first, then truncate to int
                brightness_float = float(brightness_raw)
                brightness_int = int(brightness_float)  # Truncates (doesn't round)
                brightness_values.append(brightness_int)
            except ValueError:
                print(f"Warning: Could not parse {brightness_column} value '{brightness_raw}' in row, skipping")
                continue
            
            filenames.append(row['filename'])
            
            if has_timestamps:
                timestamp_str = row['actual_time']
                parsed_time = parse_timestamp(timestamp_str)
                timestamps.append(parsed_time)
    
    return (np.array(indices), np.array(brightness_values), filenames, 
            timestamps if has_timestamps else None, has_timestamps)

def calculate_time_intervals(timestamps):
    """
    Calculate time intervals between timestamps and return in minutes.
    
    Returns:
        tuple: (time_values_minutes, average_interval_minutes, is_regular)
    """
    if not timestamps or len(timestamps) < 2:
        return None, None, False
    
    # Filter out None values
    valid_timestamps = [(i, ts) for i, ts in enumerate(timestamps) if ts is not None]
    
    if len(valid_timestamps) < 2:
        return None, None, False
    
    # Calculate time differences
    time_diffs = []
    time_values_minutes = [0]  # Start at 0
    
    start_time = valid_timestamps[0][1]
    
    for i in range(1, len(valid_timestamps)):
        prev_time = valid_timestamps[i-1][1]
        curr_time = valid_timestamps[i][1]
        
        diff_minutes = (curr_time - start_time).total_seconds() / 60
        time_values_minutes.append(diff_minutes)
        
        interval_minutes = (curr_time - prev_time).total_seconds() / 60
        time_diffs.append(interval_minutes)
    
    avg_interval = np.mean(time_diffs)
    std_interval = np.std(time_diffs)
    
    # Consider intervals "regular" if standard deviation is less than 10% of mean
    is_regular = (std_interval / avg_interval) < 0.1 if avg_interval > 0 else False
    
    return np.array(time_values_minutes), avg_interval, is_regular

def plot_fluorescence(csv_path, window_size=5, time_interval=None, time_unit='auto', output_path=None, show_plot=True, confidence_level=0.95, figsize=(12, 8), brightfield=False, days_limit=None):
    """
    Create a smoothed trend plot of summed fluorescence over time/index with confidence intervals.
    
    Args:
        csv_path (str): Path to the CSV file
        window_size (int): Size of rolling window for smoothing
        time_interval (float, optional): Time interval between indices (in minutes) - overrides auto-detection
        time_unit (str): Time unit for X-axis ('auto', 'minutes', 'hours', 'days')
        output_path (str, optional): Path to save the plot
        show_plot (bool): Whether to display the plot
        confidence_level (float): Confidence level for intervals (default: 0.95 for 95% CI)
        figsize (tuple): Figure size as (width, height) in inches (default: (12, 8))
        brightfield (bool): Use brightfield titles and expect 'sum_optical_density' column
        days_limit (float, optional): Limit plot to first N days of data
    """
    # Load data with brightfield flag
    indices, brightness_values, filenames, timestamps, has_timestamps = load_fluorescence_data(csv_path, brightfield)
    
    # Convert to percent of initial value
    initial_value = np.mean(brightness_values[:window_size])
    if initial_value == 0:
        print("Warning: Initial value is zero. Cannot calculate percentage. Using raw values.")
        brightness_percent = brightness_values
        if brightfield:
            y_label = 'Total Optical Density (AU)'
        else:
            y_label = 'Total Fluorescence Signal (AU)'
        is_percentage = False
    else:
        brightness_percent = (brightness_values / initial_value) * 100
        if brightfield:
            y_label = 'OD Weighted Area\n(% of initial)'
        else:
            y_label = 'Fluor. Intensity Weighted\nArea (% of initial)'
        is_percentage = True
        print(f"Converting to percentage of initial value: {initial_value:,.0f}")
    
    # Determine time values for X-axis
    use_actual_times = False
    
    if has_timestamps and time_interval is None:
        # Try to use actual timestamps
        time_values_minutes, avg_interval, is_regular = calculate_time_intervals(timestamps)
        
        if time_values_minutes is not None:
            use_actual_times = True
            x_values_minutes = time_values_minutes
            print(f"Using actual timestamps from CSV")
            print(f"Average interval: {avg_interval:.2f} minutes")
            print(f"Regular intervals: {'Yes' if is_regular else 'No'}")
        else:
            print("Warning: Could not parse timestamps, using indices instead")
            use_actual_times = False
    elif time_interval is not None:
        # Use manually specified time interval
        x_values_minutes = indices * time_interval
        print(f"Using manually specified time interval: {time_interval} minutes per index")
    else:
        # Use indices
        use_actual_times = False
        print("Using image indices for X-axis")
    
    # Apply days limit if specified
    if days_limit is not None:
        days_limit_minutes = days_limit * 24 * 60  # Convert days to minutes
        
        if use_actual_times or time_interval is not None:
            # Filter based on time
            time_mask = x_values_minutes <= days_limit_minutes
            if np.any(time_mask):
                indices = indices[time_mask]
                brightness_percent = brightness_percent[time_mask]
                x_values_minutes = x_values_minutes[time_mask]
                print(f"Limiting plot to first {days_limit} days ({days_limit_minutes:.0f} minutes)")
                print(f"Data points after filtering: {len(indices)} (from {len(brightness_values)} total)")
            else:
                print(f"Warning: No data points found within {days_limit} days. Showing all data.")
        else:
            # When using indices, estimate based on time_interval if available
            if time_interval is not None:
                max_index = int(days_limit_minutes / time_interval)
                index_mask = indices <= max_index
                if np.any(index_mask):
                    indices = indices[index_mask]
                    brightness_percent = brightness_percent[index_mask]
                    print(f"Limiting plot to first {days_limit} days (up to index {max_index})")
                    print(f"Data points after filtering: {len(indices)} (from {len(brightness_values)} total)")
                else:
                    print(f"Warning: No data points found within {days_limit} days. Showing all data.")
            else:
                print(f"Warning: Cannot apply days limit without time information. Use --time argument or provide timestamps.")
    
    # Set up X-axis values and labels
    if use_actual_times or time_interval is not None:
        # Determine time unit and convert values
        if time_unit == 'auto':
            # Auto-select based on total time span
            total_minutes = max(x_values_minutes)
            if total_minutes > 2880:  # More than 2 days
                time_unit = 'days'
            elif total_minutes > 240:  # More than 4 hours
                time_unit = 'hours'
            else:
                time_unit = 'minutes'
        
        # Convert to appropriate units
        if time_unit == 'days':
            x_values = x_values_minutes / (60 * 24)
            x_label = 'Time (days)'
            x_title_suffix = 'Over Time'
        elif time_unit == 'hours':
            x_values = x_values_minutes / 60
            x_label = 'Time (hours)'
            x_title_suffix = 'Over Time'
        else:  # minutes
            x_values = x_values_minutes
            x_label = 'Time (minutes)'
            x_title_suffix = 'Over Time'
    else:
        x_values = indices
        x_label = 'Image Index'
        x_title_suffix = '(Smoothed Trend)'
        time_unit = 'index'
    
    # Create the plot with specified figure size
    plt.figure(figsize=figsize)
    
    # Calculate moving average and confidence intervals if window size is valid
    if window_size >= 2 and window_size <= len(brightness_percent):
        # Calculate moving statistics with edge handling
        moving_avg = np.zeros_like(brightness_percent, dtype=float)
        moving_std = np.zeros_like(brightness_percent, dtype=float)
        half_window = window_size // 2
        
        for i in range(len(brightness_percent)):
            left = max(0, i - half_window)
            right = min(len(brightness_percent), i + half_window + 1)
            window_vals = brightness_percent[left:right]
            moving_avg[i] = np.mean(window_vals)
            moving_std[i] = np.std(window_vals, ddof=1) if len(window_vals) > 1 else 0
        
        # Calculate confidence intervals
        # Use t-distribution for small windows, normal for large windows
        from scipy import stats
        alpha = 1 - confidence_level
        
        confidence_intervals = np.zeros_like(brightness_percent, dtype=float)
        for i in range(len(brightness_percent)):
            left = max(0, i - half_window)
            right = min(len(brightness_percent), i + half_window + 1)
            n = right - left
            
            if n > 1:
                if n < 30:  # Use t-distribution for small samples
                    t_val = stats.t.ppf(1 - alpha/2, df=n-1)
                    margin_error = t_val * moving_std[i] / np.sqrt(n)
                else:  # Use normal distribution for large samples
                    z_val = stats.norm.ppf(1 - alpha/2)
                    margin_error = z_val * moving_std[i] / np.sqrt(n)
                
                confidence_intervals[i] = margin_error
            else:
                confidence_intervals[i] = 0
        
        # Calculate upper and lower bounds
        upper_bound = moving_avg + confidence_intervals
        lower_bound = moving_avg - confidence_intervals
        
        # Plot original data as scatterplot
        plt.scatter(x_values, brightness_percent, c='lightblue', alpha=0.6, s=0.5, label='Original Data', zorder=1)
        
        # Plot confidence interval as filled area (blue)
        plt.fill_between(x_values, lower_bound, upper_bound, 
                        alpha=0.3, color='blue', 
                        label=f'{confidence_level*100:.0f}% CI', zorder=2)
        
        # Plot smoothed trend prominently (navy)
        plt.plot(x_values, moving_avg, color='navy', linewidth=1, 
                label=f'Smoothed Trend', zorder=3)
        
        print(f"Confidence intervals calculated using {'t-distribution' if window_size < 30 else 'normal distribution'}")
        
    else:
        # If window size is invalid, just plot original data
        plt.scatter(x_values, brightness_percent, c='blue', alpha=0.7, s=0.5, label='Raw Data')
        print(f"Warning: Window size {window_size} is invalid. Plotting original data only.")

    #plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
    # Force specific y-axis ticks at 100%, 150%, 200%, 250%
    if is_percentage:
        # Set specific y-axis ticks
        if brightfield:
            custom_ticks = [100, 150, 200]
            plt.gca().set_yticks(custom_ticks)
        
        # Ensure y-axis limits include all the custom ticks and data
        data_min = np.min(brightness_percent)
        data_max = np.max(brightness_percent)
        y_min = min(data_min, 100) - 5  # Add small buffer below
        y_max = max(data_max, 200) + 5  # Add small buffer above
        plt.ylim(y_min, y_max)
        
        # Format y-axis to show percentage symbols
        from matplotlib.ticker import FuncFormatter
        def percent_formatter(x, pos):
            return f'{x:.0f}%'
        plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    else:
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))

    # Set appropriate title based on imaging mode
    # title_suffix = f" (First {days_limit} days)" if days_limit is not None else ""
    # if brightfield:
    #    plt.title(f'Organoid Optical Density {x_title_suffix}{title_suffix}', fontsize=6, fontweight='bold')
    #else:
    #    plt.title(f'Fluorescent Protein Content {x_title_suffix}{title_suffix}', fontsize=6, fontweight='bold')
    
    plt.xlabel(x_label, fontsize=6.5, labelpad=1)
    plt.ylabel(y_label, fontsize=6.5, labelpad=1)
    plt.tick_params(axis='both', which='major', pad=1, width=0.25, length=1, labelsize=5)

    # Set border (spine) line widths
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.25)

    # Custom tick spacing for long time series
    if time_unit == 'days' and max(x_values) < 8:
        # Set ticks every 7 days for time series longer than 14 days
        import matplotlib.ticker as ticker
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
        print(f"Using 1-day tick intervals for {max(x_values):.1f} day time series")

    # Custom tick spacing for long time series
    if time_unit == 'days' and max(x_values) > 14:
        # Set ticks every 7 days for time series longer than 14 days
        import matplotlib.ticker as ticker
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))
        print(f"Using 7-day tick intervals for {max(x_values):.1f} day time series")
    
    # Add % symbols to y-axis if using percentages
    if is_percentage:
        # Format y-axis to show percentage symbols
        from matplotlib.ticker import FuncFormatter
        def percent_formatter(x, pos):
            return f'{x:.0f}%'
        plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    
    plt.legend(fontsize=5)
    
    # Reduce padding around the graph
    plt.tight_layout(pad=0.08)  # Reduced from default padding
    
    # Further reduce margins by adjusting subplot parameters
    #plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98)
    
    # Save plot if output path specified
    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    # Print trend analysis
    if len(brightness_percent) > 1:
        measurement_type = "optical density" if brightfield else "fluorescence signal"
        
        if is_percentage:
            trend = 'Increasing' if brightness_percent[-1] > brightness_percent[0] else 'Decreasing'
            final_percent = brightness_percent[-1]
            total_change_percent = brightness_percent[-1] - brightness_percent[0]
            
            print(f"\nTrend Analysis ({measurement_type}):")
            if days_limit is not None:
                print(f"Analysis period: First {days_limit} days")
            print(f"Initial level: 100.0%")
            print(f"Final level: {final_percent:.1f}%")
            print(f"Overall trend: {trend}")
            print(f"Total change: {total_change_percent:+.1f} percentage points")
            print(f"Relative change: {(final_percent - 100):.1f}% from initial")
            
            # Classify the change magnitude
            if abs(total_change_percent) < 5:
                change_magnitude = "Minimal"
            elif abs(total_change_percent) < 20:
                change_magnitude = "Moderate"
            elif abs(total_change_percent) < 50:
                change_magnitude = "Substantial"
            else:
                change_magnitude = "Dramatic"
            
            print(f"Change magnitude: {change_magnitude}")
            
        else:
            # Fallback to original analysis if not percentage
            trend = 'Increasing' if brightness_percent[-1] > brightness_percent[0] else 'Decreasing'
            total_change = brightness_percent[-1] - brightness_percent[0]
            print(f"\nTrend Analysis ({measurement_type}):")
            if days_limit is not None:
                print(f"Analysis period: First {days_limit} days")
            print(f"Overall trend: {trend}")
            print(f"Total change: {total_change:,.0f}")
            print(f"Percent change: {total_change/brightness_percent[0]*100:.2f}%")
        
        # Add confidence interval statistics
        if window_size >= 2 and window_size <= len(brightness_percent):
            print(f"\nConfidence Interval Analysis:")
            print(f"Confidence level: {confidence_level*100:.0f}%")
            if is_percentage:
                print(f"Average CI width: ±{np.mean(confidence_intervals):.1f} percentage points")
                print(f"Maximum CI width: ±{np.max(confidence_intervals):.1f} percentage points")
                print(f"CI as % of mean signal: {(np.mean(confidence_intervals)/np.mean(brightness_percent))*100:.1f}%")
            else:
                print(f"Average CI width: ±{np.mean(confidence_intervals):,.0f}")
                print(f"Maximum CI width: ±{np.max(confidence_intervals):,.0f}")
                print(f"CI as % of mean signal: {(np.mean(confidence_intervals)/np.mean(brightness_percent))*100:.1f}%")
        
        if use_actual_times:
            print(f"\nTime Analysis:")
            print(f"Time source: Actual timestamps from CSV")
            print(f"X-axis unit: {time_unit}")
            if time_unit == 'days':
                print(f"Time span plotted: {max(x_values):.2f} days")
            elif time_unit == 'hours':
                print(f"Time span plotted: {max(x_values):.2f} hours")
            else:
                print(f"Time span plotted: {max(x_values):.1f} minutes")
        elif time_interval is not None:
            total_time_minutes = (indices[-1] - indices[0]) * time_interval
            print(f"\nTime Analysis:")
            print(f"Time interval: {time_interval} minutes per index")
            print(f"X-axis unit: {time_unit}")
            
            if time_unit == 'days':
                print(f"Time span plotted: {total_time_minutes/(60*24):.2f} days")
            elif time_unit == 'hours':
                print(f"Time span plotted: {total_time_minutes/60:.2f} hours")
            else:
                print(f"Time span plotted: {total_time_minutes:.1f} minutes")
        else:
            print(f"\nIndex Analysis:")
            print(f"X-axis: Image indices ({indices[0]} to {indices[-1]})")
            if days_limit is not None:
                print(f"Filtered to show first {days_limit} days worth of data")
    
    return indices, brightness_percent

def main():
    """
    Main function to handle command line arguments and create plots.
    """
    if len(sys.argv) < 2:
        print("Usage: python plotsummedfluorescence.py <csv_file> [options]")
        print()
        print("Options:")
        print("  --window N       Size of rolling window for smoothing (default: 5)")
        print("  --time T         Time interval between indices in minutes (overrides auto-detection)")
        print("  --unit UNIT      Time unit for X-axis: auto, minutes, hours, days (default: auto)")
        print("  --days D         Limit plot to first D days of data (requires time info)")
        print("  --confidence C   Confidence level for intervals (default: 0.95 for 95%)")
        print("  --width W        Plot width in inches (default: 12)")
        print("  --height H       Plot height in inches (default: 8)")
        print("  --brightfield    Use brightfield mode (expects 'sum_optical_density' column)")
        print("  --output PATH    Save plot to specified path")
        print("  --no-show        Don't display the plot window")
        print()
        print("Column Expectations:")
        print("  Fluorescence mode: Expects 'sum_brightness' column (default)")
        print("  Brightfield mode:  Expects 'sum_optical_density' column (--brightfield flag)")
        print("  Both modes expect: 'index' and 'filename' columns")
        print("  Optional: 'actual_time' column for timestamp correlation")
        print()
        print("Examples:")
        print("  # Fluorescence data (expects sum_brightness column)")
        print("  python plotsummedfluorescence.py sum_brightness.csv")
        print("  # Brightfield data (expects sum_optical_density column)")
        print("  python plotsummedfluorescence.py sum_optical_density.csv --brightfield")
        print("  # Plot only first 7 days with timestamps")
        print("  python plotsummedfluorescence.py data.csv --days 7")
        print("  # Plot first 3 days with manual time interval")
        print("  python plotsummedfluorescence.py data.csv --time 30 --days 3")
        print("  # With custom smoothing and confidence")
        print("  python plotsummedfluorescence.py data.csv --window 10 --confidence 0.99")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Parse options
    window_size = 5
    time_interval = None
    time_unit = 'auto'
    confidence_level = 0.95
    plot_width = 12
    plot_height = 8
    brightfield = False
    output_path = None
    show_plot = True
    days_limit = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--window' and i + 1 < len(sys.argv):
            try:
                window_size = int(sys.argv[i + 1])
                if window_size < 1:
                    print("Error: Window size must be at least 1")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid window size '{sys.argv[i + 1]}'. Must be an integer.")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--time' and i + 1 < len(sys.argv):
            try:
                time_interval = float(sys.argv[i + 1])
                if time_interval <= 0:
                    print("Error: Time interval must be positive")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid time interval '{sys.argv[i + 1]}'. Must be a number.")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--days' and i + 1 < len(sys.argv):
            try:
                days_limit = float(sys.argv[i + 1])
                if days_limit <= 0:
                    print("Error: Days limit must be positive")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid days limit '{sys.argv[i + 1]}'. Must be a positive number.")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--confidence' and i + 1 < len(sys.argv):
            try:
                confidence_level = float(sys.argv[i + 1])
                if not (0 < confidence_level < 1):
                    print("Error: Confidence level must be between 0 and 1")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid confidence level '{sys.argv[i + 1]}'. Must be a number between 0 and 1.")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--width' and i + 1 < len(sys.argv):
            try:
                plot_width = float(sys.argv[i + 1])
                if plot_width <= 0:
                    print("Error: Plot width must be positive")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid plot width '{sys.argv[i + 1]}'. Must be a positive number.")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--height' and i + 1 < len(sys.argv):
            try:
                plot_height = float(sys.argv[i + 1])
                if plot_height <= 0:
                    print("Error: Plot height must be positive")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid plot height '{sys.argv[i + 1]}'. Must be a positive number.")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--brightfield':
            brightfield = True
            i += 1
        elif sys.argv[i] == '--unit' and i + 1 < len(sys.argv):
            time_unit = sys.argv[i + 1].lower()
            if time_unit not in ['auto', 'minutes', 'hours', 'days']:
                print(f"Error: Invalid time unit '{sys.argv[i + 1]}'. Must be one of: auto, minutes, hours, days")
                sys.exit(1)
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--no-show':
            show_plot = False
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)
    
    try:
        figsize = (plot_width, plot_height)
        plot_fluorescence(csv_path, window_size, time_interval, time_unit, output_path, show_plot, confidence_level, figsize, brightfield, days_limit)
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()