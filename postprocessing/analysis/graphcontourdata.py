import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path

def load_and_validate_data(csv_path):
    """
    Load CSV data and validate required columns.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        pandas.DataFrame: Loaded and validated data
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        
        # Check for required columns
        required_cols = ['timepoint', 'filename', 'area_pixels', 'contour_found']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter only successful detections
        successful_df = df[df['contour_found'] == True].copy()
        failed_count = len(df) - len(successful_df)
        
        if failed_count > 0:
            print(f"Note: {failed_count} images failed detection and will be excluded from graphs")
        
        if len(successful_df) == 0:
            raise ValueError("No successful detections found in data")
        
        print(f"Using {len(successful_df)} successful detections for graphing")
        
        # Sort by timepoint
        successful_df = successful_df.sort_values('timepoint')
        
        return successful_df
        
    except Exception as e:
        raise ValueError(f"Failed to load CSV file '{csv_path}': {e}")

def create_area_plots(df, output_dir, pixel_size=None, figsize=(8, 6)):
    """
    Create area vs time plots.
    
    Args:
        df: DataFrame with organoid data
        output_dir: Directory to save plots
        pixel_size: Pixel size in micrometers
        figsize: Figure size as (width, height) tuple
    """
    
    # Set up the plotting style
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Area in pixels
    axes[0].plot(df['timepoint'], df['area_pixels'], 'o-', linewidth=0.5, markersize=1, 
                color='darkblue', alpha=0.8)
    axes[0].set_xlabel('Timepoint', fontsize=5)
    axes[0].set_ylabel('Area (pixels)', fontsize=5)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=5)
    
    # Plot 2: Area in μm² (if pixel size provided)
    if pixel_size and 'area_um2' in df.columns and df['area_um2'].max() > 0:
        axes[1].plot(df['timepoint'], df['area_um2'], 'o-', linewidth=0.5, markersize=1, 
                    color='darkblue', alpha=0.8)
        axes[1].set_xlabel('Timepoint', fontsize=5)
        axes[1].set_ylabel('Area (μm²)', fontsize=5)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='both', which='major', labelsize=5)
    else:
        # Hide second subplot if no μm² data
        axes[1].text(0.5, 0.5, 'No area data in μm²\n(pixel size not provided or area_um2 column missing)', 
                    transform=axes[1].transAxes, ha='center', va='center', fontsize=5,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / "organoid_area_over_time.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Area plot saved to: {plot_file}")
    plt.close()

def create_darkness_plots(df, output_dir, figsize=(8, 6)):
    """
    Create darkness analysis plots.
    
    Args:
        df: DataFrame with organoid data
        output_dir: Directory to save plots
        figsize: Figure size as (width, height) tuple
    """
    
    # Check if darkness columns exist
    darkness_cols = ['total_darkness', 'average_darkness', 'raw_intensity_mean']
    available_cols = [col for col in darkness_cols if col in df.columns]
    
    if not available_cols:
        print("No darkness data available for plotting")
        return
    
    fig, axes = plt.subplots(len(available_cols), 1, figsize=(figsize[0], figsize[1] * len(available_cols) / 2))
    if len(available_cols) == 1:
        axes = [axes]  # Make it a list for consistency
    
    for i, col in enumerate(available_cols):
        if col in df.columns and df[col].notna().any():
            axes[i].plot(df['timepoint'], df[col], 'o-', linewidth=0.5, markersize=1, 
                        color='darkblue', alpha=0.8)
            
            if col == 'total_darkness':
                axes[i].set_ylabel('Total Darkness', fontsize=5)
            elif col == 'average_darkness':
                axes[i].set_ylabel('Average Darkness', fontsize=5)
            elif col == 'raw_intensity_mean':
                axes[i].set_ylabel('Mean Intensity', fontsize=5)
            
            axes[i].set_xlabel('Timepoint', fontsize=5)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='both', which='major', labelsize=5)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    plot_file = output_path / "organoid_darkness_over_time.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Darkness plot saved to: {plot_file}")
    plt.close()

def create_combined_analysis_plot(df, output_dir, pixel_size=None, figsize=(12, 8)):
    """
    Create a combined analysis plot with area and darkness.
    
    Args:
        df: DataFrame with organoid data
        output_dir: Directory to save plots
        pixel_size: Pixel size in micrometers
        figsize: Figure size as (width, height) tuple
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Area
    area_col = 'area_um2' if pixel_size and 'area_um2' in df.columns and df['area_um2'].max() > 0 else 'area_pixels'
    area_unit = 'μm²' if area_col == 'area_um2' else 'pixels'
    
    axes[0, 0].plot(df['timepoint'], df[area_col], 'o-', linewidth=0.5, markersize=1, 
                   color='darkblue', alpha=0.8)
    axes[0, 0].set_xlabel('Timepoint', fontsize=5)
    axes[0, 0].set_ylabel(f'Area ({area_unit})', fontsize=5)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=5)
    
    # Plot 2: Average Darkness (if available)
    if 'average_darkness' in df.columns:
        axes[0, 1].plot(df['timepoint'], df['average_darkness'], 'o-', linewidth=0.5, markersize=1, 
                       color='darkblue', alpha=0.8)
        axes[0, 1].set_xlabel('Timepoint', fontsize=5)
        axes[0, 1].set_ylabel('Average Darkness', fontsize=5)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=5)
    else:
        axes[0, 1].text(0.5, 0.5, 'Average Darkness\ndata not available', 
                       transform=axes[0, 1].transAxes, ha='center', va='center', fontsize=5)
    
    # Plot 3: Area vs Darkness correlation (if darkness data available)
    if 'average_darkness' in df.columns:
        scatter = axes[1, 0].scatter(df[area_col], df['average_darkness'], 
                                   c='darkblue', alpha=0.7, s=3)
        axes[1, 0].set_xlabel(f'Area ({area_unit})', fontsize=5)
        axes[1, 0].set_ylabel('Average Darkness', fontsize=5)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=5)
        
        # Calculate correlation
        correlation = df[area_col].corr(df['average_darkness'])
        axes[1, 0].text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 0].transAxes, verticalalignment='top', fontsize=5,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1, 0].text(0.5, 0.5, 'Area vs Darkness\ncorrelation not available', 
                       transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=5)
    
    # Plot 4: Growth rate (area change over time)
    if len(df) > 1:
        # Calculate growth rate as percentage change
        area_values = df[area_col].values
        timepoints = df['timepoint'].values
        
        growth_rates = []
        growth_timepoints = []
        
        for i in range(1, len(area_values)):
            if timepoints[i] != timepoints[i-1]:  # Avoid division by zero
                rate = ((area_values[i] - area_values[i-1]) / area_values[i-1]) * 100
                growth_rates.append(rate)
                growth_timepoints.append(timepoints[i])
        
        if growth_rates:
            axes[1, 1].plot(growth_timepoints, growth_rates, 'o-', linewidth=0.5, markersize=1, 
                           color='darkblue', alpha=0.8)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Timepoint', fontsize=5)
            axes[1, 1].set_ylabel('Growth Rate (%)', fontsize=5)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='both', which='major', labelsize=5)
        else:
            axes[1, 1].text(0.5, 0.5, 'Growth rate\ncannot be calculated', 
                           transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=5)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor growth rate', 
                       transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=5)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    plot_file = output_path / "organoid_combined_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Combined analysis plot saved to: {plot_file}")
    plt.close()

def create_summary_statistics(df, output_dir, pixel_size=None):
    """
    Create and save summary statistics.
    
    Args:
        df: DataFrame with organoid data
        output_dir: Directory to save statistics
        pixel_size: Pixel size in micrometers
    """
    
    output_path = Path(output_dir)
    stats_file = output_path / "organoid_analysis_summary.txt"
    
    with open(stats_file, 'w') as f:
        f.write("ORGANOID ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total timepoints analyzed: {len(df)}\n")
        f.write(f"Timepoint range: {df['timepoint'].min()} - {df['timepoint'].max()}\n\n")
        
        # Area statistics
        f.write("AREA STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Area (pixels):\n")
        f.write(f"  Mean: {df['area_pixels'].mean():.1f} ± {df['area_pixels'].std():.1f}\n")
        f.write(f"  Min: {df['area_pixels'].min():.1f}\n")
        f.write(f"  Max: {df['area_pixels'].max():.1f}\n")
        f.write(f"  Median: {df['area_pixels'].median():.1f}\n")
        
        if pixel_size and 'area_um2' in df.columns and df['area_um2'].max() > 0:
            f.write(f"\nArea (μm², pixel size: {pixel_size} μm/pixel):\n")
            f.write(f"  Mean: {df['area_um2'].mean():.1f} ± {df['area_um2'].std():.1f}\n")
            f.write(f"  Min: {df['area_um2'].min():.1f}\n")
            f.write(f"  Max: {df['area_um2'].max():.1f}\n")
            f.write(f"  Median: {df['area_um2'].median():.1f}\n")
        
        # Darkness statistics (if available)
        if 'average_darkness' in df.columns:
            f.write(f"\nDARKNESS STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Darkness:\n")
            f.write(f"  Mean: {df['average_darkness'].mean():.2f} ± {df['average_darkness'].std():.2f}\n")
            f.write(f"  Min: {df['average_darkness'].min():.2f}\n")
            f.write(f"  Max: {df['average_darkness'].max():.2f}\n")
            f.write(f"  Median: {df['average_darkness'].median():.2f}\n")
        
        if 'raw_intensity_mean' in df.columns:
            f.write(f"\nRaw Intensity:\n")
            f.write(f"  Mean: {df['raw_intensity_mean'].mean():.2f} ± {df['raw_intensity_mean'].std():.2f}\n")
            f.write(f"  Min: {df['raw_intensity_mean'].min():.2f}\n")
            f.write(f"  Max: {df['raw_intensity_mean'].max():.2f}\n")
            f.write(f"  Median: {df['raw_intensity_mean'].median():.2f}\n")
        
        # Growth analysis
        if len(df) > 1:
            area_col = 'area_um2' if pixel_size and 'area_um2' in df.columns and df['area_um2'].max() > 0 else 'area_pixels'
            initial_area = df[area_col].iloc[0]
            final_area = df[area_col].iloc[-1]
            total_change = ((final_area - initial_area) / initial_area) * 100
            
            f.write(f"\nGROWTH ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Initial area: {initial_area:.1f}\n")
            f.write(f"Final area: {final_area:.1f}\n")
            f.write(f"Total change: {total_change:.2f}%\n")
            
            # Calculate average growth rate per timepoint
            timepoint_diff = df['timepoint'].iloc[-1] - df['timepoint'].iloc[0]
            if timepoint_diff > 0:
                avg_growth_rate = total_change / timepoint_diff
                f.write(f"Average growth rate: {avg_growth_rate:.3f}% per timepoint\n")
    
    print(f"Summary statistics saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Graph organoid analysis data from CSV files generated by brightfieldorganoidtracker.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Graph data from summary CSV
  python graphcontourdata.py organoid_areas_summary.csv

  # Graph with pixel size and custom plot size
  python graphcontourdata.py organoid_areas_summary.csv --pixel-size 0.5 --width 10 --height 8

  # Graph detailed CSV with custom output directory
  python graphcontourdata.py organoid_analysis_results.csv --output-dir ./plots

  # Create only area plots with small size
  python graphcontourdata.py data.csv --plots area --width 6 --height 4

Output:
  - organoid_area_over_time.png: Area vs time plots
  - organoid_darkness_over_time.png: Darkness analysis plots (if data available)
  - organoid_combined_analysis.png: Comprehensive analysis with correlations
  - organoid_analysis_summary.txt: Statistical summary
        """)
    
    parser.add_argument('csv_file', help='CSV file from brightfieldorganoidtracker.py')
    parser.add_argument('--output-dir', default='./plots', 
                       help='Output directory for plots (default: ./plots)')
    parser.add_argument('--pixel-size', type=float,
                       help='Pixel size in micrometers (for proper μm² units)')
    parser.add_argument('--plots', choices=['all', 'area', 'darkness', 'combined'], 
                       default='all', help='Which plots to generate (default: all)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved plots (default: 300)')
    parser.add_argument('--width', type=float, default=8,
                       help='Plot width in inches (default: 8)')
    parser.add_argument('--height', type=float, default=6,
                       help='Plot height in inches (default: 6)')
    
    args = parser.parse_args()
    
    # Validate plot size parameters
    if args.width <= 0:
        print("Error: Plot width must be positive")
        sys.exit(1)
    
    if args.height <= 0:
        print("Error: Plot height must be positive")
        sys.exit(1)
    
    try:
        # Load and validate data
        df = load_and_validate_data(args.csv_file)
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating plots in: {output_path}")
        print(f"Plot size: {args.width} x {args.height} inches")
        print(f"Data range: timepoints {df['timepoint'].min()} to {df['timepoint'].max()}")
        
        # Set global DPI and font size
        plt.rcParams['figure.dpi'] = args.dpi
        plt.rcParams['font.size'] = 5
        
        # Set figure size tuple
        figsize = (args.width, args.height)
        
        # Generate requested plots
        if args.plots in ['all', 'area']:
            create_area_plots(df, args.output_dir, args.pixel_size, figsize)
        
        if args.plots in ['all', 'darkness']:
            create_darkness_plots(df, args.output_dir, figsize)
        
        if args.plots in ['all', 'combined']:
            # Use larger size for combined plot to accommodate 4 subplots
            combined_figsize = (args.width * 1.5, args.height * 1.3)
            create_combined_analysis_plot(df, args.output_dir, args.pixel_size, combined_figsize)
        
        # Always create summary statistics
        create_summary_statistics(df, args.output_dir, args.pixel_size)
        
        print("\n" + "="*50)
        print("PLOTTING COMPLETE")
        print("="*50)
        print(f"All files saved to: {output_path.absolute()}")
        
        # Show what was created
        plot_files = list(output_path.glob("*.png"))
        text_files = list(output_path.glob("*.txt"))
        
        if plot_files:
            print(f"\nPlots created ({len(plot_files)}):")
            for f in sorted(plot_files):
                print(f"  - {f.name}")
        
        if text_files:
            print(f"\nSummary files created ({len(text_files)}):")
            for f in sorted(text_files):
                print(f"  - {f.name}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()