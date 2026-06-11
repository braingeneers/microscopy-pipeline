"""Plot organoid analysis CSVs (output of brightfield_organoid_tracker)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..cli import build_io_parser


def _load(csv_path):
    df = pd.read_csv(csv_path)
    required = {"timepoint", "filename", "area_pixels", "contour_found"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df[df["contour_found"] == True].sort_values("timepoint").copy()
    if df.empty:
        raise ValueError("no successful detections in CSV")
    return df


def _plot_area(df, out_dir, pixel_size, figsize):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes[0].plot(df["timepoint"], df["area_pixels"], "o-", linewidth=0.5, markersize=1, color="darkblue", alpha=0.8)
    axes[0].set_xlabel("Timepoint", fontsize=5)
    axes[0].set_ylabel("Area (pixels)", fontsize=5)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=5)
    if pixel_size and "area_um2" in df.columns and df["area_um2"].max() > 0:
        axes[1].plot(df["timepoint"], df["area_um2"], "o-", linewidth=0.5, markersize=1, color="darkblue", alpha=0.8)
        axes[1].set_xlabel("Timepoint", fontsize=5)
        axes[1].set_ylabel("Area (\u03bcm\u00b2)", fontsize=5)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(labelsize=5)
    else:
        axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "organoid_area_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_darkness(df, out_dir, figsize):
    import matplotlib.pyplot as plt
    cols = [c for c in ("total_darkness", "average_darkness", "raw_intensity_mean") if c in df.columns]
    if not cols:
        return
    fig, axes = plt.subplots(len(cols), 1, figsize=(figsize[0], figsize[1] * len(cols) / 2))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(df["timepoint"], df[col], "o-", linewidth=0.5, markersize=1, color="darkblue", alpha=0.8)
        ax.set_xlabel("Timepoint", fontsize=5)
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=5)
    plt.tight_layout()
    plt.savefig(out_dir / "organoid_darkness_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_combined(df, out_dir, pixel_size, figsize):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    area_col = "area_um2" if pixel_size and "area_um2" in df.columns and df["area_um2"].max() > 0 else "area_pixels"
    unit = "\u03bcm\u00b2" if area_col == "area_um2" else "pixels"
    axes[0, 0].plot(df["timepoint"], df[area_col], "o-", linewidth=0.5, markersize=1, color="darkblue", alpha=0.8)
    axes[0, 0].set_xlabel("Timepoint", fontsize=5)
    axes[0, 0].set_ylabel(f"Area ({unit})", fontsize=5)
    axes[0, 0].tick_params(labelsize=5)
    if "average_darkness" in df.columns:
        axes[0, 1].plot(df["timepoint"], df["average_darkness"], "o-", linewidth=0.5, markersize=1, color="darkblue", alpha=0.8)
        axes[0, 1].set_xlabel("Timepoint", fontsize=5)
        axes[0, 1].set_ylabel("Average Darkness", fontsize=5)
        axes[0, 1].tick_params(labelsize=5)
        axes[1, 0].scatter(df[area_col], df["average_darkness"], c="darkblue", alpha=0.7, s=3)
        axes[1, 0].set_xlabel(f"Area ({unit})", fontsize=5)
        axes[1, 0].set_ylabel("Average Darkness", fontsize=5)
        corr = df[area_col].corr(df["average_darkness"])
        axes[1, 0].text(0.02, 0.98, f"Correlation: {corr:.3f}", transform=axes[1, 0].transAxes,
                        verticalalignment="top", fontsize=5,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        axes[1, 0].tick_params(labelsize=5)
    if len(df) > 1:
        rates, tps = [], []
        for i in range(1, len(df)):
            prev = df[area_col].iloc[i - 1]
            curr = df[area_col].iloc[i]
            tp_prev = df["timepoint"].iloc[i - 1]
            tp_curr = df["timepoint"].iloc[i]
            if prev and tp_curr != tp_prev:
                rates.append((curr - prev) / prev * 100)
                tps.append(tp_curr)
        if rates:
            axes[1, 1].plot(tps, rates, "o-", linewidth=0.5, markersize=1, color="darkblue", alpha=0.8)
            axes[1, 1].axhline(0, color="red", linestyle="--", alpha=0.5)
            axes[1, 1].set_xlabel("Timepoint", fontsize=5)
            axes[1, 1].set_ylabel("Growth Rate (%)", fontsize=5)
            axes[1, 1].tick_params(labelsize=5)
    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "organoid_combined_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def _write_summary(df, out_dir, pixel_size):
    path = out_dir / "organoid_analysis_summary.txt"
    with open(path, "w") as f:
        f.write("ORGANOID ANALYSIS SUMMARY\n" + "=" * 50 + "\n\n")
        f.write(f"Total timepoints analyzed: {len(df)}\n")
        f.write(f"Timepoint range: {df['timepoint'].min()} - {df['timepoint'].max()}\n\n")
        f.write("AREA STATISTICS\n" + "-" * 20 + "\n")
        f.write(f"area_pixels mean={df['area_pixels'].mean():.1f}  std={df['area_pixels'].std():.1f}\n")
        if pixel_size and "area_um2" in df.columns and df["area_um2"].max() > 0:
            f.write(f"area_um2 mean={df['area_um2'].mean():.1f}  std={df['area_um2'].std():.1f}\n")
        if "average_darkness" in df.columns:
            f.write("\nDARKNESS STATISTICS\n" + "-" * 20 + "\n")
            f.write(f"average_darkness mean={df['average_darkness'].mean():.2f}  std={df['average_darkness'].std():.2f}\n")


def graph_contour_data(
    csv_path,
    output_dir,
    *,
    pixel_size: Optional[float] = None,
    plots: str = "all",
    width: float = 8,
    height: float = 6,
    dpi: int = 300,
):
    """Generate area/darkness/combined plots from an organoid-tracking CSV."""
    import matplotlib.pyplot as plt
    df = _load(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["font.size"] = 5
    figsize = (width, height)
    if plots in ("all", "area"):
        _plot_area(df, out, pixel_size, figsize)
    if plots in ("all", "darkness"):
        _plot_darkness(df, out, figsize)
    if plots in ("all", "combined"):
        _plot_combined(df, out, pixel_size, (width * 1.5, height * 1.3))
    _write_summary(df, out, pixel_size)


def cli(argv=None):
    parser = build_io_parser(
        "Plot organoid analysis CSV (from brightfield_organoid_tracker) into PNG figures."
    )
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="Pixel size in micrometres (enables area_um2 plots).")
    parser.add_argument("--plots", choices=("all", "area", "darkness", "combined"), default="all")
    parser.add_argument("--width", type=float, default=8)
    parser.add_argument("--height", type=float, default=6)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    graph_contour_data(args.input, args.output,
                       pixel_size=args.pixel_size, plots=args.plots,
                       width=args.width, height=args.height, dpi=args.dpi)


if __name__ == "__main__":
    cli()
