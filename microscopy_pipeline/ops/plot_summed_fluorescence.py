"""Plot summed-fluorescence/optical-density CSV time series with smoothed trend + CI."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..cli import build_io_parser

_TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%H:%M:%S",
    "%H:%M",
]


def _parse_timestamp(s: str) -> Optional[datetime]:
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(float(s))
    except (ValueError, OSError):
        return None


def _load(csv_path, brightfield: bool):
    column = "sum_optical_density" if brightfield else "sum_brightness"
    indices, values, timestamps = [], [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if column not in reader.fieldnames:
            raise ValueError(f"column '{column}' not found; available: {reader.fieldnames}")
        has_ts = "actual_time" in reader.fieldnames
        for row in reader:
            try:
                v = float(row[column])
            except (ValueError, KeyError):
                continue
            indices.append(int(row["index"]))
            values.append(int(v))
            if has_ts:
                timestamps.append(_parse_timestamp(row["actual_time"]))
    return np.array(indices), np.array(values), timestamps if has_ts else None


def plot_fluorescence(
    csv_path,
    output_path=None,
    *,
    window_size: int = 5,
    time_interval_min: Optional[float] = None,
    time_unit: str = "auto",
    confidence_level: float = 0.95,
    figsize: Tuple[float, float] = (12, 8),
    brightfield: bool = False,
    days_limit: Optional[float] = None,
    show: bool = False,
):
    """Smoothed plot of a fluorescence/OD time series with a confidence band."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from scipy import stats

    indices, values, timestamps = _load(csv_path, brightfield)
    if len(values) == 0:
        raise ValueError("no rows loaded from CSV")

    initial = float(np.mean(values[:window_size])) if window_size > 0 else float(values[0])
    if initial == 0:
        pct = values.astype(float)
        is_pct = False
        ylabel = "Total Optical Density (AU)" if brightfield else "Total Fluorescence Signal (AU)"
    else:
        pct = values.astype(float) / initial * 100.0
        is_pct = True
        ylabel = ("OD Weighted Area\n(% of initial)" if brightfield
                  else "Fluor. Intensity Weighted\nArea (% of initial)")

    use_actual = False
    if timestamps is not None and time_interval_min is None:
        valid = [(i, t) for i, t in enumerate(timestamps) if t is not None]
        if len(valid) >= 2:
            t0 = valid[0][1]
            xs_min = np.array([(t - t0).total_seconds() / 60.0
                               for _, t in [(i, ts) for i, ts in enumerate(timestamps) if ts is not None]])
            use_actual = True
    if not use_actual:
        if time_interval_min is not None:
            xs_min = indices.astype(float) * time_interval_min
        else:
            xs_min = None

    if days_limit and xs_min is not None:
        keep = xs_min <= days_limit * 24 * 60
        indices, pct, xs_min = indices[keep], pct[keep], xs_min[keep]

    if xs_min is not None:
        if time_unit == "auto":
            total = xs_min.max() if len(xs_min) else 0
            time_unit = "days" if total > 2880 else ("hours" if total > 240 else "minutes")
        if time_unit == "days":
            xs, xlabel = xs_min / (60 * 24), "Time (days)"
        elif time_unit == "hours":
            xs, xlabel = xs_min / 60, "Time (hours)"
        else:
            xs, xlabel = xs_min, "Time (minutes)"
    else:
        xs, xlabel = indices.astype(float), "Image Index"

    plt.figure(figsize=figsize)
    if 2 <= window_size <= len(pct):
        half = window_size // 2
        ma = np.zeros_like(pct)
        ci = np.zeros_like(pct)
        alpha = 1 - confidence_level
        for i in range(len(pct)):
            l = max(0, i - half); r = min(len(pct), i + half + 1)
            w = pct[l:r]
            ma[i] = np.mean(w)
            if len(w) > 1:
                std = np.std(w, ddof=1)
                if len(w) < 30:
                    crit = stats.t.ppf(1 - alpha / 2, df=len(w) - 1)
                else:
                    crit = stats.norm.ppf(1 - alpha / 2)
                ci[i] = crit * std / np.sqrt(len(w))
        plt.scatter(xs, pct, c="lightblue", s=0.5, alpha=0.6, label="Original Data", zorder=1)
        plt.fill_between(xs, ma - ci, ma + ci, alpha=0.3, color="blue",
                         label=f"{confidence_level*100:.0f}% CI", zorder=2)
        plt.plot(xs, ma, color="navy", linewidth=1, label="Smoothed Trend", zorder=3)
    else:
        plt.scatter(xs, pct, c="blue", s=0.5, alpha=0.7, label="Raw Data")

    plt.xlabel(xlabel, fontsize=6.5, labelpad=1)
    plt.ylabel(ylabel, fontsize=6.5, labelpad=1)
    plt.tick_params(axis="both", which="major", pad=1, width=0.25, length=1, labelsize=5)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.25)
    if is_pct:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
    if xlabel.startswith("Time (days)"):
        if xs.max() > 14:
            plt.gca().xaxis.set_major_locator(MultipleLocator(7))
        elif xs.max() < 8:
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.legend(fontsize=5)
    plt.tight_layout(pad=0.08)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        print(f"plot saved to {output_path}")
    if show:
        plt.show()
    plt.close()


def cli(argv=None):
    parser = build_io_parser(
        "Plot a smoothed time series from a sum_brightness/sum_optical_density CSV.",
        output_required=False,
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--time-interval-min", type=float, default=None,
                        help="Minutes between successive indices (overrides timestamps).")
    parser.add_argument("--time-unit", choices=("auto", "minutes", "hours", "days"), default="auto")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--width", type=float, default=12)
    parser.add_argument("--height", type=float, default=8)
    parser.add_argument("--brightfield", action="store_true")
    parser.add_argument("--days-limit", type=float, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    plot_fluorescence(
        args.input, args.output,
        window_size=args.window_size,
        time_interval_min=args.time_interval_min,
        time_unit=args.time_unit,
        confidence_level=args.confidence_level,
        figsize=(args.width, args.height),
        brightfield=args.brightfield,
        days_limit=args.days_limit,
        show=args.show,
    )


if __name__ == "__main__":
    cli()
