"""Slow-oscillation ORIGIN test for a human cortex video.

Motion on exposed cortex can carry a slow (~0.05 Hz) oscillation. This script
decides whether it is physiological (confined to tissue) or a global field /
optical drift (present off-tissue and phase-locked everywhere) by comparing four
ROIs -- clean tissue, a reflection patch, a vessel, and an off-brain control --
in the low-frequency band, and cross-correlating their slow components.

Usage:
    python analyze_slow_oscillation.py <video-path>

ROI boxes (fraction x,y,w,h) are the ones used in this study; edit for new views.
Writes $PULS_WORK/out/slow_oscillation_origin.{png,pdf}.
"""
import os, sys, logging
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pulsatility import load_video_gray, stabilize_frames, build_roi_masks, motion_signals

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work"))
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
ROIS = {"tissue": "0.34,0.20,0.18,0.32", "reflection": "0.436,0.400,0.247,0.440",
        "vessel": "0.55,0.55,0.11,0.196", "control": "0.83,0.03,0.14,0.25"}
COLORS = {"tissue": "#e16f24", "reflection": "#2da44e", "vessel": "#8250df", "control": "#1f6feb"}
LBL = {"tissue": "Tissue (clean cortex)", "reflection": "Reflection",
       "vessel": "Vessel/glare", "control": "Control (off-brain)"}


def hann_fft(x, fps):
    x = detrend(np.asarray(x, float), type="linear"); x = x - x.mean(); w = np.hanning(len(x))
    return np.fft.rfftfreq(len(x), d=1 / fps) * 60, np.abs(np.fft.rfft(x * w)) ** 2


def bandpass(x, fps, lo, hi):
    b, a = butter(3, [lo / (fps / 2), hi / (fps / 2)], btype="band")
    return filtfilt(b, a, detrend(np.asarray(x, float), type="linear"))


def frac(bpm, P, lo, hi):
    tot = P[(bpm >= 1.5) & (bpm <= 120)].sum()
    return P[(bpm >= lo) & (bpm <= hi)].sum() / (tot or 1)


def main(video):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    order = list(ROIS)
    frames, fps, scale = load_video_gray(video, resize_width=320, frame_stride=2,
                                         max_frames=1600, return_scale=True)
    frames = stabilize_frames(frames, mode="euclidean", reference="mid")
    regions = build_roi_masks(frames[0].shape, scale=scale,
                              rois=[f"{k}={v}" for k, v in ROIS.items()], units="fraction")
    sig, _, _ = motion_signals(frames, [r.mask for r in regions], method="flow")
    raw = {n: s.astype(float) for n, s in zip([r.name for r in regions], sig)}
    t = np.arange(len(next(iter(raw.values())))) / fps

    stats = {}
    for k in order:
        bpm, P = hann_fft(raw[k], fps)
        stats[k] = dict(bpm=bpm, P=P, slow=frac(bpm, P, 1.5, 6), card=frac(bpm, P, 80, 95))
    slow_ts = {k: bandpass(raw[k], fps, 1.5 / 60, 6 / 60) for k in order}
    edge = int(5 * fps)

    def zc(a, b):
        a, b = a[edge:-edge], b[edge:-edge]
        a = (a - a.mean()) / (a.std() or 1); b = (b - b.mean()) / (b.std() or 1)
        return float(np.corrcoef(a, b)[0, 1])

    print("ROI            slow(1.5-6)  cardiac(80-95)  corr-vs-tissue")
    for k in order:
        print(f"{k:14s} {stats[k]['slow']:.3f}       {stats[k]['card']:.3f}          {zc(slow_ts['tissue'], slow_ts[k]):+.2f}")

    fig = plt.figure(figsize=(12, 10.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.24, top=0.9, bottom=0.09)
    axA = fig.add_subplot(gs[0, 0])
    for k in order:
        s = stats[k]; m = (s["bpm"] >= 1.5) & (s["bpm"] <= 40)
        axA.semilogy(s["bpm"][m], s["P"][m] / s["P"][m].max(), color=COLORS[k], lw=1.5, label=LBL[k])
    axA.set_xlim(0, 40); axA.set_ylim(3e-3, 1.6); axA.set_xlabel("Rate (bpm)")
    axA.set_ylabel("rel. power (log)"); axA.set_title("Low-frequency spectrum by ROI"); axA.legend(fontsize=8.5)
    xs = np.arange(len(order))
    axB = fig.add_subplot(gs[0, 1])
    axB.bar(xs, [stats[k]["slow"] for k in order], color=[COLORS[k] for k in order], alpha=0.75)
    axB.set_xticks(xs); axB.set_xticklabels(order, rotation=12); axB.set_ylabel("fraction in 1.5-6 bpm")
    axB.set_title("Slow-oscillation power fraction"); axB.grid(alpha=0.15, axis="y")
    axC = fig.add_subplot(gs[1, 0])
    axC.bar(xs, [stats[k]["card"] for k in order], color=[COLORS[k] for k in order], alpha=0.75)
    axC.set_xticks(xs); axC.set_xticklabels(order, rotation=12); axC.set_ylabel("fraction in 80-95 bpm")
    axC.set_title("Cardiac power fraction (on-brain check)"); axC.grid(alpha=0.15, axis="y")
    axD = fig.add_subplot(gs[1, 1])
    for k in order:
        axD.plot(t, slow_ts[k] / (np.std(slow_ts[k]) or 1), color=COLORS[k], lw=1.2,
                 label=f"{k} (r={zc(slow_ts['tissue'], slow_ts[k]):+.2f})")
    axD.set_xlim(0, t[-1]); axD.set_xlabel("Time (s)"); axD.set_ylabel("slow band (z)")
    axD.set_title("Slow-band time course (r vs tissue)"); axD.legend(fontsize=8.5)
    fig.suptitle("Where does the slow oscillation come from? — ROI origin test", y=0.955, fontsize=14.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slow_oscillation_origin.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote slow_oscillation_origin.{png,pdf}")


if __name__ == "__main__":
    main(sys.argv[1])
