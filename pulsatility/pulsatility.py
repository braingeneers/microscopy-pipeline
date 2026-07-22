"""Quantify bulk parenchymal pulsatility from a brightfield/microscopy video.

The tool measures how much the parenchyma (a perfused gel block, or brain) moves
with each pulse of the perfusing flow and expresses it as a **continuous
pulsatility waveform**, together with the pulse rate, a pulsatility index and a
spatial map of where the tissue pulses most.

Pipeline
--------
1. ``load_video_gray``  -- decode the video to a list of grayscale frames + fps.
2. ``motion_signal``    -- per-frame tissue-motion signal via dense optical flow
                           (default) or frame differencing, plus a per-pixel
                           amplitude map.
3. ``analyze_pulsatility`` -- detrend, find the dominant pulsation frequency,
                           band-pass to a clean wave, detect pulses and compute
                           rate / variability / pulsatility index.
4. ``plot_pulsatility`` -- render the waveform, spectrum and amplitude map.

Everything below the I/O boundary is pure ``numpy``; ``analyze_pulsatility``
works on any list of frames, so it is testable without decoding a real video.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


# --------------------------------------------------------------------------- #
# 1. Video I/O
# --------------------------------------------------------------------------- #
def load_video_gray(
    path,
    *,
    resize_width: Optional[int] = 320,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> Tuple[List[np.ndarray], float]:
    """Decode ``path`` to a list of float32 grayscale frames and return ``(frames, fps)``.

    Frames are optionally downscaled to ``resize_width`` (preserving aspect
    ratio). Pulsatility is a bulk, low-spatial-frequency phenomenon, so
    downscaling both speeds up optical flow and suppresses pixel noise without
    losing the signal. ``frame_stride`` keeps every Nth frame (the returned fps
    is divided accordingly) and ``max_frames`` caps how many frames are read.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        logger.warning("video reported fps<=0; defaulting to 30.0")
        fps = 30.0

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if resize_width and gray.shape[1] > resize_width:
                scale = resize_width / gray.shape[1]
                new_size = (resize_width, max(1, int(round(gray.shape[0] * scale))))
                gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
            frames.append(gray.astype(np.float32))
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if len(frames) < 3:
        raise ValueError(f"video has too few frames ({len(frames)}) to analyze")
    effective_fps = fps / frame_stride
    logger.info("loaded %d frames at %.3f fps (%.2fs)", len(frames), effective_fps,
                len(frames) / effective_fps)
    return frames, effective_fps


# --------------------------------------------------------------------------- #
# 2. Tissue-motion signal
# --------------------------------------------------------------------------- #
def motion_signal(
    frames: Sequence[np.ndarray],
    *,
    method: str = "flow",
    smooth_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame-pair tissue-motion signal + a per-pixel amplitude map.

    ``method="flow"`` (default) uses dense Farneback optical flow and takes the
    mean flow *magnitude* over the field as the bulk tissue speed for that
    frame-pair -- a direct, illumination-robust measure of how fast the
    parenchyma is moving. ``method="diff"`` uses the mean absolute frame
    difference, which is faster but conflates real motion with brightness change.

    Returns ``(signal, amplitude_map, reference)`` where ``signal`` has
    ``len(frames)-1`` samples (one per consecutive pair), ``amplitude_map`` is the
    per-pixel temporal standard deviation of the local motion (highlighting where
    pulsation is strongest), and ``reference`` is the median frame for display.
    """
    import cv2

    if method not in ("flow", "diff"):
        raise ValueError("method must be 'flow' or 'diff'")

    n = len(frames)
    signal = np.zeros(n - 1, dtype=np.float64)
    h, w = frames[0].shape
    acc_sum = np.zeros((h, w), dtype=np.float64)
    acc_sq = np.zeros((h, w), dtype=np.float64)

    prev = frames[0]
    if smooth_sigma and method == "flow":
        prev = cv2.GaussianBlur(prev, (0, 0), smooth_sigma)
    for i in range(1, n):
        cur = frames[i]
        if smooth_sigma and method == "flow":
            cur = cv2.GaussianBlur(cur, (0, 0), smooth_sigma)
        if method == "flow":
            flow = cv2.calcOpticalFlowFarneback(
                prev, cur, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            local = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        else:
            local = np.abs(cur - prev)
        signal[i - 1] = float(local.mean())
        acc_sum += local
        acc_sq += local * local
        prev = cur

    m = n - 1
    mean_map = acc_sum / m
    amplitude_map = np.sqrt(np.maximum(acc_sq / m - mean_map ** 2, 0.0))
    reference = np.median(np.stack(frames), axis=0).astype(np.float32)
    return signal, amplitude_map, reference


# --------------------------------------------------------------------------- #
# 3. Signal analysis
# --------------------------------------------------------------------------- #
@dataclass
class PulsatilityResult:
    """Everything needed to describe and plot the pulsatility of one video."""

    fps: float
    times: np.ndarray                 # seconds, one per signal sample
    raw: np.ndarray                   # tissue-motion signal (px/frame)
    detrended: np.ndarray             # drift removed, waveform shape preserved
    wave: np.ndarray                  # narrow-band clean oscillation
    freqs: np.ndarray                 # spectrum frequency axis (Hz)
    power: np.ndarray                 # spectrum power
    dominant_hz: float                # fundamental pulsation frequency
    bpm_spectral: float               # dominant_hz * 60
    bpm_peaks: float                  # rate from peak-to-peak intervals
    rate_cv: float                    # coeff. of variation of pulse intervals
    peak_indices: np.ndarray          # indices into the signal of detected pulses
    pulsatility_index: float          # modulation depth (P95-P5)/mean of raw
    amplitude_rms: float              # RMS of the narrow-band wave (px/frame)
    amplitude_p2p: float              # median peak-to-trough of raw per cycle
    spectral_snr: float               # fraction of band power at the fundamental
    amplitude_map: np.ndarray         # per-pixel pulsation amplitude
    reference: np.ndarray             # reference frame for overlay
    method: str = "flow"
    pixel_size_um: Optional[float] = None
    band_hz: Tuple[float, float] = (0.5, 5.0)
    n_pulses: int = 0
    extras: dict = field(default_factory=dict)

    # unit helpers -------------------------------------------------------- #
    @property
    def dominant_period_s(self) -> float:
        return 1.0 / self.dominant_hz if self.dominant_hz > 0 else float("nan")

    def summary(self) -> str:
        px = self.pixel_size_um
        amp_unit = "px/frame"
        amp_rms = self.amplitude_rms
        amp_p2p = self.amplitude_p2p
        if px:
            amp_rms *= px * self.fps
            amp_p2p *= px * self.fps
            amp_unit = "um/s"
        lines = [
            "Bulk parenchymal pulsatility",
            "=" * 30,
            f"method                : optical-{self.method}",
            f"pulse rate (spectral) : {self.bpm_spectral:6.1f} pulses/min "
            f"({self.dominant_hz:.3f} Hz)",
            f"pulse rate (peaks)    : {self.bpm_peaks:6.1f} pulses/min "
            f"over {self.n_pulses} pulses",
            f"rate variability (CV) : {self.rate_cv*100:5.1f} %",
            f"pulsatility index     : {self.pulsatility_index:6.3f}  "
            f"(modulation depth of tissue motion)",
            f"wave amplitude (RMS)  : {amp_rms:8.4f} {amp_unit}",
            f"wave amplitude (p2p)  : {amp_p2p:8.4f} {amp_unit}",
            f"spectral SNR          : {self.spectral_snr*100:5.1f} % of band power "
            f"at the fundamental",
        ]
        if px:
            lines.append(f"pixel size            : {px:.4f} um/px")
        return "\n".join(lines)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average with edge-reflection padding (odd window)."""
    win = max(1, int(win))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, pad, mode="reflect")
    kernel = np.ones(win) / win
    return np.convolve(xp, kernel, mode="valid")


def _dominant_frequency(
    sig: np.ndarray, fps: float, band_hz: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(freqs, power, f0, snr)`` for the Hann-windowed spectrum of ``sig``.

    ``f0`` is the peak frequency inside ``band_hz``; ``snr`` is the fraction of
    in-band power concentrated within +/-15% of ``f0`` (a coherence measure).
    """
    s = sig - sig.mean()
    if s.std() > 0:
        s = s * np.hanning(len(s))
    # zero-pad to >=4x for finer frequency resolution
    nfft = int(2 ** np.ceil(np.log2(max(len(s) * 4, 256))))
    spec = np.fft.rfft(s, n=nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fps)
    power = np.abs(spec) ** 2

    lo, hi = band_hz
    hi = min(hi, fps / 2.0 * 0.98)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return freqs, power, 0.0, 0.0
    band_idx = np.where(band)[0]
    f0 = float(freqs[band_idx[np.argmax(power[band_idx])]])

    near = (freqs >= f0 * 0.85) & (freqs <= f0 * 1.15)
    total = power[band].sum()
    snr = float(power[near].sum() / total) if total > 0 else 0.0
    return freqs, power, f0, snr


def _bandpass(sig: np.ndarray, fps: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass; falls back gracefully on short signals."""
    from scipy.signal import butter, filtfilt

    nyq = fps / 2.0
    lo_n = max(1e-4, lo / nyq)
    hi_n = min(0.999, hi / nyq)
    if hi_n <= lo_n:
        return sig - sig.mean()
    try:
        b, a = butter(2, [lo_n, hi_n], btype="band")
        pad = 3 * max(len(a), len(b))
        if len(sig) <= pad:
            return sig - sig.mean()
        return filtfilt(b, a, sig)
    except ValueError:
        return sig - sig.mean()


def analyze_pulsatility(
    frames: Sequence[np.ndarray],
    fps: float,
    *,
    method: str = "flow",
    min_bpm: float = 30.0,
    max_bpm: float = 300.0,
    pixel_size_um: Optional[float] = None,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> PulsatilityResult:
    """Turn a list of grayscale frames into a :class:`PulsatilityResult`.

    ``min_bpm``/``max_bpm`` bound the search for the pulsation frequency (default
    30-300 pulses/min covers physiological heart rates and typical perfusion-pump
    rates). Pass ``precomputed=(signal, amplitude_map, reference)`` to reuse a
    motion signal already computed by :func:`motion_signal`.
    """
    from scipy.signal import find_peaks

    band_hz = (min_bpm / 60.0, max_bpm / 60.0)
    if precomputed is not None:
        raw, amplitude_map, reference = precomputed
    else:
        raw, amplitude_map, reference = motion_signal(frames, method=method)

    m = len(raw)
    times = (np.arange(m) + 0.5) / fps

    # Detrend: remove drift slower than the slowest expected pulse while keeping
    # the pulse waveform shape intact.
    baseline_win = int(round(fps / band_hz[0] * 1.5))
    baseline = _moving_average(raw, baseline_win)
    detrended = raw - baseline

    freqs, power, f0, snr = _dominant_frequency(detrended, fps, band_hz)
    if f0 <= 0:
        f0 = float(np.mean(band_hz))

    # Narrow band around the fundamental -> clean continuous wave.
    wave = _bandpass(detrended, fps, f0 * 0.7, min(f0 * 1.6, fps / 2 * 0.95))

    # Peak detection on the clean wave gives pulse timing.
    min_dist = max(1, int(round(fps / f0 * 0.6)))
    prom = 0.5 * np.std(wave) if np.std(wave) > 0 else None
    peaks, _ = find_peaks(wave, distance=min_dist, prominence=prom)

    if len(peaks) >= 2:
        intervals = np.diff(peaks) / fps
        bpm_peaks = 60.0 / float(np.mean(intervals))
        rate_cv = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) else 0.0
    else:
        bpm_peaks, rate_cv, intervals = float("nan"), float("nan"), np.array([])

    # Pulsatility index = modulation depth of the (drift-removed) tissue speed.
    speed = raw - baseline + float(np.mean(raw))
    mean_speed = float(np.mean(raw))
    pulsatility_index = (
        float(np.percentile(speed, 95) - np.percentile(speed, 5)) / mean_speed
        if mean_speed > 0 else 0.0
    )

    # Per-cycle peak-to-trough amplitude of the raw signal.
    amp_p2p = float("nan")
    if len(peaks) >= 2:
        troughs, _ = find_peaks(-wave, distance=min_dist)
        if len(troughs):
            p2p = []
            for pk in peaks:
                nearest = troughs[np.argmin(np.abs(troughs - pk))]
                p2p.append(abs(raw[pk] - raw[nearest]))
            amp_p2p = float(np.median(p2p))

    return PulsatilityResult(
        fps=fps,
        times=times,
        raw=raw,
        detrended=detrended,
        wave=wave,
        freqs=freqs,
        power=power,
        dominant_hz=f0,
        bpm_spectral=f0 * 60.0,
        bpm_peaks=bpm_peaks,
        rate_cv=rate_cv,
        peak_indices=peaks,
        pulsatility_index=pulsatility_index,
        amplitude_rms=float(np.sqrt(np.mean(wave ** 2))),
        amplitude_p2p=amp_p2p,
        spectral_snr=snr,
        amplitude_map=amplitude_map,
        reference=reference,
        method=method,
        pixel_size_um=pixel_size_um,
        band_hz=band_hz,
        n_pulses=int(len(peaks)),
        extras={"intervals_s": intervals, "baseline": baseline},
    )


# --------------------------------------------------------------------------- #
# 4. Plot & file outputs
# --------------------------------------------------------------------------- #
def plot_pulsatility(result: PulsatilityResult, output_path, *, title: Optional[str] = None):
    """Render the pulsatility waveform, spectrum, amplitude map and metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    r = result
    fig = plt.figure(figsize=(13, 8.5))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.15, 1.0, 0.9],
                  hspace=0.55, wspace=0.32)

    # --- A: the continuous pulsatility waveform (headline panel) ---------- #
    axw = fig.add_subplot(gs[0, :])
    axw.plot(r.times, r.detrended, color="0.6", lw=0.8, label="tissue motion (drift removed)")
    axw.plot(r.times, r.wave, color="#1f6feb", lw=1.6, label="pulsatility wave (band-pass)")
    if len(r.peak_indices):
        axw.plot(r.times[r.peak_indices], r.wave[r.peak_indices], "v",
                 color="#d1242f", ms=6, label=f"pulses (n={r.n_pulses})")
    axw.axhline(0, color="k", lw=0.5, alpha=0.4)
    axw.set_xlabel("time (s)")
    axw.set_ylabel("tissue motion\n(px / frame)")
    axw.set_title("Bulk parenchymal pulsatility — continuous wave", fontsize=12, weight="bold")
    axw.set_xlim(r.times[0], r.times[-1])
    axw.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    axw.grid(alpha=0.2)

    # --- B: power spectrum ------------------------------------------------ #
    axs = fig.add_subplot(gs[1, :2])
    lo, hi = r.band_hz
    m = (r.freqs >= max(0, lo * 0.5)) & (r.freqs <= hi * 1.4)
    axs.plot(r.freqs[m] * 60, r.power[m] / (r.power[m].max() or 1), color="#1f6feb", lw=1.2)
    axs.axvline(r.bpm_spectral, color="#d1242f", ls="--", lw=1.2,
                label=f"{r.bpm_spectral:.1f} pulses/min")
    axs.set_xlabel("rate (pulses / min)")
    axs.set_ylabel("relative power")
    axs.set_title("Pulsation spectrum", fontsize=11)
    axs.legend(fontsize=9)
    axs.grid(alpha=0.2)

    # --- C: spatial pulsatility amplitude map ----------------------------- #
    axm = fig.add_subplot(gs[1:, 2])
    axm.imshow(r.reference, cmap="gray")
    amap = r.amplitude_map
    vmax = np.percentile(amap, 99.5) or amap.max() or 1.0
    im = axm.imshow(amap, cmap="inferno", alpha=0.6, vmin=0, vmax=vmax)
    axm.set_title("where it pulses\n(motion amplitude)", fontsize=10)
    axm.axis("off")
    cb = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    # --- D: metrics summary text ------------------------------------------ #
    axt = fig.add_subplot(gs[2, :2])
    axt.axis("off")
    axt.text(0.0, 1.0, r.summary(), va="top", ha="left", family="monospace",
             fontsize=9.5, transform=axt.transAxes)

    if title:
        fig.suptitle(title, fontsize=13, weight="bold", y=0.995)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved to %s", output_path)


def _write_waveform_csv(result: PulsatilityResult, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    peak_set = set(int(i) for i in result.peak_indices)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "time_s", "tissue_motion_px_per_frame",
                    "detrended", "pulsatility_wave", "is_pulse_peak"])
        for i in range(len(result.raw)):
            w.writerow([i, f"{result.times[i]:.5f}", f"{result.raw[i]:.6f}",
                        f"{result.detrended[i]:.6f}", f"{result.wave[i]:.6f}",
                        int(i in peak_set)])
    logger.info("waveform CSV saved to %s", path)


def _save_amplitude_map(result: PulsatilityResult, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(result.reference, cmap="gray")
    vmax = np.percentile(result.amplitude_map, 99.5) or result.amplitude_map.max() or 1.0
    im = ax.imshow(result.amplitude_map, cmap="inferno", alpha=0.6, vmin=0, vmax=vmax)
    ax.set_title("Spatial pulsatility amplitude")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="motion amplitude (px/frame)")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. File-level entry point
# --------------------------------------------------------------------------- #
def analyze_pulsatility_video(
    video_path,
    output_dir,
    *,
    method: str = "flow",
    resize_width: Optional[int] = 320,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    fps_override: Optional[float] = None,
    min_bpm: float = 30.0,
    max_bpm: float = 300.0,
    pixel_size_um: Optional[float] = None,
    make_plot: bool = True,
) -> PulsatilityResult:
    """Full video -> analysis + artifacts. Returns the :class:`PulsatilityResult`.

    Writes ``pulsatility_analysis.png`` (waveform + spectrum + map + metrics),
    ``pulsatility_waveform.csv``, ``pulsatility_amplitude_map.png`` and
    ``pulsatility_summary.txt`` into ``output_dir``.
    """
    video_path = Path(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames, fps = load_video_gray(
        video_path, resize_width=resize_width,
        max_frames=max_frames, frame_stride=frame_stride,
    )
    if fps_override:
        fps = fps_override

    result = analyze_pulsatility(
        frames, fps, method=method,
        min_bpm=min_bpm, max_bpm=max_bpm, pixel_size_um=pixel_size_um,
    )

    _write_waveform_csv(result, out / "pulsatility_waveform.csv")
    _save_amplitude_map(result, out / "pulsatility_amplitude_map.png")
    (out / "pulsatility_summary.txt").write_text(result.summary() + "\n")
    if make_plot:
        plot_pulsatility(result, out / "pulsatility_analysis.png",
                         title=video_path.name)

    logger.info("\n%s", result.summary())
    return result


# --------------------------------------------------------------------------- #
# 6. CLI
# --------------------------------------------------------------------------- #
def cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Quantify bulk parenchymal pulsatility from a brightfield/"
                    "microscopy video as a continuous wave + pulse rate.",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="input video (mp4/mov/avi/...)")
    parser.add_argument("-o", "--output", required=True,
                        help="output directory for graph, CSV and summary")
    parser.add_argument("--method", choices=("flow", "diff"), default="flow",
                        help="motion measure: optical flow (default) or frame diff")
    parser.add_argument("--resize-width", type=int, default=320,
                        help="downscale frames to this width (0 = keep full res)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="keep every Nth frame (fps scaled accordingly)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="limit number of frames analyzed")
    parser.add_argument("--fps", type=float, default=None,
                        help="override the video's reported frame rate")
    parser.add_argument("--min-bpm", type=float, default=30.0,
                        help="lower bound of pulse-rate search (pulses/min)")
    parser.add_argument("--max-bpm", type=float, default=300.0,
                        help="upper bound of pulse-rate search (pulses/min)")
    parser.add_argument("--pixel-size-um", type=float, default=None,
                        help="pixel size to report motion in um/s")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = analyze_pulsatility_video(
        args.input, args.output,
        method=args.method,
        resize_width=args.resize_width or None,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        fps_override=args.fps,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        pixel_size_um=args.pixel_size_um,
    )
    print(result.summary())
    print(f"\nArtifacts written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    cli()
