"""Quantify bulk parenchymal pulsatility from a brightfield/microscopy video.

The tool measures how much the parenchyma (a perfused gel block, or brain) moves
with each pulse of the perfusing flow and expresses it as a **continuous
pulsatility waveform**, together with the pulse rate, a pulsatility index and a
spatial map of where the tissue pulses most.

Pipeline
--------
1. ``load_video_gray``  -- decode the video to a list of grayscale frames + fps.
2. ``build_roi_masks``  -- (optional) turn up to five user rectangles / a label
                           image into region masks on the frame.
3. ``motion_signals``   -- per-region tissue-motion signal via dense optical flow
                           (default) or frame differencing, plus a per-pixel
                           amplitude map. ``motion_signal`` is the whole-frame
                           special case.
4. ``analyze_pulsatility`` -- detrend, find the dominant pulsation frequency,
                           band-pass to a clean wave, detect pulses and compute
                           rate / variability / pulsatility index.
5. ``plot_pulsatility`` / ``plot_pulsatility_multi`` -- render the waveform,
                           spectrum, amplitude map (and per-ROI comparison).

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
    return_scale: bool = False,
):
    """Decode ``path`` to a list of float32 grayscale frames and return ``(frames, fps)``.

    Frames are optionally downscaled to ``resize_width`` (preserving aspect
    ratio). Pulsatility is a bulk, low-spatial-frequency phenomenon, so
    downscaling both speeds up optical flow and suppresses pixel noise without
    losing the signal. ``frame_stride`` keeps every Nth frame (the returned fps
    is divided accordingly) and ``max_frames`` caps how many frames are read.

    With ``return_scale=True`` the return becomes ``(frames, fps, scale)`` where
    ``scale = downscaled_width / original_width`` (1.0 if not downscaled). The
    factor lets callers map ROI coordinates given in original-video pixels onto
    the downscaled frames.
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
    orig_w = None
    scale = 1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if orig_w is None:
                orig_w = gray.shape[1]
            if resize_width and gray.shape[1] > resize_width:
                sc = resize_width / gray.shape[1]
                new_size = (resize_width, max(1, int(round(gray.shape[0] * sc))))
                gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
            frames.append(gray.astype(np.float32))
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if len(frames) < 3:
        raise ValueError(f"video has too few frames ({len(frames)}) to analyze")
    if orig_w:
        scale = frames[0].shape[1] / orig_w
    effective_fps = fps / frame_stride
    logger.info("loaded %d frames at %.3f fps (%.2fs)", len(frames), effective_fps,
                len(frames) / effective_fps)
    if return_scale:
        return frames, effective_fps, scale
    return frames, effective_fps


# --------------------------------------------------------------------------- #
# 2. Regions of interest
# --------------------------------------------------------------------------- #
MAX_ROIS = 5

# distinct, colour-blind-friendly colours for up to 5 ROIs (full field is grey)
ROI_COLORS = ["#1f6feb", "#d1242f", "#2da44e", "#bf8700", "#8250df"]


@dataclass
class Region:
    """A named region of interest as a boolean mask on the (downscaled) frame."""

    name: str
    mask: np.ndarray                      # bool, frame shape
    bbox: Tuple[int, int, int, int]       # x, y, w, h in downscaled pixels

    @property
    def npix(self) -> int:
        return int(self.mask.sum())


def parse_roi_spec(spec: str) -> Tuple[Optional[str], Tuple[float, float, float, float]]:
    """Parse ``"name=x,y,w,h"`` or ``"x,y,w,h"`` into ``(name, (x, y, w, h))``.

    Values are kept in whatever units the caller declares (pixels or fractions).
    """
    name = None
    body = spec
    if "=" in spec:
        name, body = spec.split("=", 1)
        name = name.strip() or None
    parts = [p for p in body.replace(";", ",").split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError(
            f"ROI '{spec}' must be x,y,w,h (optionally name=x,y,w,h)")
    x, y, w, h = (float(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI '{spec}' has non-positive width/height")
    return name, (x, y, w, h)


def build_roi_masks(
    frame_shape: Tuple[int, int],
    *,
    scale: float = 1.0,
    rois: Optional[Sequence[str]] = None,
    units: str = "pixel",
    roi_mask_path=None,
) -> List[Region]:
    """Turn user ROI specs into :class:`Region` masks on the downscaled frame.

    ``rois`` is a list of ``"[name=]x,y,w,h"`` rectangles. With ``units="pixel"``
    the coordinates are original-video pixels (mapped through ``scale``); with
    ``units="fraction"`` they are fractions ``0..1`` of the frame. ``roi_mask_path``
    is an alternative: a label image whose distinct non-zero values each become
    one ROI (useful for hand-drawn, non-rectangular regions). At most
    :data:`MAX_ROIS` regions are returned.
    """
    H, W = frame_shape
    regions: List[Region] = []

    if roi_mask_path is not None:
        import cv2

        lbl = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            raise IOError(f"could not read ROI mask image: {roi_mask_path}")
        if lbl.shape != (H, W):
            lbl = cv2.resize(lbl, (W, H), interpolation=cv2.INTER_NEAREST)
        values = [v for v in np.unique(lbl) if v != 0]
        # largest regions first, capped
        values.sort(key=lambda v: int((lbl == v).sum()), reverse=True)
        if len(values) > MAX_ROIS:
            logger.warning("ROI mask has %d regions; keeping the %d largest",
                           len(values), MAX_ROIS)
            values = values[:MAX_ROIS]
        for k, v in enumerate(values, 1):
            mask = lbl == v
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            regions.append(Region(f"roi{k}", mask, bbox))
        return regions

    if not rois:
        return regions
    if len(rois) > MAX_ROIS:
        raise ValueError(f"at most {MAX_ROIS} ROIs are supported (got {len(rois)})")
    if units not in ("pixel", "fraction"):
        raise ValueError("units must be 'pixel' or 'fraction'")

    for k, spec in enumerate(rois, 1):
        name, (x, y, w, h) = parse_roi_spec(spec)
        if units == "fraction":
            x, y, w, h = x * W, y * H, w * W, h * H
        else:
            x, y, w, h = x * scale, y * scale, w * scale, h * scale
        x0 = int(round(max(0, x)))
        y0 = int(round(max(0, y)))
        x1 = int(round(min(W, x + w)))
        y1 = int(round(min(H, y + h)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"ROI '{spec}' falls outside the frame")
        mask = np.zeros((H, W), dtype=bool)
        mask[y0:y1, x0:x1] = True
        regions.append(Region(name or f"roi{k}", mask, (x0, y0, x1 - x0, y1 - y0)))
    return regions


# --------------------------------------------------------------------------- #
# 3. Tissue-motion signal
# --------------------------------------------------------------------------- #
def motion_signals(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    method: str = "flow",
    smooth_sigma: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Per-mask tissue-motion signals computed in a single motion pass.

    The optical flow (or frame difference) is evaluated once per frame-pair over
    the whole frame; the mean local motion is then taken *within each mask*, so
    any number of ROIs costs almost nothing beyond the single global pass.

    Returns ``(signals, amplitude_map, reference)`` where ``signals[k]`` is the
    ``len(frames)-1`` sample waveform for ``masks[k]``, ``amplitude_map`` is the
    per-pixel temporal std of the local motion (the "where it pulses" map), and
    ``reference`` is the median frame.
    """
    import cv2

    if method not in ("flow", "diff"):
        raise ValueError("method must be 'flow' or 'diff'")

    n = len(frames)
    h, w = frames[0].shape
    mask_bool = [m.astype(bool) for m in masks]
    counts = [int(m.sum()) for m in mask_bool]
    signals = [np.zeros(n - 1, dtype=np.float64) for _ in masks]
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
        for k, mb in enumerate(mask_bool):
            signals[k][i - 1] = float(local[mb].mean()) if counts[k] else 0.0
        acc_sum += local
        acc_sq += local * local
        prev = cur

    m = n - 1
    mean_map = acc_sum / m
    amplitude_map = np.sqrt(np.maximum(acc_sq / m - mean_map ** 2, 0.0))
    reference = np.median(np.stack(frames), axis=0).astype(np.float32)
    return signals, amplitude_map, reference


def motion_signal(
    frames: Sequence[np.ndarray],
    *,
    method: str = "flow",
    smooth_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-frame tissue-motion signal + per-pixel amplitude map.

    ``method="flow"`` (default) uses dense Farneback optical flow and takes the
    mean flow *magnitude* over the field as the bulk tissue speed for that
    frame-pair -- a direct, illumination-robust measure of how fast the
    parenchyma is moving. ``method="diff"`` uses the mean absolute frame
    difference, which is faster but conflates real motion with brightness change.

    Returns ``(signal, amplitude_map, reference)``. This is the whole-frame
    special case of :func:`motion_signals`.
    """
    full = np.ones(frames[0].shape, dtype=bool)
    signals, amplitude_map, reference = motion_signals(
        frames, [full], method=method, smooth_sigma=smooth_sigma)
    return signals[0], amplitude_map, reference


# --------------------------------------------------------------------------- #
# 4. Signal analysis
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
# 5. Plot & file outputs
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


def _draw_roi_boxes(ax, regions: Sequence[Region]):
    from matplotlib.patches import Rectangle

    for k, reg in enumerate(regions):
        color = ROI_COLORS[k % len(ROI_COLORS)]
        x, y, w, h = reg.bbox
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=2))
        ax.text(x + 2, y + 2, reg.name, color="white", fontsize=8,
                va="top", ha="left",
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.85, pad=1))


def plot_pulsatility_multi(
    results,                     # dict[name -> PulsatilityResult], insertion-ordered
    regions: Sequence[Region],
    output_path,
    *,
    title: Optional[str] = None,
):
    """Compare per-ROI pulsatility: overlaid waves, spectra, ROI map and a table.

    ``results`` maps region name -> :class:`PulsatilityResult`; the first entry is
    expected to be the whole-field result. ``regions`` are the ROI boxes to draw
    (without the full-field entry).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    names = list(results.keys())
    ref = results[names[0]].reference
    band_hz = results[names[0]].band_hz

    def color_for(name, idx):
        if name in ("full field", "full_field"):
            return "0.5"
        # match the ROI box colour by region order
        for k, reg in enumerate(regions):
            if reg.name == name:
                return ROI_COLORS[k % len(ROI_COLORS)]
        return ROI_COLORS[idx % len(ROI_COLORS)]

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1.0, 0.9],
                  hspace=0.5, wspace=0.32)

    # --- A: overlaid pulsatility waves ------------------------------------ #
    axw = fig.add_subplot(gs[0, :])
    for idx, name in enumerate(names):
        r = results[name]
        c = color_for(name, idx)
        lw = 1.0 if name.startswith("full") else 1.6
        alpha = 0.6 if name.startswith("full") else 0.95
        axw.plot(r.times, r.wave, color=c, lw=lw, alpha=alpha,
                 label=f"{name} — {r.bpm_spectral:.0f} ppm")
    axw.axhline(0, color="k", lw=0.5, alpha=0.4)
    axw.set_xlabel("time (s)")
    axw.set_ylabel("tissue motion\n(px / frame)")
    axw.set_title("Per-ROI pulsatility waves", fontsize=12, weight="bold")
    axw.set_xlim(results[names[0]].times[0], results[names[0]].times[-1])
    axw.legend(loc="upper right", fontsize=8, ncol=len(names), framealpha=0.9)
    axw.grid(alpha=0.2)

    # --- B: overlaid spectra ---------------------------------------------- #
    axs = fig.add_subplot(gs[1, :2])
    lo, hi = band_hz
    for idx, name in enumerate(names):
        r = results[name]
        m = (r.freqs >= max(0, lo * 0.5)) & (r.freqs <= hi * 1.4)
        p = r.power[m] / (r.power[m].max() or 1)
        axs.plot(r.freqs[m] * 60, p, color=color_for(name, idx),
                 lw=1.3 if not name.startswith("full") else 0.9,
                 alpha=0.6 if name.startswith("full") else 0.95, label=name)
    axs.set_xlabel("rate (pulses / min)")
    axs.set_ylabel("relative power")
    axs.set_title("Per-ROI spectra", fontsize=11)
    axs.grid(alpha=0.2)

    # --- C: reference frame with ROI boxes -------------------------------- #
    axm = fig.add_subplot(gs[1:, 2])
    axm.imshow(ref, cmap="gray")
    _draw_roi_boxes(axm, regions)
    axm.set_title("ROIs", fontsize=10)
    axm.axis("off")

    # --- D: metrics table ------------------------------------------------- #
    axt = fig.add_subplot(gs[2, :2])
    axt.axis("off")
    col = ["region", "rate (ppm)", "rate CV %", "PI", "amp RMS", "SNR %"]
    rows = []
    for name in names:
        r = results[name]
        rows.append([
            name, f"{r.bpm_spectral:.1f}",
            f"{r.rate_cv*100:.1f}" if r.rate_cv == r.rate_cv else "—",
            f"{r.pulsatility_index:.2f}", f"{r.amplitude_rms:.4f}",
            f"{r.spectral_snr*100:.0f}",
        ])
    tbl = axt.table(cellText=rows, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for idx, name in enumerate(names):
        tbl[(idx + 1, 0)].get_text().set_color(color_for(name, idx))

    if title:
        fig.suptitle(title, fontsize=13, weight="bold", y=0.995)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("multi-ROI plot saved to %s", output_path)


def _write_roi_csv(results, path):
    """One CSV with a time column plus the pulsatility wave of every region."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    times = results[names[0]].times
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "time_s"] + [f"wave[{n}]" for n in names]
                   + [f"motion[{n}]" for n in names])
        for i in range(len(times)):
            row = [i, f"{times[i]:.5f}"]
            row += [f"{results[n].wave[i]:.6f}" for n in names]
            row += [f"{results[n].raw[i]:.6f}" for n in names]
            w.writerow(row)
    logger.info("per-ROI waveform CSV saved to %s", path)


def _roi_summary_text(results) -> str:
    lines = ["Per-ROI pulsatility", "=" * 60,
             f"{'region':<14}{'rate(ppm)':>11}{'CV%':>7}{'PI':>7}"
             f"{'ampRMS':>10}{'SNR%':>7}{'pulses':>8}"]
    for name, r in results.items():
        cv = f"{r.rate_cv*100:5.1f}" if r.rate_cv == r.rate_cv else "  -  "
        lines.append(f"{name:<14}{r.bpm_spectral:>11.1f}{cv:>7}"
                     f"{r.pulsatility_index:>7.2f}{r.amplitude_rms:>10.4f}"
                     f"{r.spectral_snr*100:>7.0f}{r.n_pulses:>8}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. File-level entry point
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
    rois: Optional[Sequence[str]] = None,
    roi_units: str = "pixel",
    roi_mask=None,
    make_plot: bool = True,
) -> PulsatilityResult:
    """Full video -> analysis + artifacts. Returns the whole-field :class:`PulsatilityResult`.

    Writes ``pulsatility_analysis.png`` (waveform + spectrum + map + metrics),
    ``pulsatility_waveform.csv``, ``pulsatility_amplitude_map.png`` and
    ``pulsatility_summary.txt`` into ``output_dir``.

    Pass ``rois`` (up to :data:`MAX_ROIS` ``"[name=]x,y,w,h"`` rectangles, in
    original-video pixels unless ``roi_units="fraction"``) and/or a ``roi_mask``
    label image to also measure pulsatility inside specific regions. When regions
    are given, the whole field plus every ROI are analysed in one motion pass and
    an extra ``pulsatility_rois.png``, ``pulsatility_roi_waveforms.csv`` and
    ``pulsatility_roi_summary.txt`` are written; each ROI's
    :class:`PulsatilityResult` is also attached to
    ``result.extras['roi_results']``.
    """
    video_path = Path(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames, fps, scale = load_video_gray(
        video_path, resize_width=resize_width,
        max_frames=max_frames, frame_stride=frame_stride, return_scale=True,
    )
    if fps_override:
        fps = fps_override

    regions = build_roi_masks(
        frames[0].shape, scale=scale, rois=rois, units=roi_units,
        roi_mask_path=roi_mask,
    )

    if regions:
        full = np.ones(frames[0].shape, dtype=bool)
        masks = [full] + [r.mask for r in regions]
        names = ["full field"] + [r.name for r in regions]
        signals, amap, ref = motion_signals(frames, masks, method=method)
        results = {}
        for name, sig in zip(names, signals):
            results[name] = analyze_pulsatility(
                frames, fps, method=method, min_bpm=min_bpm, max_bpm=max_bpm,
                pixel_size_um=pixel_size_um, precomputed=(sig, amap, ref),
            )
        result = results["full field"]
        result.extras["roi_results"] = results

        # whole-field artifacts (unchanged names) ...
        _write_waveform_csv(result, out / "pulsatility_waveform.csv")
        _save_amplitude_map(result, out / "pulsatility_amplitude_map.png")
        (out / "pulsatility_summary.txt").write_text(result.summary() + "\n")
        if make_plot:
            plot_pulsatility(result, out / "pulsatility_analysis.png",
                             title=video_path.name)
        # ... plus the ROI comparison artifacts
        _write_roi_csv(results, out / "pulsatility_roi_waveforms.csv")
        (out / "pulsatility_roi_summary.txt").write_text(_roi_summary_text(results) + "\n")
        if make_plot:
            plot_pulsatility_multi(results, regions, out / "pulsatility_rois.png",
                                   title=f"{video_path.name} — {len(regions)} ROI(s)")
        logger.info("\n%s", _roi_summary_text(results))
        return result

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
# 7. CLI
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
    parser.add_argument("--roi", action="append", dest="rois", metavar="[NAME=]X,Y,W,H",
                        help="region of interest rectangle (repeatable, up to 5). "
                             "Coordinates are original-video pixels unless "
                             "--roi-units fraction. Optional NAME= prefix.")
    parser.add_argument("--roi-units", choices=("pixel", "fraction"), default="pixel",
                        help="interpret --roi coordinates as original pixels (default) "
                             "or fractions 0..1 of the frame")
    parser.add_argument("--roi-mask", default=None,
                        help="label image whose distinct non-zero values each "
                             "define an ROI (alternative to --roi; up to 5 regions)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.rois and len(args.rois) > MAX_ROIS:
        parser.error(f"at most {MAX_ROIS} --roi rectangles are supported "
                     f"(got {len(args.rois)})")

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
        rois=args.rois,
        roi_units=args.roi_units,
        roi_mask=args.roi_mask,
    )
    roi_results = result.extras.get("roi_results")
    if roi_results:
        print(_roi_summary_text(roi_results))
    else:
        print(result.summary())
    print(f"\nArtifacts written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    cli()
