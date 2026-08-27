"""Smoothing and pressure-gradient analysis for MPRLS bioreactor pressure waveforms.

The problem
-----------
Three MPRLS pressure sensors sit on a perfused bioreactor: an **input** sensor
(closest to the pump), an **output** sensor, and a **midpoint** sensor. The flow
is *pulsatile* -- a pump drives it at roughly 100-105 bpm -- and the goal is the
**pressure gradient** between the input and output waveforms.

The raw recordings are dominated by broadband **vibration noise** (present at all
pump speeds while liquid is pushed through). The physiological signal, however,
is highly structured, and that structure is exactly what lets us smooth it
without inventing anything:

* it is **quasi-periodic** at a single fundamental ``f0`` (~1.75 Hz = 105 bpm);
* it is **multiphasic**, so the shape is carried by a *small* number of
  harmonics of ``f0`` (typically 2-4; negligible beyond ~5x f0 on the clean
  channels);
* the vibration is **broadband and stationary** -- it has no discrete tone to
  notch out, it just fills the spectrum above (and under) the harmonics.

So "meaningful smoothing based on what it likely is" = **band-limit the signal to
the pulse fundamental and its first few harmonics**, with two complementary
tools:

1. :func:`smooth_pressure` -- a zero-phase (``filtfilt``) low-pass whose cutoff
   is placed just above the highest meaningful harmonic
   (``cutoff = n_harmonics * f0``). Zero-phase filtering preserves the *timing*
   of features, which matters because the gradient is a difference of two
   channels and any phase shift would corrupt it. This keeps the full continuous
   recording, just de-noised.

2. :func:`ensemble_average` -- coherent (beat-synchronous) averaging. Because the
   signal repeats every beat, averaging many beats onto a common phase axis beats
   the *broadband* noise down by ~``sqrt(n_beats)`` (hundreds of beats -> ~15x
   noise reduction) and yields one clean **representative pulse** per channel plus
   a beat-to-beat scatter band. This is the strongest denoiser available here and
   it is the natural way to state a per-beat pressure gradient.

Before any of that the input channel needs **artifact repair**: MPRLS reads over
I2C occasionally drop out (NaN, ~0.1-2% of samples) and occasionally spike, so we
interpolate the dropouts (:func:`fill_dropouts`) and reject spikes with a Hampel
filter (:func:`hampel_filter`).

Layering (mirrors the rest of the repo)
---------------------------------------
* Core, pure-numpy functions: :func:`fill_dropouts`, :func:`hampel_filter`,
  :func:`to_uniform_grid`, :func:`estimate_fundamental`, :func:`smooth_pressure`,
  :func:`detect_beats`, :func:`ensemble_average`.
* Assembly: :func:`analyze_channel`, :func:`pressure_gradient`,
  :func:`analyze_pressure_csv`.
* CLI: :func:`cli` (installed as ``mp-pressure-gradient``).
"""
from __future__ import annotations

import csv as _csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_BPM_RANGE: Tuple[float, float] = (60.0, 180.0)  # fundamental search band
DEFAULT_N_HARMONICS: int = 4        # keep f0..4*f0 (multiphasic shape lives here)
DEFAULT_PHASE_BINS: int = 240       # samples in one ensemble-averaged beat
DEFAULT_TEMPLATE_SMOOTH_FRAC: float = 0.07  # phase-domain smoother window (cycle fraction)
DEFAULT_HAMPEL_WINDOW_S: float = 0.15   # spike-detector half-window, seconds
DEFAULT_HAMPEL_NSIGMA: float = 5.0

# Standard channel layout of the recording CSVs.
CHANNEL_COLUMNS: Dict[str, str] = {
    "input": "pressure_mmhg_1",
    "output": "pressure_mmhg_2",
    "midpoint": "pressure_mmhg_3",
}


# --------------------------------------------------------------------------- #
# 1. Artifact repair (dropouts + spikes)
# --------------------------------------------------------------------------- #
def fill_dropouts(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN dropouts in ``x``.

    MPRLS sensors periodically fail an I2C read and emit no value; in these CSVs
    that surfaces as a NaN in the pressure column (and its raw_count). The gaps
    are short (mostly single samples), so linear interpolation across them is
    faithful. Returns ``(filled, was_nan_mask)``. Leading/trailing NaNs are
    back/forward filled from the nearest valid sample.
    """
    x = np.asarray(x, dtype=float)
    mask = np.isnan(x)
    if not mask.any():
        return x.copy(), mask
    good = ~mask
    if good.sum() == 0:
        raise ValueError("channel is entirely NaN")
    idx = np.arange(len(x))
    filled = x.copy()
    filled[mask] = np.interp(idx[mask], idx[good], x[good])
    return filled, mask


def hampel_filter(
    x: np.ndarray,
    window: int,
    n_sigma: float = DEFAULT_HAMPEL_NSIGMA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace spike outliers with the local median (Hampel identifier).

    For each sample, compare it to the median of a ``+/-window`` neighbourhood; if
    it deviates by more than ``n_sigma`` robust standard deviations
    (``1.4826 * MAD``) it is judged a spike and replaced by that local median.
    This removes the isolated pressure spikes (pump/electrical transients) without
    touching the smooth pulsatile excursions, which stay close to their local
    median. Returns ``(cleaned, spike_mask)``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    window = max(1, int(window))
    if n == 0:
        return x.copy(), np.zeros(0, dtype=bool)

    k = 1.4826  # MAD -> std for a normal distribution
    pad = np.pad(x, window, mode="reflect")
    # Sliding windows of length (2*window+1) over the padded signal.
    shape = (n, 2 * window + 1)
    strides = (pad.strides[0], pad.strides[0])
    win = np.lib.stride_tricks.as_strided(pad, shape=shape, strides=strides)
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1)
    sigma = k * mad
    # Where MAD is ~0 (flat neighbourhood) fall back to a global robust sigma so
    # we do not flag every sample of a quiet stretch.
    global_sigma = k * np.median(np.abs(x - np.median(x)))
    sigma = np.where(sigma > 0, sigma, global_sigma if global_sigma > 0 else np.inf)
    spike = np.abs(x - med) > n_sigma * sigma
    cleaned = x.copy()
    cleaned[spike] = med[spike]
    return cleaned, spike


# --------------------------------------------------------------------------- #
# 2. Uniform resampling (the raw timebase is jittered)
# --------------------------------------------------------------------------- #
def to_uniform_grid(
    t: np.ndarray,
    x: np.ndarray,
    fs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resample ``(t, x)`` onto an evenly spaced grid.

    The acquisition timestamps are jittered (median ~18 ms but with occasional
    long gaps), and every spectral / filtering step assumes a uniform sample
    rate. We pick ``fs`` from the *median* sample interval (robust to the gaps)
    and linearly interpolate onto ``arange(t0, t1, 1/fs)``. Returns
    ``(t_uniform, x_uniform, fs)``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(t)
    t, x = t[order], x[order]
    if fs is None:
        dt = np.median(np.diff(t))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("cannot infer sample rate from timestamps")
        fs = 1.0 / dt
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    xu = np.interp(tu, t, x)
    return tu, xu, float(fs)


# --------------------------------------------------------------------------- #
# 3. Fundamental frequency of the pulse
# --------------------------------------------------------------------------- #
def estimate_fundamental(
    x: np.ndarray,
    fs: float,
    bpm_range: Tuple[float, float] = DEFAULT_BPM_RANGE,
) -> Tuple[float, float]:
    """Find the pulse fundamental ``f0`` (Hz) inside ``bpm_range``.

    Uses a Hann-windowed, zero-padded FFT and takes the peak within the search
    band. Also returns a spectral SNR = fraction of in-band power lying within
    +/-15% of ``f0`` (a coherence / confidence measure). The ``bpm_range`` default
    (60-180) brackets the expected 100-105 bpm with margin while rejecting slow
    drift and fast vibration.
    """
    x = np.asarray(x, dtype=float)
    s = x - x.mean()
    if s.std() > 0:
        s = s * np.hanning(len(s))
    nfft = int(2 ** np.ceil(np.log2(max(len(s) * 4, 256))))
    power = np.abs(np.fft.rfft(s, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    lo, hi = bpm_range[0] / 60.0, bpm_range[1] / 60.0
    hi = min(hi, fs / 2.0 * 0.98)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return 0.0, 0.0
    bidx = np.where(band)[0]
    f0 = float(freqs[bidx[np.argmax(power[bidx])]])
    near = (freqs >= f0 * 0.85) & (freqs <= f0 * 1.15)
    total = power[band].sum()
    snr = float(power[near].sum() / total) if total > 0 else 0.0
    return f0, snr


# --------------------------------------------------------------------------- #
# 4. The smoother: band-limit to fundamental + first few harmonics
# --------------------------------------------------------------------------- #
def smooth_pressure(
    x: np.ndarray,
    fs: float,
    f0: float,
    *,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    order: int = 4,
    highpass_hz: Optional[float] = None,
) -> np.ndarray:
    """Zero-phase band-limited smoothing of a pressure channel.

    The physiological content is the pulse fundamental ``f0`` and its first
    ``n_harmonics`` harmonics; everything above ``cutoff = n_harmonics * f0`` is
    vibration and is removed with a zero-phase Butterworth low-pass (``filtfilt``,
    so no phase distortion -- essential for a channel-to-channel gradient).

    ``highpass_hz`` optionally removes slow baseline drift below the pulse (e.g.
    ``0.5 * f0``) while keeping the pulse and its DC-free shape; leave ``None`` to
    preserve the mean level (needed for the *mean* pressure gradient).
    """
    from scipy.signal import butter, filtfilt, sosfiltfilt

    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    cutoff = n_harmonics * f0
    cutoff_n = min(cutoff / nyq, 0.99)
    if cutoff_n <= 0:
        return x.copy()

    pad = 3 * (2 * order + 1)
    if len(x) <= pad:
        return x.copy()

    if highpass_hz and highpass_hz > 0:
        lo_n = max(highpass_hz / nyq, 1e-4)
        if lo_n < cutoff_n:
            sos = butter(order, [lo_n, cutoff_n], btype="band", output="sos")
            return sosfiltfilt(sos, x)
    b, a = butter(order, cutoff_n, btype="low")
    return filtfilt(b, a, x)


# --------------------------------------------------------------------------- #
# 5. Beat detection + ensemble (beat-synchronous) averaging
# --------------------------------------------------------------------------- #
def detect_beats(
    x: np.ndarray,
    fs: float,
    f0: float,
) -> np.ndarray:
    """Return sample indices of one consistent fiducial per beat.

    A narrow zero-phase band-pass around ``f0`` isolates the fundamental so that
    peak-picking finds exactly one landmark per cycle (robust to the multiphasic
    shape, which can have several local maxima per beat). Peaks are constrained to
    be at least ``0.6/f0`` apart.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    lo_n = max(0.6 * f0 / nyq, 1e-4)
    hi_n = min(1.5 * f0 / nyq, 0.99)
    if hi_n <= lo_n or len(x) <= 3 * 5:
        return np.zeros(0, dtype=int)
    b, a = butter(2, [lo_n, hi_n], btype="band")
    narrow = filtfilt(b, a, x - x.mean())
    min_dist = max(1, int(round(fs / f0 * 0.6)))
    prom = 0.4 * np.std(narrow) if np.std(narrow) > 0 else None
    peaks, _ = find_peaks(narrow, distance=min_dist, prominence=prom)
    return peaks


@dataclass
class Ensemble:
    """Beat-synchronous average of one channel."""
    phase: np.ndarray        # 0..1 over one cycle
    mean: np.ndarray         # representative pulse
    std: np.ndarray          # beat-to-beat scatter at each phase
    sem: np.ndarray          # standard error of the mean pulse (std / sqrt(n))
    n_beats: int
    coherence: float         # fraction of in-band variance that is beat-locked


def ensemble_average(
    x: np.ndarray,
    fs: float,
    f0: float,
    *,
    beats: Optional[np.ndarray] = None,
    n_bins: int = DEFAULT_PHASE_BINS,
) -> Ensemble:
    """Beat-synchronous average of ``x`` -> one representative pulse.

    Each beat (``beats[i]`` to ``beats[i+1]``) is time-normalised onto a common
    phase axis of ``n_bins`` points and the beats are averaged. Because the
    broadband vibration is uncorrelated beat-to-beat it averages down by roughly
    ``sqrt(n_beats)``; the coherent pulse survives.

    ``beats`` defaults to fiducials detected on ``x`` itself. Averaging each
    channel on *its own* beats is deliberate here: the channels are
    phase-decorrelated (separate flow paths + external U-bend) and only ~15-25%
    beat-locked, so their peak-picked fiducials are too jittery to transfer to
    another channel -- forcing a shared fiducial smears the other channels' pulses
    to near-zero. The gradient is instead taken in the time domain (see
    :func:`pressure_gradient`).

    ``coherence`` is the fraction of the total in-band variance that is
    beat-locked, ``var(mean) / (var(mean) + mean(var_between_beats))`` -- a low
    value (here ~0.2) means most of the in-band signal is incoherent vibration and
    quantifies why averaging (not single beats) is required.
    """
    x = np.asarray(x, dtype=float)
    if beats is None:
        beats = detect_beats(x, fs, f0)
    phase = np.linspace(0.0, 1.0, n_bins, endpoint=False)
    nan = np.full(n_bins, np.nan)
    if len(beats) < 3:
        return Ensemble(phase, nan, nan, nan, 0, 0.0)

    cycles = []
    for a, b in zip(beats[:-1], beats[1:]):
        seg = x[a:b]
        if len(seg) < 4:
            continue
        src = np.linspace(0.0, 1.0, len(seg), endpoint=False)
        cycles.append(np.interp(phase, src, seg))
    if len(cycles) < 2:
        return Ensemble(phase, nan, nan, nan, 0, 0.0)
    stack = np.vstack(cycles)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    n = len(cycles)
    var_coh = float(np.var(mean))
    var_incoh = float(np.mean(std ** 2))
    coherence = var_coh / (var_coh + var_incoh) if (var_coh + var_incoh) > 0 else 0.0
    return Ensemble(phase, mean, std, std / np.sqrt(n), n, coherence)


def smooth_template(y: np.ndarray, frac: float = 0.07, poly: int = 3) -> np.ndarray:
    """Gently smooth a periodic representative pulse (phase-domain Savitzky-Golay).

    Applied to the *already ensemble-averaged* template, with wrap-around padding
    (the template is one full cycle). This removes residual un-averaged noise
    WITHOUT the overshoot/"notch" a time-domain low-pass leaves on the sharp
    systolic upstroke: the peak spans ~15% of the cycle, far wider than the
    ``frac`` (~7%) smoother window, so it is preserved while sub-window noise is
    removed. ``frac`` is the window as a fraction of the cycle; ``frac=0`` is a
    no-op.
    """
    from scipy.signal import savgol_filter

    y = np.asarray(y, dtype=float)
    n = len(y)
    if frac <= 0 or n < poly + 2 or not np.isfinite(y).all():
        return y.copy()
    win = int(round(frac * n))
    win = max(poly + 2, win)
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
    if win < poly + 2:
        return y.copy()
    yp = np.concatenate([y[-win:], y, y[:win]])
    ys = savgol_filter(yp, win, poly)
    return ys[win:win + n]


def roll_to_foot(ens: Ensemble) -> Ensemble:
    """Roll an :class:`Ensemble` so its diastolic minimum sits at phase 0.

    Purely cosmetic (a circular shift of one periodic template): it makes the
    pulse read left-to-right as upstroke -> peak -> dicrotic features -> diastole.
    Each channel is rolled to *its own* foot; because the input and output are
    separate flow-through channels joined by an external U-bend (not two ends of
    one pipe) they are phase-decorrelated, so there is no common phase origin to
    share and none is implied.
    """
    if ens.n_beats == 0 or not np.isfinite(ens.mean).all():
        return ens
    r = int(-np.argmin(ens.mean)) % len(ens.mean)
    return Ensemble(
        ens.phase, np.roll(ens.mean, r), np.roll(ens.std, r),
        np.roll(ens.sem, r), ens.n_beats, ens.coherence,
    )


# --------------------------------------------------------------------------- #
# 6. Per-channel assembly
# --------------------------------------------------------------------------- #
@dataclass
class ChannelResult:
    """Everything computed for one pressure channel."""
    name: str
    fs: float
    t: np.ndarray                 # uniform time grid (s)
    raw: np.ndarray               # on-grid, dropouts filled, spikes kept
    cleaned: np.ndarray           # + spikes removed (pre-smoothing)
    smooth: np.ndarray            # band-limited zero-phase smoothed
    f0: float                     # fundamental (Hz)
    bpm: float                    # f0 * 60
    spectral_snr: float
    n_dropouts: int
    n_spikes: int
    beats: np.ndarray             # beat fiducial indices (into t/smooth)
    phase: np.ndarray             # 0..1
    template: np.ndarray          # ensemble-averaged representative pulse
    template_std: np.ndarray      # beat-to-beat scatter (1 SD)
    template_sem: np.ndarray      # standard error of the mean pulse
    n_beats: int
    coherence: float              # fraction of in-band variance that is beat-locked
    cutoff_hz: float
    extras: dict = field(default_factory=dict)

    @property
    def mean_pressure(self) -> float:
        return float(np.mean(self.smooth))

    @property
    def pulse_amplitude(self) -> float:
        """Peak-to-peak of the representative pulse (pulse pressure)."""
        if self.n_beats == 0:
            return float("nan")
        return float(np.nanmax(self.template) - np.nanmin(self.template))

    @property
    def noise_rms_removed(self) -> float:
        """RMS of what the smoother removed (raw-with-spikes minus smooth)."""
        return float(np.sqrt(np.mean((self.raw - self.smooth) ** 2)))


def analyze_channel(
    t: np.ndarray,
    x: np.ndarray,
    name: str,
    *,
    bpm_range: Tuple[float, float] = DEFAULT_BPM_RANGE,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    hampel_nsigma: float = DEFAULT_HAMPEL_NSIGMA,
    n_bins: int = DEFAULT_PHASE_BINS,
    f0: Optional[float] = None,
    beats: Optional[np.ndarray] = None,
    template_smooth_frac: float = DEFAULT_TEMPLATE_SMOOTH_FRAC,
) -> ChannelResult:
    """Full single-channel pipeline: repair -> resample -> smooth -> ensemble.

    Pass ``f0`` to force a common fundamental across channels (recommended -- all
    sensors see the same pump). Each channel is ensemble-averaged on *its own*
    beat fiducials: the channels are phase-decorrelated (separate flow-through
    paths joined by an external U-bend), and each channel is only ~20-25%
    beat-locked, so its peak-picked fiducials are too jittery to transfer to
    another channel. ``beats`` may override the detected fiducials (used in tests).
    The template is rolled so its diastolic foot is at phase 0.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    filled, nan_mask = fill_dropouts(x)
    tu, raw, fs = to_uniform_grid(t, filled)

    hampel_win = max(1, int(round(DEFAULT_HAMPEL_WINDOW_S * fs)))
    cleaned, spike_mask = hampel_filter(raw, hampel_win, hampel_nsigma)

    _, snr = estimate_fundamental(cleaned, fs, bpm_range)
    if f0 is None:
        f0, snr = estimate_fundamental(cleaned, fs, bpm_range)
    if f0 <= 0:
        f0 = np.mean(bpm_range) / 60.0

    smooth = smooth_pressure(cleaned, fs, f0, n_harmonics=n_harmonics)
    if beats is None:
        beats = detect_beats(cleaned, fs, f0)
    # The representative pulse is ensemble-averaged from the *cleaned* signal, NOT
    # the low-passed one. Averaging hundreds of beats is itself a strong denoiser
    # (~sqrt(n_beats)) and, unlike the Butterworth low-pass, it adds no ringing --
    # low-passing before averaging puts an artefactual overshoot/"notch" on the
    # sharp systolic peak. `smooth` is kept only for the continuous trace.
    ens = roll_to_foot(ensemble_average(cleaned, fs, f0, beats=beats, n_bins=n_bins))
    template = smooth_template(ens.mean, template_smooth_frac)

    return ChannelResult(
        name=name, fs=fs, t=tu, raw=raw, cleaned=cleaned, smooth=smooth,
        f0=f0, bpm=f0 * 60.0, spectral_snr=snr,
        n_dropouts=int(nan_mask.sum()), n_spikes=int(spike_mask.sum()),
        beats=beats, phase=ens.phase, template=template, template_std=ens.std,
        template_sem=ens.sem, n_beats=ens.n_beats, coherence=ens.coherence,
        cutoff_hz=n_harmonics * f0,
    )


# --------------------------------------------------------------------------- #
# 7. Pressure gradient (input - output)
# --------------------------------------------------------------------------- #
@dataclass
class GradientResult:
    """Pressure gradient between two channels (by convention input - output)."""
    a_name: str
    b_name: str
    fs: float
    t: np.ndarray                 # common time grid
    gradient: np.ndarray          # smoothed a.smooth - b.smooth on common grid
    phase: np.ndarray             # 0..1
    gradient_template: np.ndarray # per-beat representative gradient pulse
    gradient_template_std: np.ndarray
    gradient_template_sem: np.ndarray
    mean_gradient: float          # time-averaged (DC) gradient, mmHg
    pulsatile_amplitude: float    # peak-to-peak of the representative gradient
    f0: float
    n_beats: int
    cross_coherence_f0: float = float("nan")  # MSC(input,output) at f0
    extras: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Pressure gradient  ({self.a_name} - {self.b_name})",
            f"  pulse rate            : {self.f0*60:7.1f} bpm  ({self.f0:.3f} Hz)",
            f"  mean gradient (DC)    : {self.mean_gradient:7.3f} mmHg",
            f"  pulsatile gradient    : {self.pulsatile_amplitude:7.3f} mmHg peak-to-peak",
            f"  in-out coherence @f0  : {self.cross_coherence_f0:7.2f} "
            f"(0=decorrelated, 1=locked)",
            f"  beats averaged        : {self.n_beats}",
        ]
        return "\n".join(lines)


def cross_coherence(a: ChannelResult, b: ChannelResult, f0: float) -> float:
    """Magnitude-squared coherence of two channels at the fundamental ``f0``.

    ~1 means the channels carry the same pulse in lockstep; ~0 means their
    pulsations are unrelated. For these recordings it sits around 0.4 for
    input-output: the two flow-through channels are joined only by a long external
    U-bend, so their pulses are largely *decorrelated* -- which is exactly why the
    gradient pulse is built from the instantaneous difference (below), not from a
    beat-locked subtraction of the two channels' templates.
    """
    from scipy.signal import coherence as _coh
    n = min(len(a.cleaned), len(b.cleaned))
    if n < 512:
        return float("nan")
    freqs, cxy = _coh(a.cleaned[:n], b.cleaned[:n], fs=a.fs,
                      nperseg=min(2048, n))
    return float(cxy[np.argmin(np.abs(freqs - f0))])


def pressure_gradient(
    a: ChannelResult,
    b: ChannelResult,
    *,
    n_bins: int = DEFAULT_PHASE_BINS,
) -> GradientResult:
    """Gradient ``a - b`` from two :class:`ChannelResult` (input, output).

    The instantaneous gradient ``g(t) = a.smooth - b.smooth`` is the momentary
    pressure difference (the channels share the acquisition clock and uniform
    grid). The **representative gradient pulse** is obtained by ensemble-averaging
    ``g(t)`` on *its own* beats -- a single self-referential average, not a
    subtraction of two separately-aligned templates. That distinction matters
    here: the input and output are phase-decorrelated flow-through channels
    (external U-bend), so a beat-locked subtraction would be meaningless. Averaging
    ``g(t)`` directly, the decorrelated output pulsation averages toward its mean
    (contributing to the DC drop but not the pulse), so the gradient pulse is the
    honest pulsatile pressure difference dominated by the input-side pulsation.
    """
    n = min(len(a.smooth), len(b.smooth))
    t = a.t[:n]
    grad = a.smooth[:n] - b.smooth[:n]              # continuous trace (display, mean)

    # Representative gradient pulse: ensemble-average the *cleaned* difference (no
    # low-pass) so the template carries no filter ringing (see analyze_channel).
    grad_clean = a.cleaned[:n] - b.cleaned[:n]
    beats_g = detect_beats(grad_clean, a.fs, a.f0)
    ens = roll_to_foot(ensemble_average(grad_clean, a.fs, a.f0, beats=beats_g, n_bins=n_bins))
    gradient_template = smooth_template(ens.mean, DEFAULT_TEMPLATE_SMOOTH_FRAC)
    pulsatile = (float(np.nanmax(gradient_template) - np.nanmin(gradient_template))
                 if ens.n_beats else float("nan"))

    return GradientResult(
        a_name=a.name, b_name=b.name, fs=a.fs, t=t, gradient=grad,
        phase=ens.phase, gradient_template=gradient_template,
        gradient_template_std=ens.std, gradient_template_sem=ens.sem,
        mean_gradient=float(np.mean(grad)),
        pulsatile_amplitude=pulsatile, f0=a.f0, n_beats=ens.n_beats,
        cross_coherence_f0=cross_coherence(a, b, a.f0),
    )


# --------------------------------------------------------------------------- #
# 8. File-level driver
# --------------------------------------------------------------------------- #
@dataclass
class PressureAnalysis:
    """Result of analysing one recording CSV (all channels + the gradient)."""
    source: str
    channels: Dict[str, ChannelResult]
    gradient: GradientResult

    def summary(self) -> str:
        g = self.gradient
        window = (f"  window analysed       : {g.t[0]:.1f}-{g.t[-1]:.1f} s "
                  f"({g.t[-1]-g.t[0]:.0f} s continuous)" if len(g.t) else "")
        lines = [f"MPRLS pressure analysis: {self.source}", window, ""]
        for ch in self.channels.values():
            lines.append(
                f"  {ch.name:9s}: {ch.bpm:6.1f} bpm  "
                f"mean={ch.mean_pressure:7.3f}  pulse(p2p)={ch.pulse_amplitude:6.3f} mmHg  "
                f"| beat-locked={ch.coherence*100:4.1f}%  beats={ch.n_beats}  "
                f"dropouts={ch.n_dropouts} spikes={ch.n_spikes}  "
                f"noise removed(rms)={ch.noise_rms_removed:6.3f}"
            )
        lines.append("")
        lines.append(self.gradient.summary())
        return "\n".join(lines)


def load_pressure_csv(path) -> "Tuple[np.ndarray, Dict[str, np.ndarray]]":
    """Read a recording CSV -> ``(elapsed_s, {channel_name: pressure_array})``.

    Only depends on the ``elapsed_s`` and ``pressure_mmhg_{1,2,3}`` columns; empty
    fields become NaN (dropouts). No pandas dependency in the core path.
    """
    path = Path(path)
    with open(path, newline="") as fh:
        reader = _csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")

    def col(name: str) -> np.ndarray:
        out = np.empty(len(rows), dtype=float)
        for i, r in enumerate(rows):
            v = r.get(name, "")
            out[i] = float(v) if v not in (None, "", "nan", "NaN") else np.nan
        return out

    t = col("elapsed_s")
    channels = {name: col(c) for name, c in CHANNEL_COLUMNS.items()}
    return t, channels


def select_window(
    t: np.ndarray,
    channels: Dict[str, np.ndarray],
    *,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    last_s: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Trim ``(t, channels)`` to a time window.

    ``last_s`` keeps the final ``last_s`` seconds (the intended analysis window for
    these recordings: a continuous, steady-flow stretch after the startup
    transient). ``start_s``/``end_s`` select an explicit ``[start, end]`` span
    instead. Times are in the recording's own ``elapsed_s``.
    """
    lo = t[0] if start_s is None else start_s
    hi = t[-1] if end_s is None else end_s
    if last_s is not None:
        lo = max(lo, t[-1] - last_s)
    mask = (t >= lo) & (t <= hi)
    if mask.sum() < 16:
        raise ValueError(f"time window [{lo:.1f}, {hi:.1f}] s selects too few samples")
    return t[mask], {k: v[mask] for k, v in channels.items()}


def analyze_pressure_csv(
    path,
    *,
    bpm_range: Tuple[float, float] = DEFAULT_BPM_RANGE,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    hampel_nsigma: float = DEFAULT_HAMPEL_NSIGMA,
    n_bins: int = DEFAULT_PHASE_BINS,
    gradient_from: Tuple[str, str] = ("input", "output"),
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    last_s: Optional[float] = None,
) -> PressureAnalysis:
    """Analyse one recording CSV end to end.

    A single fundamental (the median of the per-channel estimates) is imposed on
    all channels -- every sensor sees the same pump, so the pulse rate is
    physically identical and disagreement is noise. Each channel is then
    ensemble-averaged on its *own* beats (the channels are phase-decorrelated, so
    they cannot share fiducials), and the gradient pulse is built from the
    instantaneous ``input - output`` difference (see :func:`pressure_gradient`).

    ``last_s`` / ``start_s`` / ``end_s`` restrict the analysis to a time window
    (see :func:`select_window`); the recordings have a startup transient, so the
    intended use is ``last_s`` = the final steady-flow minutes.
    """
    t, raw_channels = load_pressure_csv(path)
    if start_s is not None or end_s is not None or last_s is not None:
        t, raw_channels = select_window(t, raw_channels, start_s=start_s,
                                        end_s=end_s, last_s=last_s)
    ref_name, out_name = gradient_from

    # --- shared fundamental: MEDIAN of the per-channel estimates ---
    # On a noisy record a single channel's peak can be pulled to a modulation
    # sideband (f0 +/- f_mod) -- e.g. one channel reading 114 and another 93 bpm
    # around a true 105 -- and the median rejects those while a highest-SNR or
    # summed-spectrum pick can follow the noisiest channel.
    per_channel_f0 = []
    for name, x in raw_channels.items():
        filled, _ = fill_dropouts(x)
        _, xu, fs = to_uniform_grid(t, filled)
        win = max(1, int(round(DEFAULT_HAMPEL_WINDOW_S * fs)))
        cleaned, _ = hampel_filter(xu, win, hampel_nsigma)
        f0, _ = estimate_fundamental(cleaned, fs, bpm_range)
        if f0 > 0:
            per_channel_f0.append(f0)
    f0_shared = float(np.median(per_channel_f0)) if per_channel_f0 else \
        float(np.mean(bpm_range) / 60.0)

    channels = {
        name: analyze_channel(
            t, x, name, bpm_range=bpm_range, n_harmonics=n_harmonics,
            hampel_nsigma=hampel_nsigma, n_bins=n_bins, f0=f0_shared,
        )
        for name, x in raw_channels.items()
    }
    grad = pressure_gradient(channels[ref_name], channels[out_name], n_bins=n_bins)
    return PressureAnalysis(source=str(path), channels=channels, gradient=grad)


# --------------------------------------------------------------------------- #
# 9. Plotting & CSV export
# --------------------------------------------------------------------------- #
def plot_analysis(analysis: PressureAnalysis, output_path, *, zoom_s: float = 6.0):
    """Headline figure: raw-vs-smoothed, spectra, representative pulses, gradient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chans = analysis.channels
    grad = analysis.gradient
    colors = {"input": "#1f77b4", "output": "#d62728", "midpoint": "#2ca02c"}

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.32, wspace=0.22)

    # (0,0) raw vs smoothed, input channel, zoomed
    ax = fig.add_subplot(gs[0, 0])
    ci = chans["input"]
    t0 = ci.t[0] + (ci.t[-1] - ci.t[0]) * 0.4
    m = (ci.t >= t0) & (ci.t <= t0 + zoom_s)
    ax.plot(ci.t[m], ci.raw[m], color="0.7", lw=0.7, label="raw (repaired)")
    ax.plot(ci.t[m], ci.smooth[m], color=colors["input"], lw=1.8, label="smoothed")
    ax.set_title(f"input channel: raw vs smoothed ({zoom_s:.0f}s)")
    ax.set_xlabel("s"); ax.set_ylabel("mmHg"); ax.legend(fontsize=8)

    # (0,1) all smoothed channels, same zoom
    ax = fig.add_subplot(gs[0, 1])
    for name, ch in chans.items():
        mm = (ch.t >= t0) & (ch.t <= t0 + zoom_s)
        ax.plot(ch.t[mm], ch.smooth[mm], color=colors[name], lw=1.5, label=name)
    ax.set_title("smoothed channels")
    ax.set_xlabel("s"); ax.set_ylabel("mmHg"); ax.legend(fontsize=8)

    # (1,0) spectra with harmonic markers
    ax = fig.add_subplot(gs[1, 0])
    from scipy.signal import welch
    f0 = grad.f0
    for name, ch in chans.items():
        f, P = welch(ch.cleaned - ch.cleaned.mean(), fs=ch.fs,
                     nperseg=min(8192, len(ch.cleaned)))
        ax.semilogy(f, P, color=colors[name], lw=0.8, label=name)
    for h in range(1, DEFAULT_N_HARMONICS + 1):
        ax.axvline(f0 * h, color="0.5", ls=":", lw=0.7)
    ax.axvspan(f0 * DEFAULT_N_HARMONICS, chans["input"].fs / 2, color="0.85",
               alpha=0.4, label="removed band")
    ax.set_xlim(0, chans["input"].fs / 2)
    ax.set_title(f"power spectra (f0={f0*60:.1f} bpm; dotted=harmonics kept)")
    ax.set_xlabel("Hz"); ax.set_ylabel("PSD"); ax.legend(fontsize=7)

    # (1,1) representative pulses (ensemble averages) with +/- SEM band.
    # Absolute mmHg scale (honest: the input pulse really is much larger, which is
    # why it dominates the gradient). Band is SEM -- how well the *mean* pulse is
    # known -- not the per-beat scatter.
    ax = fig.add_subplot(gs[1, 1])
    for name, ch in chans.items():
        if ch.n_beats:
            ax.plot(ch.phase, ch.template, color=colors[name], lw=1.8,
                    label=f"{name} (coh {ch.coherence*100:.0f}%, n={ch.n_beats})")
            ax.fill_between(ch.phase, ch.template - ch.template_sem,
                            ch.template + ch.template_sem, color=colors[name], alpha=0.25)
    ax.set_title("representative pulse (avg +/- SEM; each aligned to its own foot)")
    ax.set_xlabel("phase (fraction of beat)"); ax.set_ylabel("mmHg"); ax.legend(fontsize=7)

    # (2,0) instantaneous gradient, zoomed
    ax = fig.add_subplot(gs[2, 0])
    mg = (grad.t >= t0) & (grad.t <= t0 + zoom_s)
    ax.plot(grad.t[mg], grad.gradient[mg], color="#6a3d9a", lw=1.6)
    ax.axhline(grad.mean_gradient, color="k", ls="--", lw=1,
               label=f"mean = {grad.mean_gradient:.3f} mmHg")
    ax.set_title(f"instantaneous gradient ({grad.a_name} - {grad.b_name})")
    ax.set_xlabel("s"); ax.set_ylabel("mmHg"); ax.legend(fontsize=8)

    # (2,1) representative gradient pulse: SEM (dark) shows how well the mean
    # gradient pulse is known; the faint band is the per-beat SD (the vibration
    # that averaging removes).
    ax = fig.add_subplot(gs[2, 1])
    if grad.n_beats:
        gt = grad.gradient_template
        ax.fill_between(grad.phase, gt - grad.gradient_template_std,
                        gt + grad.gradient_template_std, color="#6a3d9a",
                        alpha=0.10, label="+/- SD (per beat)")
        ax.fill_between(grad.phase, gt - grad.gradient_template_sem,
                        gt + grad.gradient_template_sem, color="#6a3d9a", alpha=0.35,
                        label="+/- SEM")
        ax.plot(grad.phase, gt, color="#6a3d9a", lw=2.0,
                label=f"gradient pulse (n={grad.n_beats})")
        ax.axhline(grad.mean_gradient, color="k", ls="--", lw=1,
                   label=f"mean = {grad.mean_gradient:.3f}")
    ax.set_title(f"representative gradient pulse  "
                 f"(pulsatile p2p = {grad.pulsatile_amplitude:.3f} mmHg; "
                 f"in-out coh@f0 = {grad.cross_coherence_f0:.2f})")
    ax.set_xlabel("phase (fraction of beat)"); ax.set_ylabel("mmHg"); ax.legend(fontsize=7)

    fig.suptitle(f"MPRLS pressure smoothing & gradient — {Path(analysis.source).name}",
                 fontsize=13)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_waveform_csv(analysis: PressureAnalysis, path):
    """Per-sample smoothed channels + gradient on the common time grid."""
    grad = analysis.gradient
    chans = analysis.channels
    n = len(grad.t)
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["time_s", "input_smooth", "output_smooth", "midpoint_smooth",
                    "gradient_input_minus_output"])
        for i in range(n):
            w.writerow([
                f"{grad.t[i]:.4f}",
                f"{chans['input'].smooth[i]:.5f}",
                f"{chans['output'].smooth[i]:.5f}",
                f"{chans['midpoint'].smooth[i]:.5f}",
                f"{grad.gradient[i]:.5f}",
            ])
    return path


def write_template_csv(analysis: PressureAnalysis, path):
    """Representative (beat-averaged) pulse of each channel + the gradient pulse."""
    grad = analysis.gradient
    chans = analysis.channels
    phase = grad.phase
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["phase", "input_pulse", "input_sd", "output_pulse", "output_sd",
                    "midpoint_pulse", "midpoint_sd", "gradient_pulse", "gradient_sd"])
        for i in range(len(phase)):
            w.writerow([
                f"{phase[i]:.4f}",
                f"{chans['input'].template[i]:.5f}", f"{chans['input'].template_std[i]:.5f}",
                f"{chans['output'].template[i]:.5f}", f"{chans['output'].template_std[i]:.5f}",
                f"{chans['midpoint'].template[i]:.5f}", f"{chans['midpoint'].template_std[i]:.5f}",
                f"{grad.gradient_template[i]:.5f}", f"{grad.gradient_template_std[i]:.5f}",
            ])
    return path


# --------------------------------------------------------------------------- #
# 10. CLI
# --------------------------------------------------------------------------- #
def analyze_pressure_file(
    input_path,
    output_dir,
    *,
    bpm_range: Tuple[float, float] = DEFAULT_BPM_RANGE,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    hampel_nsigma: float = DEFAULT_HAMPEL_NSIGMA,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    last_s: Optional[float] = None,
) -> PressureAnalysis:
    """Analyse ``input_path`` and write figure + CSVs + summary to ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    analysis = analyze_pressure_csv(
        input_path, bpm_range=bpm_range, n_harmonics=n_harmonics,
        hampel_nsigma=hampel_nsigma, start_s=start_s, end_s=end_s, last_s=last_s,
    )
    plot_analysis(analysis, out / "pressure_analysis.png")
    write_waveform_csv(analysis, out / "pressure_smoothed.csv")
    write_template_csv(analysis, out / "pressure_representative_pulse.csv")
    (out / "pressure_summary.txt").write_text(analysis.summary() + "\n")
    return analysis


def cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="mp-pressure-gradient",
        description="Smooth noisy MPRLS bioreactor pressure waveforms (band-limit "
                    "to the pulse fundamental + harmonics, repair dropouts/spikes, "
                    "beat-average) and compute the input-output pressure gradient.",
    )
    p.add_argument("-i", "--input", required=True, help="recording CSV")
    p.add_argument("-o", "--output", required=True, help="output directory")
    p.add_argument("--min-bpm", type=float, default=DEFAULT_BPM_RANGE[0])
    p.add_argument("--max-bpm", type=float, default=DEFAULT_BPM_RANGE[1])
    p.add_argument("--harmonics", type=int, default=DEFAULT_N_HARMONICS,
                   help="number of pulse harmonics to keep (cutoff = harmonics * f0)")
    p.add_argument("--hampel-nsigma", type=float, default=DEFAULT_HAMPEL_NSIGMA,
                   help="spike-rejection threshold in robust SDs (higher = gentler)")
    p.add_argument("--last-seconds", type=float, default=240.0,
                   help="analyse only the final N seconds (steady-flow window; "
                        "default 240). Use 0 for the whole recording.")
    p.add_argument("--start-s", type=float, default=None,
                   help="window start in elapsed_s (overrides --last-seconds)")
    p.add_argument("--end-s", type=float, default=None,
                   help="window end in elapsed_s")
    args = p.parse_args(argv)

    explicit = args.start_s is not None or args.end_s is not None
    last_s = None if explicit or not args.last_seconds else args.last_seconds
    analysis = analyze_pressure_file(
        args.input, args.output,
        bpm_range=(args.min_bpm, args.max_bpm),
        n_harmonics=args.harmonics, hampel_nsigma=args.hampel_nsigma,
        start_s=args.start_s, end_s=args.end_s, last_s=last_s,
    )
    print(analysis.summary())
    print(f"\nwrote outputs to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
