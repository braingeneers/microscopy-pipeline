"""Tests for the MPRLS pressure smoothing / gradient tool (``mprls_pressure``).

All synthetic (no external CSV needed). A helper builds a *multiphasic* pulse at
a known rate, buried in broadband vibration noise, with the same warts as the real
recordings -- jittered timestamps, NaN dropouts and spike outliers -- so the
artifact-repair, smoothing, ensemble-averaging and gradient paths are all
exercised against a known ground truth.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from mprls_pressure import (
    ChannelResult,
    GradientResult,
    analyze_channel,
    analyze_pressure_csv,
    analyze_pressure_file,
    cross_coherence,
    detect_beats,
    ensemble_average,
    estimate_fundamental,
    fill_dropouts,
    hampel_filter,
    pressure_gradient,
    smooth_pressure,
    to_uniform_grid,
)
from mprls_pressure.pressure import CHANNEL_COLUMNS, smooth_template


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def multiphasic_pulse(phase, harmonics=(1.0, 0.5, 0.35, 0.15), phases=(0, 0.4, 0.8, 1.2)):
    """A deterministic multiphasic waveform on ``phase`` in [0,1)."""
    w = np.zeros_like(phase)
    for k, (amp, ph) in enumerate(zip(harmonics, phases), start=1):
        w += amp * np.cos(2 * np.pi * k * phase - ph)
    return w


def synth_channel(fs=55.0, dur=120.0, bpm=105.0, amp=5.0, noise=3.0,
                  seed=0, dc=0.0, phase_lag=0.0):
    """Uniformly-sampled multiphasic pulse train + broadband noise.

    Returns ``(t, x)``. ``noise`` is the SD of white (broadband) vibration added
    on top -- large ``noise`` mimics the real recordings where the pulse is only
    ~20% of the in-band variance.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs
    f0 = bpm / 60.0
    ph = (t * f0 + phase_lag) % 1.0
    x = dc + amp * multiphasic_pulse(ph) + noise * rng.standard_normal(n)
    return t, x


def jitter_and_damage(t, x, drop_frac=0.01, spike_frac=0.003, jitter=0.15, seed=1):
    """Add timestamp jitter, NaN dropouts and spike outliers (like the real data)."""
    rng = np.random.default_rng(seed)
    dt = np.median(np.diff(t))
    tj = t + rng.uniform(-jitter, jitter, size=len(t)) * dt
    tj = np.maximum.accumulate(tj)  # keep monotonic
    xd = x.copy()
    drop = rng.random(len(x)) < drop_frac
    xd[drop] = np.nan
    spike = rng.random(len(x)) < spike_frac
    xd[spike] += rng.choice([-1, 1], size=spike.sum()) * (8 * np.std(x))
    return tj, xd


# --------------------------------------------------------------------------- #
# Artifact repair
# --------------------------------------------------------------------------- #
def test_fill_dropouts_interpolates():
    x = np.linspace(0, 10, 200)
    x[50:53] = np.nan
    x[0] = np.nan  # leading
    filled, mask = fill_dropouts(x)
    assert not np.isnan(filled).any()
    assert mask.sum() == 4
    # interpolated interior values sit on the underlying line
    assert filled[51] == pytest.approx(np.linspace(0, 10, 200)[51], abs=1e-6)


def test_hampel_removes_spikes_keeps_signal():
    t = np.arange(1000) / 55.0
    clean = 5 * np.sin(2 * np.pi * 1.75 * t)
    x = clean.copy()
    x[100] += 50
    x[500] -= 40
    cleaned, spike = hampel_filter(x, window=8, n_sigma=5.0)
    assert spike[100] and spike[500]
    # the spikes are gone
    assert abs(cleaned[100] - clean[100]) < 5
    assert abs(cleaned[500] - clean[500]) < 5
    # the untouched signal is essentially preserved
    ok = ~spike
    assert np.corrcoef(cleaned[ok], clean[ok])[0, 1] > 0.999


def test_to_uniform_grid_from_jitter():
    t, x = synth_channel(dur=30, seed=3)
    tj, _ = jitter_and_damage(t, x, drop_frac=0, spike_frac=0)
    tu, xu, fs = to_uniform_grid(tj, x)
    assert 50 < fs < 60
    assert np.allclose(np.diff(tu), 1.0 / fs)


# --------------------------------------------------------------------------- #
# Fundamental & smoothing
# --------------------------------------------------------------------------- #
def test_estimate_fundamental_recovers_rate():
    t, x = synth_channel(bpm=105.0, noise=2.0, seed=4)
    f0, snr = estimate_fundamental(x, fs=1 / (t[1] - t[0]))
    assert f0 * 60 == pytest.approx(105.0, abs=2.0)
    assert 0.0 < snr <= 1.0


def test_smoothing_reduces_noise_and_preserves_shape():
    fs = 55.0
    t, clean = synth_channel(fs=fs, noise=0.0, seed=5)          # noise-free truth
    _, noisy = synth_channel(fs=fs, noise=4.0, seed=5)          # same pulse + noise
    f0 = 105.0 / 60.0
    sm = smooth_pressure(noisy, fs, f0, n_harmonics=4)
    # Noise power is substantially reduced. Low-passing white noise to the cutoff
    # (4*f0 ~ 7 Hz, ~25% of Nyquist) keeps ~1/4 of its variance, so the residual
    # error vs the truth should drop by roughly 3-4x.
    assert np.var(noisy - clean) > 3 * np.var(sm - clean)
    # ... while the pulse shape is preserved: smoothing moves the signal markedly
    # closer to the noise-free truth than the raw signal was.
    corr_noisy = np.corrcoef(noisy, clean)[0, 1]
    corr_sm = np.corrcoef(sm, clean)[0, 1]
    assert corr_sm > corr_noisy
    assert corr_sm > 0.85


def test_smoothing_lowpass_removes_high_frequency():
    fs = 55.0
    t = np.arange(int(fs * 60)) / fs
    f0 = 1.75
    pulse = np.sin(2 * np.pi * f0 * t)
    hf = 0.8 * np.sin(2 * np.pi * 18.0 * t)   # well above 4*f0 ~ 7 Hz
    sm = smooth_pressure(pulse + hf, fs, f0, n_harmonics=4)
    # the 18 Hz component is strongly attenuated
    assert np.corrcoef(sm, pulse)[0, 1] > 0.98
    assert np.var(sm - pulse) < 0.05


# --------------------------------------------------------------------------- #
# Beats & ensemble averaging
# --------------------------------------------------------------------------- #
def test_detect_beats_counts_pulses():
    dur, bpm = 60.0, 105.0
    t, x = synth_channel(dur=dur, bpm=bpm, noise=2.0, seed=6)
    beats = detect_beats(x, fs=1 / (t[1] - t[0]), f0=bpm / 60.0)
    expected = dur * bpm / 60.0
    assert abs(len(beats) - expected) < 0.1 * expected


def test_ensemble_average_beats_down_noise():
    fs, bpm = 55.0, 105.0
    t, clean = synth_channel(fs=fs, dur=180, bpm=bpm, noise=0.0, seed=7)
    _, noisy = synth_channel(fs=fs, dur=180, bpm=bpm, noise=5.0, seed=7)
    f0 = bpm / 60.0
    beats = detect_beats(noisy, fs, f0)
    ens = ensemble_average(noisy, fs, f0, beats=beats)
    assert ens.n_beats > 250
    # the averaged pulse is well-determined: SEM is much smaller than per-beat SD
    assert np.nanmean(ens.sem) < 0.25 * np.nanmean(ens.std)
    # coherence is a fraction in (0, 1); with this much noise it is modest
    assert 0.0 < ens.coherence < 1.0
    # template p2p is close to the noise-free pulse's p2p
    clean_p2p = clean.max() - clean.min()
    assert (np.nanmax(ens.mean) - np.nanmin(ens.mean)) == pytest.approx(clean_p2p, rel=0.3)


def test_smooth_template_denoises_without_notching_the_peak():
    """Phase-domain template smoothing removes noise but does not carve a notch
    into a smooth systolic peak (the failure mode a time-domain low-pass has)."""
    phase = np.linspace(0, 1, 240, endpoint=False)
    # a smooth asymmetric single-peak pulse (fast up, slow down), peak at ~0.35
    clean = np.exp(-((phase - 0.35) ** 2) / (2 * 0.09 ** 2))
    rng = np.random.default_rng(0)
    noisy = clean + 0.05 * rng.standard_normal(phase.size)
    sm = smooth_template(noisy, frac=0.07)
    # noise reduced
    assert np.var(noisy - clean) > 3 * np.var(sm - clean)
    # the peak is a single maximum, not split by a notch: the smoothed value at
    # the true peak is >= its neighbours a few bins away
    pk = int(np.argmax(clean))
    assert sm[pk] >= sm[pk - 8] and sm[pk] >= sm[(pk + 8) % 240]
    # frac=0 is a no-op
    assert np.allclose(smooth_template(noisy, frac=0.0), noisy)


def test_ensemble_average_too_few_beats_is_nan():
    fs = 55.0
    t, x = synth_channel(fs=fs, dur=1.0, noise=1.0, seed=8)  # ~1.75 beats
    ens = ensemble_average(x, fs, 1.75, beats=np.array([1, 2]))
    assert ens.n_beats == 0
    assert np.isnan(ens.mean).all()


# --------------------------------------------------------------------------- #
# Cross-coherence
# --------------------------------------------------------------------------- #
def test_cross_coherence_high_for_shared_low_for_independent():
    fs, bpm = 55.0, 105.0
    t, a = synth_channel(fs=fs, dur=120, bpm=bpm, noise=1.0, seed=10)
    # b shares the pulse (shifted) -> should be coherent
    _, b_shared = synth_channel(fs=fs, dur=120, bpm=bpm, noise=1.0, seed=11, phase_lag=0.2)
    # c is independent noise at the same rate but different realisation, huge noise
    _, c_indep = synth_channel(fs=fs, dur=120, bpm=bpm, amp=0.0, noise=3.0, seed=12)

    def mk(x):
        return ChannelResult(
            name="x", fs=fs, t=t, raw=x, cleaned=x, smooth=x, f0=bpm / 60.0,
            bpm=bpm, spectral_snr=0.0, n_dropouts=0, n_spikes=0,
            beats=np.array([]), phase=np.array([]), template=np.array([]),
            template_std=np.array([]), template_sem=np.array([]), n_beats=0,
            coherence=0.0, cutoff_hz=7.0,
        )
    f0 = bpm / 60.0
    coh_shared = cross_coherence(mk(a), mk(b_shared), f0)
    coh_indep = cross_coherence(mk(a), mk(c_indep), f0)
    assert coh_shared > coh_indep
    assert coh_shared > 0.5


# --------------------------------------------------------------------------- #
# Channel + gradient assembly
# --------------------------------------------------------------------------- #
def test_analyze_channel_end_to_end():
    t, x = synth_channel(bpm=105.0, amp=5.0, noise=3.0, dc=2.0, seed=13)
    tj, xd = jitter_and_damage(t, x)
    r = analyze_channel(tj, xd, "input")
    assert isinstance(r, ChannelResult)
    assert r.bpm == pytest.approx(105.0, abs=3.0)
    assert r.n_dropouts > 0 and r.n_spikes > 0
    assert r.mean_pressure == pytest.approx(2.0, abs=1.0)
    assert r.template.shape == r.phase.shape
    # template is foot-aligned: its minimum is at (or very near) phase 0
    assert int(np.argmin(r.template)) < 5 or int(np.argmin(r.template)) > len(r.template) - 5


def test_pressure_gradient_recovers_mean_offset():
    fs, bpm = 55.0, 105.0
    # input sits 4 mmHg above output on average
    t, xin = synth_channel(fs=fs, dur=120, bpm=bpm, amp=5.0, noise=3.0, dc=4.0, seed=14)
    _, xout = synth_channel(fs=fs, dur=120, bpm=bpm, amp=1.0, noise=1.0, dc=0.0, seed=15)
    a = analyze_channel(t, xin, "input")
    b = analyze_channel(t, xout, "output")
    grad = pressure_gradient(a, b)
    assert isinstance(grad, GradientResult)
    assert grad.mean_gradient == pytest.approx(4.0, abs=1.0)
    assert grad.pulsatile_amplitude > 0
    assert grad.n_beats > 150


# --------------------------------------------------------------------------- #
# File driver
# --------------------------------------------------------------------------- #
def _write_csv(path, t, chans):
    import csv
    cols = list(CHANNEL_COLUMNS.values())
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "elapsed_s", *cols, "raw_count_1", "raw_count_2",
                    "raw_count_3", "event"])
        for i in range(len(t)):
            vals = ["" if np.isnan(chans[c][i]) else f"{chans[c][i]:.4f}" for c in cols]
            w.writerow(["2026-01-01T00:00:00", f"{t[i]:.3f}", *vals, "0", "0", "0", ""])


def test_analyze_pressure_csv_and_file(tmp_path):
    fs, bpm = 55.0, 105.0
    t, xin = synth_channel(fs=fs, dur=90, bpm=bpm, amp=6.0, noise=3.0, dc=5.0, seed=20)
    _, xmid = synth_channel(fs=fs, dur=90, bpm=bpm, amp=3.0, noise=2.0, dc=2.0, seed=21)
    _, xout = synth_channel(fs=fs, dur=90, bpm=bpm, amp=1.0, noise=1.0, dc=0.0, seed=22)
    tj, xin = jitter_and_damage(t, xin, seed=30)
    _, xmid = jitter_and_damage(t, xmid, seed=31)
    _, xout = jitter_and_damage(t, xout, seed=32)
    csv_path = tmp_path / "rec.csv"
    _write_csv(csv_path, tj, {CHANNEL_COLUMNS["input"]: xin,
                              CHANNEL_COLUMNS["output"]: xout,
                              CHANNEL_COLUMNS["midpoint"]: xmid})

    analysis = analyze_pressure_csv(csv_path)
    assert set(analysis.channels) == {"input", "output", "midpoint"}
    assert analysis.gradient.f0 * 60 == pytest.approx(105.0, abs=3.0)
    assert analysis.gradient.mean_gradient == pytest.approx(5.0, abs=1.5)
    assert "bpm" in analysis.summary()

    # full file driver writes the figure + CSVs + summary
    out = tmp_path / "results"
    analyze_pressure_file(csv_path, out)
    for fname in ["pressure_analysis.png", "pressure_smoothed.csv",
                  "pressure_representative_pulse.csv", "pressure_summary.txt"]:
        assert (out / fname).exists() and (out / fname).stat().st_size > 0
