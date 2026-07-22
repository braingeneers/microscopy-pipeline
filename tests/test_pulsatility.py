"""Smoke tests for the pulsatility quantification tool.

Two levels, both on synthetic data (no external video needed):

* the analysis core (`analyze_pulsatility`) is fed a clean signal of known
  frequency and must recover the rate;
* the motion front-end (`motion_signal`, optical flow) is exercised on a
  synthetic translating texture whose bulk speed pulses at a known rate, then
  the whole pipeline (incl. plotting + CSV) is run end-to-end.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndshift

from pulsatility import (
    analyze_pulsatility,
    analyze_pulsatility_video,
    motion_signal,
    plot_pulsatility,
)


def _pulsing_texture_frames(fps=30.0, dur=6.0, f=1.0, seed=0):
    """Frames of a random texture translating with a once-per-cycle speed pulse.

    Velocity stays positive (V0 > V1) and is modulated at frequency ``f``, so the
    optical-flow *speed* has its fundamental at ``f`` -- mimicking one tissue
    motion burst per perfusion pulse.
    """
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.standard_normal((96, 96)).astype(np.float32), 1.5)
    tex = (tex - tex.min()) / np.ptp(tex) * 200 + 20
    n = int(fps * dur)
    t = np.arange(n) / fps
    V0, V1 = 0.9, 0.7
    disp = V0 * t + (V1 / (2 * np.pi * f)) * np.sin(2 * np.pi * f * t)
    return [ndshift(tex, (0.0, float(d)), order=1, mode="grid-wrap") for d in disp], fps


def test_core_recovers_known_rate():
    """analyze_pulsatility on a clean 90-bpm signal recovers ~90 bpm."""
    fps, f = 30.0, 1.5  # 90 pulses/min
    t = np.arange(300) / fps
    rng = np.random.default_rng(1)
    raw = 1.0 + 0.5 * np.sin(2 * np.pi * f * t) + 0.03 * rng.standard_normal(t.size)
    dummy = np.zeros((4, 4), np.float32)
    r = analyze_pulsatility([dummy] * (t.size + 1), fps, min_bpm=30, max_bpm=240,
                            precomputed=(raw, dummy, dummy))
    assert abs(r.bpm_spectral - 90.0) < 6.0
    assert abs(r.bpm_peaks - 90.0) < 8.0
    assert r.spectral_snr > 0.5
    assert r.pulsatility_index > 0
    assert "pulses/min" in r.summary()


def test_motion_pipeline_on_synthetic_frames():
    """Optical-flow motion signal + analysis recover a 60-bpm translating texture."""
    frames, fps = _pulsing_texture_frames(f=1.0)  # 60 pulses/min
    sig, amap, ref = motion_signal(frames, method="flow")
    assert sig.shape == (len(frames) - 1,)
    assert amap.shape == frames[0].shape
    r = analyze_pulsatility(frames, fps, min_bpm=30, max_bpm=180,
                            precomputed=(sig, amap, ref))
    assert abs(r.bpm_spectral - 60.0) < 8.0
    assert r.n_pulses >= 3


def test_diff_method_runs():
    """The cheaper frame-difference method also produces a usable signal."""
    frames, fps = _pulsing_texture_frames(f=1.0)
    sig, amap, ref = motion_signal(frames, method="diff")
    assert sig.shape == (len(frames) - 1,)
    r = analyze_pulsatility(frames, fps, min_bpm=30, max_bpm=180,
                            precomputed=(sig, amap, ref))
    assert r.dominant_hz > 0


def test_end_to_end_writes_artifacts(tmp_path):
    """analyze_pulsatility_video (via a written mp4) emits all four artifacts.

    Skipped if the environment has no MP4 encoder/decoder available.
    """
    import cv2

    frames, fps = _pulsing_texture_frames(f=1.0)
    video = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(video), fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        pytest.skip("no MP4 encoder available in this environment")
    for fr in frames:
        g = np.clip(fr, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    writer.release()
    if not video.exists() or video.stat().st_size == 0:
        pytest.skip("MP4 writer produced no file")

    out = tmp_path / "results"
    try:
        r = analyze_pulsatility_video(video, out, resize_width=96,
                                      min_bpm=30, max_bpm=180)
    except (IOError, ValueError) as exc:
        pytest.skip(f"MP4 decode unavailable: {exc}")

    for name in ("pulsatility_analysis.png", "pulsatility_waveform.csv",
                 "pulsatility_amplitude_map.png", "pulsatility_summary.txt"):
        assert (out / name).exists(), name
    assert r.dominant_hz > 0


def test_plot_only(tmp_path):
    """plot_pulsatility renders a figure without raising."""
    frames, fps = _pulsing_texture_frames(f=1.0, dur=4.0)
    sig, amap, ref = motion_signal(frames, method="diff")
    r = analyze_pulsatility(frames, fps, precomputed=(sig, amap, ref))
    out = tmp_path / "plot.png"
    plot_pulsatility(r, out, title="synthetic")
    assert out.exists() and out.stat().st_size > 0
