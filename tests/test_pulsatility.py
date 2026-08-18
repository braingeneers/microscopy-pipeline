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
    MAX_ROIS,
    analyze_pulsatility,
    analyze_pulsatility_video,
    build_roi_masks,
    decompose_respiration,
    load_video_gray,
    motion_signal,
    motion_signals,
    plot_breathing_decomposition,
    plot_pulsatility,
    plot_pulsatility_comparison,
    stabilize_frames,
)
from pulsatility.pulsatility import parse_roi_spec


def _write_synthetic_mp4(path, frames, fps):
    """Write frames to an mp4; return True on success, False if no encoder."""
    import cv2

    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h), isColor=True)
    if not writer.isOpened():
        return False
    for fr in frames:
        writer.write(cv2.cvtColor(np.clip(fr, 0, 255).astype(np.uint8),
                                  cv2.COLOR_GRAY2BGR))
    writer.release()
    return path.exists() and path.stat().st_size > 0


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


# --------------------------------------------------------------------------- #
# ROI support
# --------------------------------------------------------------------------- #
def test_parse_roi_spec():
    assert parse_roi_spec("10,20,30,40") == (None, (10, 20, 30, 40))
    assert parse_roi_spec("vessel=1,2,3,4") == ("vessel", (1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError):
        parse_roi_spec("1,2,3")            # too few
    with pytest.raises(ValueError):
        parse_roi_spec("1,2,0,4")          # zero width


def test_build_roi_masks_pixel_and_fraction():
    # frame 100x200 (H,W); scale 0.5 means the original video was 400 wide
    regions = build_roi_masks((100, 200), scale=0.5,
                              rois=["a=20,40,80,100"], units="pixel")
    assert len(regions) == 1 and regions[0].name == "a"
    # 20,40,80,100 original px * 0.5 -> x=10,y=20,w=40,h=50
    assert regions[0].bbox == (10, 20, 40, 50)
    assert regions[0].npix == 40 * 50

    frac = build_roi_masks((100, 200), rois=["0.5,0.5,0.25,0.25"], units="fraction")
    # x=100,y=50,w=50,h=25
    assert frac[0].bbox == (100, 50, 50, 25)


def test_build_roi_masks_caps_and_clips():
    with pytest.raises(ValueError):
        build_roi_masks((100, 100), rois=[f"{i},0,10,10" for i in range(MAX_ROIS + 1)])
    # a rectangle running off the frame is clipped, not rejected
    reg = build_roi_masks((50, 50), rois=["40,40,100,100"], units="pixel")[0]
    assert reg.bbox == (40, 40, 10, 10)


def test_build_roi_masks_from_label_image(tmp_path):
    import cv2

    lbl = np.zeros((60, 80), np.uint8)
    lbl[5:25, 5:35] = 1
    lbl[30:55, 40:75] = 2
    p = tmp_path / "labels.png"
    cv2.imwrite(str(p), lbl)
    regions = build_roi_masks((60, 80), roi_mask_path=p)
    assert len(regions) == 2
    assert all(r.npix > 0 for r in regions)


def test_motion_signals_matches_single_for_full_mask():
    frames, _ = _pulsing_texture_frames(f=1.0, dur=3.0)
    full = np.ones(frames[0].shape, bool)
    sigs, amap, ref = motion_signals(frames, [full], method="diff")
    single, amap2, _ = motion_signal(frames, method="diff")
    assert sigs[0].shape == single.shape
    assert np.allclose(sigs[0], single)
    assert amap.shape == frames[0].shape


def test_end_to_end_with_rois_writes_roi_artifacts(tmp_path):
    """analyze_pulsatility_video with ROIs emits the ROI comparison artifacts."""
    import cv2

    frames, fps = _pulsing_texture_frames(f=1.0)
    video = tmp_path / "roi.mp4"
    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h), isColor=True)
    if not writer.isOpened():
        pytest.skip("no MP4 encoder available")
    for fr in frames:
        writer.write(cv2.cvtColor(np.clip(fr, 0, 255).astype(np.uint8),
                                  cv2.COLOR_GRAY2BGR))
    writer.release()
    if not video.exists() or video.stat().st_size == 0:
        pytest.skip("MP4 writer produced no file")

    out = tmp_path / "res"
    try:
        r = analyze_pulsatility_video(
            video, out, resize_width=0, min_bpm=30, max_bpm=180,
            rois=["left=0,0,40,96", "right=56,0,40,96"], roi_units="pixel",
        )
    except (IOError, ValueError) as exc:
        pytest.skip(f"MP4 decode unavailable: {exc}")

    roi_results = r.extras.get("roi_results")
    assert roi_results is not None
    assert set(roi_results) == {"full field", "left", "right"}
    for name in ("pulsatility_rois.png", "pulsatility_roi_waveforms.csv",
                 "pulsatility_roi_summary.txt", "pulsatility_analysis.png"):
        assert (out / name).exists(), name


def test_frame_stride_grab_path(tmp_path):
    """load_video_gray with frame_stride skips-decodes and scales fps down."""
    frames, fps = _pulsing_texture_frames(f=1.0, dur=4.0)  # 120 frames @30fps
    video = tmp_path / "stride.mp4"
    if not _write_synthetic_mp4(video, frames, fps):
        pytest.skip("no MP4 encoder available")
    got, got_fps = load_video_gray(video, resize_width=0, frame_stride=3)
    assert abs(got_fps - fps / 3) < 1e-6
    assert abs(len(got) - len(frames) / 3) <= 2


def test_plot_comparison(tmp_path):
    """plot_pulsatility_comparison renders across two videos without raising."""
    fa, fps = _pulsing_texture_frames(f=1.0, dur=4.0)
    fb, _ = _pulsing_texture_frames(f=1.5, dur=4.0, seed=3)
    ra = analyze_pulsatility(fa, fps, precomputed=motion_signal(fa, method="diff"))
    rb = analyze_pulsatility(fb, fps, precomputed=motion_signal(fb, method="diff"))
    out = tmp_path / "cmp.png"
    plot_pulsatility_comparison({"A": ra, "B": rb}, out, title="A vs B")
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Stabilization
# --------------------------------------------------------------------------- #
def test_stabilize_reduces_global_shake():
    """stabilize_frames removes injected global jitter (frame-to-frame motion drops)."""
    rng = np.random.default_rng(5)
    tex = gaussian_filter(rng.standard_normal((80, 100)).astype(np.float32), 1.5)
    tex = (tex - tex.min()) / np.ptp(tex) * 200 + 20
    shifts = [(1.6 * np.sin(i * 0.5), 1.3 * np.cos(i * 0.4)) for i in range(30)]
    frames = [ndshift(tex, (dy, dx), order=1, mode="reflect") for dx, dy in shifts]

    def mafd(fr):
        return float(np.mean([np.abs(fr[i + 1] - fr[i]).mean() for i in range(len(fr) - 1)]))

    before = mafd(frames)
    stab = stabilize_frames(frames, mode="euclidean", reference="mid")
    after = mafd(stab)
    assert len(stab) == len(frames)
    assert stab[0].shape == frames[0].shape
    assert after < 0.5 * before  # most of the shake removed


def test_build_six_rois_now_allowed():
    """MAX_ROIS was raised past 5; six ROIs build fine."""
    assert MAX_ROIS >= 6
    specs = [f"r{i}=0.{i}0,0.10,0.08,0.08" for i in range(6)]
    regions = build_roi_masks((200, 200), rois=specs, units="fraction")
    assert len(regions) == 6


# --------------------------------------------------------------------------- #
# Respiration / cardiac decomposition
# --------------------------------------------------------------------------- #
def test_decompose_recovers_breathing_and_cardiac():
    from scipy.signal import hilbert

    fps = 25.0
    n = int(fps * 40)
    t = np.arange(n) / fps
    fc, fr = 1.2, 0.2  # 72 ppm cardiac, 12 /min breathing
    rng = np.random.default_rng(7)
    resp = 0.3 * np.sin(2 * np.pi * fr * t)
    am = 1.0 + 0.5 * np.sin(2 * np.pi * fr * t)   # breathing modulates pulse amplitude
    cardiac = am * np.sin(2 * np.pi * fc * t)
    sig = 1.0 + resp + cardiac + 0.02 * rng.standard_normal(n)

    res = decompose_respiration(sig, fps, resp_bpm=(6, 30), cardiac_bpm=(40, 140))
    assert abs(res.cardiac_bpm - 72) < 6
    assert abs(res.resp_bpm - 12) < 4
    assert res.modulation_index > 0.1
    # deconvolution flattens the respiratory amplitude modulation
    cv_before = np.std(np.abs(hilbert(res.cardiac_wave))) / np.mean(np.abs(hilbert(res.cardiac_wave)))
    cv_after = np.std(np.abs(hilbert(res.cardiac_deconv))) / np.mean(np.abs(hilbert(res.cardiac_deconv)))
    assert cv_after < cv_before


def test_plot_breathing_decomposition(tmp_path):
    fps = 25.0
    n = int(fps * 30)
    t = np.arange(n) / fps
    am = 1.0 + 0.4 * np.sin(2 * np.pi * 0.2 * t)
    sig = 1.0 + 0.3 * np.sin(2 * np.pi * 0.2 * t) + am * np.sin(2 * np.pi * 1.2 * t)
    res = decompose_respiration(sig, fps)
    out = tmp_path / "breathing.png"
    plot_breathing_decomposition(res, out, title="synthetic")
    assert out.exists() and out.stat().st_size > 0
