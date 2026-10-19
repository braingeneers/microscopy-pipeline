#!/usr/bin/env python3
"""Rate-matched side-by-side pulsatility video.

Build a side-by-side comparison video of two perfused-parenchyma clips (e.g. a
human cortical exposure and a hydrogel bioreactor) in which every cardiac /
perfusion cycle is time-warped to a *common tempo* so the two beat in lockstep,
with each panel's displacement (~pressure) waveform drawn underneath and a
synchronized playhead.

Pipeline per side:
  load -> (optional) ECC stabilize -> ROI signed-velocity via Farneback optical
  flow (PCA principal axis) -> displacement = band-passed cumulative integral ->
  cycle segmentation (diastolic troughs) -> pick the K cleanest consecutive
  cycles -> resample each cycle to M frames (this is what locks tempo AND phase)
  -> (optional) crop -> (optional) Eulerian motion magnification -> compose.

Notes
-----
* Stabilization removes whole-field translation/rotation, which is right when a
  rigid frame surrounds locally-deforming tissue (cortex). For a soft chamber
  whose *bulk* motion is the signal you want to see, pass --no-<side>-stabilize.
* Motion magnification (--<side>-magnify N) is linear Eulerian amplification in a
  narrow band around the measured rate; it exaggerates a real but sub-pixel
  motion for visibility and is NOT real-scale. Label it when you use it.
* Amplitudes are normalized per panel (the gel typically moves ~30x less than
  cortex), so the waveforms are shape/timing comparisons, not amplitude.

Example
-------
    python analysis/pulsatility_sidebyside.py \
        --left human.mp4  --left-label "human brain" \
        --right gel.mp4   --right-label "hydrogel bioreactor" \
        --no-right-stabilize --right-crop 0.06,0.02,0.66,0.82 --right-magnify 2 \
        --out sidebyside.mp4 --gif sidebyside.gif --minimal
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import cv2
from scipy.signal import butter, filtfilt, find_peaks

cv2.setNumThreads(1)


# --------------------------------------------------------------------------- #
# ffmpeg discovery (system ffmpeg, or the imageio-ffmpeg bundled binary)
# --------------------------------------------------------------------------- #
def ffmpeg_exe() -> str:
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("No ffmpeg found. Install ffmpeg or `pip install "
                           "imageio-ffmpeg`, or set $FFMPEG.") from e


# --------------------------------------------------------------------------- #
# signal helpers
# --------------------------------------------------------------------------- #
def spectrum(x: np.ndarray, fps: float):
    x = x - x.mean()
    w = np.hanning(len(x))
    sp = np.abs(np.fft.rfft(x * w))
    f = np.fft.rfftfreq(len(x), 1.0 / fps)
    return f, sp


def dominant_f0(sig: np.ndarray, fps: float, band_bpm=(36, 170)) -> float:
    f, p = spectrum(sig, fps)
    bpm = f * 60
    m = (bpm >= band_bpm[0]) & (bpm <= band_bpm[1])
    return float(f[m][np.argmax(p[m])])


def bandpass(x: np.ndarray, fps: float, f0: float, lo=0.7, hi=1.4) -> np.ndarray:
    nyq = fps / 2.0
    b, a = butter(2, [max(0.02, f0 * lo / nyq), min(0.95, f0 * hi / nyq)], btype="band")
    return filtfilt(b, a, x)


def displacement(sig: np.ndarray, fps: float, f0: float) -> np.ndarray:
    """~pressure: band-passed cumulative integral of signed velocity (smooth)."""
    return bandpass(np.cumsum(sig - sig.mean()), fps, f0)


def displacement_raw(sig: np.ndarray, fps: float, f0: float) -> np.ndarray:
    """Displacement with drift removed only (high-pass) -- keeps the real detail,
    no low-pass smoothing."""
    nyq = fps / 2.0
    b, a = butter(2, max(0.02, 0.5 * f0 / nyq), btype="high")
    return filtfilt(b, a, np.cumsum(sig - sig.mean()))


# --------------------------------------------------------------------------- #
# video helpers
# --------------------------------------------------------------------------- #
def load_video(path: str, width: int, max_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        if w != width:
            f = cv2.resize(f, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, float(fps)


def _norm(a):
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def ecc_stabilize(frames_bgr):
    """ECC euclidean-align every frame to a mid reference; warp color + gray alike."""
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    n = len(gray)
    h, w = gray[0].shape
    ref = _norm(gray[n // 2])
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4)
    cs, gs = [], []
    nfail = 0
    for fb, fg in zip(frames_bgr, gray):
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ref, _norm(fg), warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
            fl = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            cb = cv2.warpAffine(fb, warp, (w, h), flags=fl, borderMode=cv2.BORDER_REPLICATE)
            gb = cv2.warpAffine(fg, warp, (w, h), flags=fl, borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            cb, gb, nfail = fb.copy(), fg.copy(), nfail + 1
        cs.append(cb)
        gs.append(gb.astype(np.float32))
    if nfail:
        print(f"  stabilize: {nfail}/{n} frames did not converge (left as-is)")
    return cs, gs


def signed_velocity(gray, roi=(0.30, 0.70, 0.30, 0.70)):
    """ROI-mean Farneback flow projected on its principal (PCA) axis."""
    H, W = gray[0].shape
    x0, x1, y0, y1 = roi
    m = np.zeros((H, W), bool)
    m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
    n = len(gray) - 1
    mu = np.zeros(n)
    mv = np.zeros(n)
    prev = cv2.GaussianBlur(gray[0], (0, 0), 1.0)
    for i in range(1, len(gray)):
        cur = cv2.GaussianBlur(gray[i], (0, 0), 1.0)
        fl = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mu[i - 1] = fl[..., 0][m].mean()
        mv[i - 1] = fl[..., 1][m].mean()
        prev = cur
    A = np.stack([mu - mu.mean(), mv - mv.mean()], 1)
    w, V = np.linalg.eigh(A.T @ A)
    return A @ V[:, int(np.argmax(w))]


def crop_frames(frames, box):
    if box is None:
        return frames
    h, w = frames[0].shape[:2]
    x0, y0, x1, y1 = box
    a, b, c, d = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    return [f[b:d, a:c] for f in frames]


def apply_look(frames, contrast=1.0, gamma=1.0):
    """S-curve contrast around mid-grey + gamma (gamma>1 deepens shadows). LUT, fast."""
    if contrast == 1.0 and gamma == 1.0:
        return frames
    x = np.arange(256) / 255.0
    y = np.clip((x - 0.5) * contrast + 0.5, 0, 1) ** gamma
    lut = np.clip(y * 255, 0, 255).astype(np.uint8)
    return [cv2.LUT(f, lut) for f in frames]


def eulerian_magnify(frames_bgr, fps, f0, alpha, blur=2.2):
    """Linear Eulerian motion magnification in a narrow band around f0."""
    arr = np.stack(frames_bgr, 0).astype(np.float32)
    n, nyq = arr.shape[0], fps / 2.0
    b, a = butter(2, [max(0.02, f0 * 0.7 / nyq), min(0.95, f0 * 1.4 / nyq)], btype="band")
    for ch in range(3):
        band = filtfilt(b, a, arr[..., ch], axis=0).astype(np.float32)
        if blur:
            for i in range(n):
                band[i] = cv2.GaussianBlur(band[i], (0, 0), blur)
        arr[..., ch] += alpha * band
        del band
    return [np.clip(f, 0, 255).astype(np.uint8) for f in arr]


# --------------------------------------------------------------------------- #
# cycle segmentation / rate-matching
# --------------------------------------------------------------------------- #
def cycle_troughs(disp, fps, f0):
    dist = max(3, int(0.62 * fps / f0))
    tr, _ = find_peaks(-disp, distance=dist)
    return tr


def pick_window(disp, troughs, K):
    """Choose the K consecutive cycles whose shapes correlate best (most regular)."""
    cyc = []
    for a, b in zip(troughs[:-1], troughs[1:]):
        seg = disp[a:b]
        cyc.append(np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(seg)), seg)
                   if len(seg) >= 4 else None)
    valid = [c for c in cyc if c is not None]
    if not valid:
        raise RuntimeError("no usable cycles detected")
    med = np.median(np.stack([c / (np.ptp(c) + 1e-9) for c in valid]), 0)
    corr = np.array([float(np.corrcoef(c / (np.ptp(c) + 1e-9), med)[0, 1]) if c is not None else -1
                     for c in cyc])
    nseg = len(cyc)
    if nseg <= K:
        return 0, nseg
    return int(np.argmax([corr[i:i + K].mean() for i in range(nseg - K + 1)])), K


def build_montage(frames, disp, troughs, start, K, M):
    """Resample each of K cycles to M frames -> common tempo + phase alignment."""
    idxs, wav = [], []
    for j in range(start, start + K):
        a, b = troughs[j], troughs[j + 1]
        for m in range(M):
            src = a + (b - a) * m / M
            idxs.append(int(round(min(src, len(frames) - 1))))
            wav.append(float(np.interp(src, np.arange(len(disp)), disp)))
    return idxs, np.array(wav)


def resize_panels(frames, idxs, width):
    out = []
    for i in idxs:
        f = frames[i]
        h, w = f.shape[:2]
        out.append(cv2.resize(f, (width, int(round(h * width / w))), interpolation=cv2.INTER_CUBIC))
    return out


# --------------------------------------------------------------------------- #
# per-side preparation
# --------------------------------------------------------------------------- #
def prepare_side(path, *, width, max_frames, stabilize, crop, magnify,
                 band_bpm, cycles, mpc, contrast=1.0, gamma=1.0, raw_waveform=False,
                 roi=(0.30, 0.70, 0.30, 0.70)):
    frames, fps = load_video(path, width, max_frames)
    if stabilize:
        frames, gray = ecc_stabilize(frames)
    else:
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    sig = signed_velocity(gray, roi)
    f0 = dominant_f0(sig, fps, band_bpm)
    disp = displacement(sig, fps, f0)            # smooth: robust cycle segmentation
    troughs = cycle_troughs(disp, fps, f0)
    if len(troughs) < 2:
        raise RuntimeError(f"{path}: could not segment cycles at {f0*60:.0f} bpm")
    start, K = pick_window(disp, troughs, cycles)
    wave = displacement_raw(sig, fps, f0) if raw_waveform else disp   # what gets drawn
    idxs, wav = build_montage(frames, wave, troughs, start, K, mpc)
    if abs(wav.min()) > abs(wav.max()):   # orient systolic excursion upward
        wav = -wav
    frames = crop_frames(frames, crop)
    if magnify > 0:
        frames = eulerian_magnify(frames, fps, f0, magnify)
    frames = apply_look(frames, contrast, gamma)
    return dict(panel=resize_panels(frames, idxs, width), wav=wav,
                bpm=f0 * 60, K=K, M=mpc, fps=fps)


# --------------------------------------------------------------------------- #
# compositing
# --------------------------------------------------------------------------- #
def _draw_strip(canvas, x0, y0, w, h, trace, t, color, label, M, minimal, bpm):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (248, 248, 248), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 1)
    mid = y0 + h // 2
    tr = trace / (np.max(np.abs(trace)) + 1e-9)
    xs = (x0 + np.arange(len(tr)) * (w - 1) / (len(tr) - 1)).astype(int)
    ys = (mid - tr * (h * 0.40)).astype(int)
    for b in range(0, len(tr) + 1, M):
        xb = int(x0 + min(b, len(tr) - 1) * (w - 1) / (len(tr) - 1))
        cv2.line(canvas, (xb, y0 + 3), (xb, y0 + h - 3), (230, 230, 230), 1)
    cv2.line(canvas, (x0, mid), (x0 + w, mid), (205, 205, 205), 1)
    pts = np.stack([xs, ys], 1)
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, (200, 200, 200), 1, cv2.LINE_AA)
    if t >= 1:
        cv2.polylines(canvas, [pts[:t + 1].reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    px, py = xs[t], ys[t]
    cv2.line(canvas, (px, y0 + 4), (px, y0 + h - 4), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label, (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    if not minimal:
        cv2.putText(canvas, f"{bpm:.0f} bpm  -  displacement (norm.)", (x0 + 8, y0 + h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)


def compose(left, right, out_path, *, labels, colors, fps, loops, minimal, title):
    PW = left["panel"][0].shape[1]
    PH = max(left["panel"][0].shape[0], right["panel"][0].shape[0])
    MAR, GAP, WH, VWSP = 24, 24, 140, 8
    TOP = 16 if minimal else 64
    BOT = 14 if minimal else 22
    CW = MAR + PW + GAP + PW + MAR
    CH = TOP + PH + VWSP + WH + BOT
    CW -= CW % 2
    CH -= CH % 2
    T = min(len(left["panel"]), len(right["panel"]))
    sides = [(MAR, left, labels[0], colors[0]),
             (MAR + PW + GAP, right, labels[1], colors[1])]
    proc = subprocess.Popen(
        [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{CW}x{CH}", "-r", str(fps), "-i", "-", "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path],
        stdin=subprocess.PIPE)
    for g in range(T * loops):
        t = g % T
        canvas = np.full((CH, CW, 3), 255, np.uint8)
        if title and not minimal:
            cv2.putText(canvas, title, (MAR, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        for px, S, label, color in sides:
            fr = S["panel"][t]
            ph = fr.shape[0]
            yoff = TOP + (PH - ph) // 2
            canvas[yoff:yoff + ph, px:px + PW] = fr
            cv2.rectangle(canvas, (px, yoff), (px + PW, yoff + ph), (180, 180, 180), 1)
            _draw_strip(canvas, px, TOP + PH + VWSP, PW, WH, S["wav"][:T], t, color, label,
                        S["M"], minimal, S["bpm"])
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out_path}  ({CW}x{CH}, {T*loops/fps:.1f}s)")
    return out_path


def to_gif(mp4, gif, *, fps=15, width=820, seconds=5.0, max_mb=30):
    """One seamless loop -> palette-optimized GIF, shrinking until under max_mb."""
    ff = ffmpeg_exe()
    for W, F in [(width, fps), (700, fps), (640, 12), (560, 12), (480, 10)]:
        pal = gif + ".pal.png"
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-t", str(seconds),
                        "-i", mp4, "-vf", f"fps={F},scale={W}:-1:flags=lanczos,palettegen=max_colors=192",
                        pal], check=True)
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-t", str(seconds), "-i", mp4,
                        "-i", pal, "-lavfi",
                        f"fps={F},scale={W}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        "-loop", "0", gif], check=True)
        os.remove(pal)
        mb = os.path.getsize(gif) / 1e6
        print(f"  gif {W}px {F}fps -> {mb:.1f}MB")
        if mb <= max_mb:
            return gif
    print(f"  WARN: {gif} still over {max_mb}MB")
    return gif


def _box(s):
    if s is None:
        return None
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("crop must be x0,y0,x1,y1 (fractions)")
    return tuple(v)


def _bgr(hexstr):
    h = hexstr.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def main(argv=None):
    p = argparse.ArgumentParser(description="Rate-matched side-by-side pulsatility video.")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--left-label", default="left")
    p.add_argument("--right-label", default="right")
    p.add_argument("--out", required=True)
    p.add_argument("--gif", default=None, help="also export a <30MB looping GIF to this path")
    p.add_argument("--left-stabilize", dest="lstab", action="store_true", default=True)
    p.add_argument("--no-left-stabilize", dest="lstab", action="store_false")
    p.add_argument("--right-stabilize", dest="rstab", action="store_true", default=True)
    p.add_argument("--no-right-stabilize", dest="rstab", action="store_false")
    p.add_argument("--left-crop", type=_box, default=None)
    p.add_argument("--right-crop", type=_box, default=None)
    p.add_argument("--left-magnify", type=float, default=0.0)
    p.add_argument("--right-magnify", type=float, default=0.0)
    p.add_argument("--rate-band", default="36,170", help="bpm search band 'lo,hi'")
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--mpc", type=int, default=30, help="output frames per cycle")
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--panel-width", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--contrast", type=float, default=1.0, help="S-curve contrast gain (1.0 = none); default for both panels")
    p.add_argument("--gamma", type=float, default=1.0, help="shadow gamma (>1 deepens shadows); default for both panels")
    p.add_argument("--left-contrast", type=float, default=None, help="override contrast for the left panel")
    p.add_argument("--left-gamma", type=float, default=None)
    p.add_argument("--right-contrast", type=float, default=None, help="override contrast for the right panel")
    p.add_argument("--right-gamma", type=float, default=None)
    p.add_argument("--raw-waveform", action="store_true",
                   help="draw the detrended (unsmoothed) displacement instead of the band-passed one")
    p.add_argument("--minimal", action="store_true", help="only the two panel labels, no other text")
    p.add_argument("--title", default="Parenchymal pulsatility  -  rate-matched side-by-side")
    p.add_argument("--left-color", default="1f6feb")
    p.add_argument("--right-color", default="d1242f")
    a = p.parse_args(argv)
    band = tuple(float(x) for x in a.rate_band.split(","))

    lc = a.left_contrast if a.left_contrast is not None else a.contrast
    lg = a.left_gamma if a.left_gamma is not None else a.gamma
    rc = a.right_contrast if a.right_contrast is not None else a.contrast
    rg = a.right_gamma if a.right_gamma is not None else a.gamma
    L = prepare_side(a.left, width=a.panel_width, max_frames=a.max_frames, stabilize=a.lstab,
                     crop=a.left_crop, magnify=a.left_magnify, band_bpm=band, cycles=a.cycles,
                     mpc=a.mpc, contrast=lc, gamma=lg, raw_waveform=a.raw_waveform)
    R = prepare_side(a.right, width=a.panel_width, max_frames=a.max_frames, stabilize=a.rstab,
                     crop=a.right_crop, magnify=a.right_magnify, band_bpm=band, cycles=a.cycles,
                     mpc=a.mpc, contrast=rc, gamma=rg, raw_waveform=a.raw_waveform)
    print(f"left {L['bpm']:.0f}bpm K={L['K']}; right {R['bpm']:.0f}bpm K={R['K']}")
    compose(L, R, a.out, labels=(a.left_label, a.right_label),
            colors=(_bgr(a.left_color), _bgr(a.right_color)),
            fps=a.fps, loops=a.loops, minimal=a.minimal, title=a.title)
    if a.gif:
        to_gif(a.out, a.gif)
    return 0


if __name__ == "__main__":
    sys.exit(main())
