"""Extract cardiac pulsatility from a hands-in-frame SURGICAL human cortex video.

Unlike the clean 4K recordings handled by run_human.py, intra-operative clips have
a gloved hand and instruments crossing the field almost continuously, injecting
optical-flow transients tens of times larger than the cortical pulse. This script
rejects those transients (robust median/MAD outliers, dilated) and keeps only
contiguous hand-free runs, then extracts a band-limited cardiac signal from each
usable run.

Output ($PULS_WORK/out/<key>_surgical.pkl) is a list of per-run (band-limited
signal, fps, f0) records plus metadata. Downstream:
  * build_gold_standard.py folds the pooled clean-run cycles into the human gold
    standard as a third, down-weighted real human (noisier, intra-operative);
  * pool_surgical_human.py shows the single-run vs pooled ensemble (why pooling the
    short clean runs sharpens the mean pulse).

Only runs whose length is >= min_run seconds are kept (default 14 s -- long enough
for a stable cardiac spectrum at ~40-60 bpm). A recording whose clean runs do not
share a reproducible fundamental (e.g. too hand-contaminated) is not usable even if
runs are found; inspect the printed per-run rates before trusting it.

Usage:
    python run_human_surgical.py <key> <video-path> [roi] [min_run_s]
    <roi> is a cortex box "x,y,w,h" in fraction-of-frame; default is this study's
    human-4 box. Paths rooted at $PULS_WORK (default ./pulsatility_work).
"""
import os, sys, pickle
from pathlib import Path
import numpy as np, cv2
from scipy.ndimage import binary_dilation
from numpy.fft import rfft, rfftfreq
from pulsatility import stabilize_frames, build_roi_masks, motion_signals
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work"))
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
ROISETS = {"human4": "0.46,0.18,0.20,0.28"}
STRIDE = 2; WINLEN = 60.0                       # per-window stabilization (camera pans/zooms)


def _window(video, a, b, fps0):
    """Grayscale, downscaled frames for [a, b) seconds, decimated by STRIDE."""
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(a * fps0))
    frames = []; need = int((b - a) * fps0)
    for i in range(need):
        if not cap.grab():
            break
        if i % STRIDE == 0:
            ok, fr = cap.retrieve()
            if ok:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1]))))
    cap.release()
    return frames


def _clean_runs(raw, fps, min_run):
    """Contiguous runs with no hand/instrument transient, >= min_run seconds."""
    med = np.median(raw); mad = np.median(np.abs(raw - med)) + 1e-9
    art = binary_dilation(np.abs((raw - med) / (1.4826 * mad)) > 6, iterations=int(round(fps * 0.5)))
    runs = []; s = None
    for k in range(len(art)):
        if not art[k] and s is None:
            s = k
        if (art[k] or k == len(art) - 1) and s is not None:
            runs.append((s, k if art[k] else k + 1)); s = None
    return [(a, e) for a, e in runs if (e - a) / fps >= min_run]


def _f0(sig, fps):
    """Cardiac fundamental (bpm->Hz) from the boxcar-detrended clean run.

    The strongest in-band peak can be a harmonic (a weak-signal run's 2nd harmonic
    may top its fundamental), so prefer the lowest sub-multiple (pk/2, pk/3) that
    still carries substantial power -- this rejects e.g. a spurious 2x reading."""
    d = sig - np.convolve(np.pad(sig, 30, mode="edge"), np.ones(61) / 61, "valid"); d -= d.mean()
    f = rfftfreq(len(d), 1 / fps); P = np.abs(rfft(d * np.hanning(len(d)))) ** 2; bpm = f * 60
    band = (bpm >= 35) & (bpm <= 110); fb, Pb = bpm[band], P[band]
    pk = fb[int(np.argmax(Pb))]
    cands = [pk]
    for k in (2, 3):                                   # demote harmonics to the fundamental
        sub = pk / k
        near = np.abs(fb - sub) <= 3
        if sub >= 35 and near.any() and Pb[near].max() >= 0.4 * Pb.max():
            cands.append(float(fb[near][int(np.argmax(Pb[near]))]))
    return float(min(cands) / 60)


def main(key, video, roi=None, min_run=14.0):
    roi = roi or ROISETS.get(key, "0.46,0.18,0.20,0.28")
    cap = cv2.VideoCapture(str(video)); fps0 = cap.get(cv2.CAP_PROP_FPS); dur = cap.get(7) / fps0; cap.release()
    fps = fps0 / STRIDE
    runs = []
    for a in np.arange(0, dur - 5, WINLEN):
        frames = _window(video, a, min(a + WINLEN, dur), fps0)
        if len(frames) < 30:
            continue
        frames = stabilize_frames(frames, mode="euclidean", reference="mid")
        reg = build_roi_masks(frames[0].shape, rois=[f"cortex={roi}"], units="fraction")
        sig, _, _ = motion_signals(frames, [reg[0].mask], method="flow")
        raw = sig[0]
        for (s, e) in _clean_runs(raw, fps, min_run):
            sub = raw[s:e]; f0 = _f0(sub, fps)
            runs.append(dict(signal=figures._bandlimit(sub, fps, f0), fps=fps, f0=f0,
                             start_s=float(a + s / fps), dur_s=float((e - s) / fps)))
        print(f"win {a:.0f}-{min(a + WINLEN, dur):.0f}s: {len(runs)} clean runs so far", flush=True)

    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(dur / 2, 190) * fps0))
    ok, fr = cap.read(); cap.release()
    color = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (720, int(720 * fr.shape[0] / fr.shape[1]))) if ok else None
    with open(OUT / f"{key}_surgical.pkl", "wb") as fh:
        pickle.dump(dict(runs=runs, fps=fps, roi_frac=tuple(float(x) for x in roi.split(",")),
                         color=color, key=key), fh)
    if runs:
        rates = [r["f0"] * 60 for r in runs]
        print(f"{key}: {len(runs)} clean runs, rate {min(rates):.0f}-{max(rates):.0f} bpm "
              f"(median {np.median(rates):.0f}), total {sum(r['dur_s'] for r in runs):.0f}s clean")
    else:
        print(f"{key}: no clean runs >= {min_run:.0f}s found")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
         float(sys.argv[4]) if len(sys.argv) > 4 else 14.0)
