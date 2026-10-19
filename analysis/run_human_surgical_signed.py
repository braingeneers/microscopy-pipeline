"""Signed-projection velocity for a SURGICAL human's hand-free clean runs.

run_human_surgical.py recovers rectified tissue SPEED from the contiguous hand-free
runs of an intra-operative clip. This script adds the SIGNED (directional) tissue
velocity for those same runs, so the surgical patient can also enter the signed-
velocity / displacement panels. For each clean run it reloads the recorded time
window, stabilizes it, and projects the ROI-mean optical flow (u, v) onto its
principal motion axis (first PCA component) -- preserving direction (systolic push
vs diastolic recoil), which the flow magnitude discards.

Reads  $PULS_WORK/out/<key>_surgical.pkl  (needs the per-run start_s/dur_s/f0).
Writes $PULS_WORK/out/signed_<key>.pkl = dict(runs=[{signed, fps, f0, start_s,
dur_s}], fps): one signed-velocity trace per clean run.

Usage:
    python run_human_surgical_signed.py <key> <video-path> [roi]
    <roi> "x,y,w,h" in fraction-of-frame; default is this study's patient-4 box.
"""
import os, sys, pickle
from pathlib import Path
import numpy as np, cv2
from pulsatility import stabilize_frames, build_roi_masks

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work")); OUT = WORK / "out"
ROISETS = {"human4": "0.46,0.18,0.20,0.28"}
STRIDE = 2


def _window(video, a, dur, fps0):
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(a * fps0))
    frames = []; need = int(dur * fps0)
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


def _signed(video, a, dur, fps0, roi):
    frames = _window(video, a, dur, fps0)
    frames = stabilize_frames(frames, mode="euclidean", reference="mid")
    m = build_roi_masks(frames[0].shape, rois=[f"c={roi}"], units="fraction")[0].mask.astype(bool)
    mu = np.zeros(len(frames) - 1); mv = np.zeros(len(frames) - 1)
    prev = cv2.GaussianBlur(frames[0], (0, 0), 1.0)
    for i in range(1, len(frames)):
        cur = cv2.GaussianBlur(frames[i], (0, 0), 1.0)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mu[i - 1] = flow[..., 0][m].mean(); mv[i - 1] = flow[..., 1][m].mean(); prev = cur
    A = np.stack([mu - mu.mean(), mv - mv.mean()], 1)
    wv, V = np.linalg.eigh(A.T @ A)
    return A @ V[:, int(np.argmax(wv))]                 # project on principal motion axis


def main(key, video, roi=None):
    roi = roi or ROISETS.get(key, "0.46,0.18,0.20,0.28")
    srec = pickle.load(open(OUT / f"{key}_surgical.pkl", "rb"))
    cap = cv2.VideoCapture(str(video)); fps0 = cap.get(cv2.CAP_PROP_FPS); cap.release(); fps = fps0 / STRIDE
    runs = []
    for r in srec["runs"]:
        s = _signed(video, r["start_s"], r["dur_s"], fps0, roi)
        runs.append(dict(signed=s, fps=fps, f0=r["f0"], start_s=r["start_s"], dur_s=r["dur_s"]))
        print(f"run {r['start_s']:.0f}s ({r['dur_s']:.0f}s): {len(s)} samples, f0 {r['f0'] * 60:.0f} bpm", flush=True)
    pickle.dump(dict(runs=runs, fps=fps, key=key), open(OUT / f"signed_{key}.pkl", "wb"))
    print(f"saved signed_{key}.pkl ({len(runs)} runs)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
