#!/usr/bin/env python3
"""Rank perfused-parenchyma clips by how clean their pulsatility signal is.

For each input video: ECC-stabilize, take the ROI signed-velocity (Farneback +
PCA principal axis), and score the spectrum in a physiological band:

  * f0        - dominant rate (bpm)
  * snr       - peak power / median off-peak power in the band
  * conc      - fraction of band power within +/-2 bins of the peak
  * amp       - std of the signed-velocity signal (relative motion magnitude)

Clips are ranked by snr * conc (clean AND concentrated at one rate). Use it to
pick the "best" clip of each kind before building a comparison video.

    python analysis/score_pulsatility_clips.py bio1.mp4 bio2.mp4 brain1.mov ...
"""
from __future__ import annotations
import os, sys, json

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pulsatility_sidebyside import load_video, ecc_stabilize, signed_velocity, spectrum


def score(path, width=400, max_frames=900, band_bpm=(36, 170), roi=(0.30, 0.70, 0.30, 0.70)):
    frames, fps = load_video(path, width, max_frames)
    _, gray = ecc_stabilize(frames)
    sig = signed_velocity(gray, roi)
    f, p = spectrum(sig, fps)
    bpm = f * 60
    m = (bpm >= band_bpm[0]) & (bpm <= band_bpm[1])
    fb, pb = f[m], p[m]
    k = int(np.argmax(pb))
    keep = np.ones(len(pb), bool)
    for d in (-1, 0, 1):
        if 0 <= k + d < len(pb):
            keep[k + d] = False
    snr = pb[k] / (np.median(pb[keep]) + 1e-12)
    lo, hi = max(0, k - 2), min(len(pb), k + 3)
    conc = pb[lo:hi].sum() / (pb.sum() + 1e-12)
    return dict(clip=os.path.basename(path), f0_bpm=round(float(fb[k] * 60), 1),
                snr=round(float(snr), 2), conc=round(float(conc), 3),
                amp=round(float(np.std(sig)), 4), n=len(frames), fps=round(fps, 1))


def main(argv=None):
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        print(__doc__); return 2
    rows = []
    for p in paths:
        try:
            r = score(p); rows.append(r); print(json.dumps(r))
        except Exception as e:
            print(json.dumps({"clip": os.path.basename(p), "error": str(e)}))
    print("\n=== ranking (snr x conc) ===")
    for r in sorted(rows, key=lambda d: d.get("snr", 0) * d.get("conc", 0), reverse=True):
        print(f"  {r['clip']:28s} f0={r['f0_bpm']:6.1f}bpm  snr={r['snr']:7.2f}  "
              f"conc={r['conc']:.3f}  amp={r['amp']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
