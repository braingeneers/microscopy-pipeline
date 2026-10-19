"""Analyse one bioreactor replicate video and pickle its per-ROI PulsatilityResults.

Usage:
    python run_bioreactor.py <rep-number> <video-path>

Paths are rooted at $PULS_WORK (default ./pulsatility_work); results go to
$PULS_WORK/out/bio_rep<N>_full.pkl. ROI boxes below are the ones used for the
three replicates in this study (fraction-of-frame x,y,w,h); edit for new videos.
"""
import os, sys, logging, pickle
from pathlib import Path
import numpy as np, cv2
from pulsatility import (load_video_gray, stabilize_frames, build_roi_masks,
                         motion_signals, analyze_pulsatility)

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work"))
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)

ROISETS = {
    1: {"o1": "0.378,0.398,0.082,0.045", "o2": "0.458,0.418,0.082,0.045",
        "within-vessel": "0.045,0.33,0.165,0.17", "open-field": "0.44,0.225,0.30,0.095",
        "housing": "0.36,0.75,0.24,0.10"},
    2: {"o1": "0.260,0.340,0.060,0.060", "o2": "0.330,0.340,0.060,0.060",
        "o3": "0.420,0.320,0.060,0.060", "o4": "0.500,0.300,0.060,0.060",
        "within-vessel": "0.900,0.290,0.100,0.120", "housing": "0.390,0.845,0.220,0.110"},
    3: {"o1": "0.320,0.353,0.060,0.095", "o2": "0.393,0.373,0.060,0.095",
        "o3": "0.467,0.393,0.060,0.095", "o4": "0.540,0.413,0.060,0.095",
        "within-vessel": "0.000,0.330,0.110,0.130", "housing": "0.350,0.850,0.200,0.100"},
}


def main(rep, video):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rois = ROISETS[rep]
    frames, fps, scale = load_video_gray(video, resize_width=320, frame_stride=2,
                                         max_frames=900, return_scale=True)
    frames = stabilize_frames(frames, mode="euclidean", reference="mid")
    regions = build_roi_masks(frames[0].shape, scale=scale,
                              rois=[f"{k}={v}" for k, v in rois.items()], units="fraction")
    full = np.ones(frames[0].shape, bool)
    names = ["full field"] + [r.name for r in regions]
    sig, amap, ref = motion_signals(frames, [full] + [r.mask for r in regions], method="flow")
    res = {n: analyze_pulsatility(frames, fps, precomputed=(s, amap, ref), min_bpm=30, max_bpm=300)
           for n, s in zip(names, sig)}

    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ok, fr = cap.read(); cap.release()
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); fr = cv2.resize(fr, (680, int(680 * fr.shape[0] / fr.shape[1])))
    with open(OUT / f"bio_rep{rep}_full.pkl", "wb") as fh:
        pickle.dump(dict(results=res, fps=fps, color=fr,
                         roi_fracs={k: tuple(float(x) for x in v.split(",")) for k, v in rois.items()}), fh)
    for n in names:
        print(f"rep{rep}", n, round(res[n].bpm_spectral, 1), "SNR", round(res[n].spectral_snr * 100))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
