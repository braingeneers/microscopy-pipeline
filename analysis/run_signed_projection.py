"""Signed-projection motion extraction.

Instead of the rectified flow *magnitude* (tissue speed), project the mean optical
flow inside each ROI onto that ROI's principal motion axis (PCA), giving a SIGNED
velocity that can be integrated to a displacement (proportional to parenchymal
pressure). Preserves direction (systolic push vs diastolic recoil).

Usage:
    python run_signed_projection.py <roi_key> <video-path>
    roi_key in {human1, human2, bio1, bio2, bio3} selects the ROI set.

Writes $PULS_WORK/out/signed_<roi_key>.pkl.
"""
import os, sys, pickle
from pathlib import Path
import numpy as np, cv2
from pulsatility import load_video_gray, stabilize_frames, build_roi_masks

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work"))
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)
ROISETS = {
    "human1": {"cortex": "0.40,0.30,0.24,0.34"},
    "human2": {"cortex": "0.436,0.400,0.247,0.440"},
    "bio1": {"o1": "0.378,0.398,0.082,0.045", "o2": "0.458,0.418,0.082,0.045",
             "within-vessel": "0.045,0.33,0.165,0.17", "housing": "0.36,0.75,0.24,0.10"},
    "bio2": {"o1": "0.260,0.340,0.060,0.060", "o2": "0.330,0.340,0.060,0.060",
             "o3": "0.420,0.320,0.060,0.060", "o4": "0.500,0.300,0.060,0.060",
             "within-vessel": "0.900,0.290,0.100,0.120", "housing": "0.390,0.845,0.220,0.110"},
    "bio3": {"o1": "0.320,0.353,0.060,0.095", "o2": "0.393,0.373,0.060,0.095",
             "o3": "0.467,0.393,0.060,0.095", "o4": "0.540,0.413,0.060,0.095",
             "within-vessel": "0.000,0.330,0.110,0.130", "housing": "0.350,0.850,0.200,0.100"},
}


def main(key, video):
    maxf = 1600 if key.startswith("human") else 900
    frames, fps, scale = load_video_gray(video, resize_width=320, frame_stride=2,
                                         max_frames=maxf, return_scale=True)
    frames = stabilize_frames(frames, mode="euclidean", reference="mid")
    regions = build_roi_masks(frames[0].shape, scale=scale,
                              rois=[f"{k}={v}" for k, v in ROISETS[key].items()], units="fraction")
    masks = {r.name: r.mask.astype(bool) for r in regions}
    names = list(masks); N = len(frames) - 1
    mu = {n: np.zeros(N) for n in names}; mv = {n: np.zeros(N) for n in names}
    prev = cv2.GaussianBlur(frames[0], (0, 0), 1.0)
    for i in range(1, len(frames)):
        cur = cv2.GaussianBlur(frames[i], (0, 0), 1.0)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u, v = flow[..., 0], flow[..., 1]
        for n in names:
            m = masks[n]; mu[n][i - 1] = float(u[m].mean()); mv[n][i - 1] = float(v[m].mean())
        prev = cur
    signed = {}
    for n in names:
        A = np.stack([mu[n] - mu[n].mean(), mv[n] - mv[n].mean()], 1)
        wv, V = np.linalg.eigh(A.T @ A)
        signed[n] = A @ V[:, int(np.argmax(wv))]
    with open(OUT / f"signed_{key}.pkl", "wb") as fh:
        pickle.dump(dict(signed=signed, fps=fps, names=names), fh)
    print(key, "done", N, "frames", round(fps, 1), "fps")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
