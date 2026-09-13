"""Analyse one human (neurosurgical cortex) video: cortex pulsatility + respiration.

Usage:
    python run_human.py <label> <video-path> <roi> [cortex-roi-frame-index]

    <roi> is a cortex box "x,y,w,h" in fraction-of-frame coordinates.

Paths rooted at $PULS_WORK (default ./pulsatility_work); result pickled to
$PULS_WORK/out/<label>_full.pkl (e.g. human1_full.pkl).
"""
import os, sys, logging, pickle
from pathlib import Path
import numpy as np, cv2
from pulsatility import (load_video_gray, stabilize_frames, build_roi_masks,
                         motion_signals, analyze_pulsatility, decompose_respiration)

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work"))
OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)


def main(label, video, roi, frame_idx=1764):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frames, fps, scale = load_video_gray(video, resize_width=320, frame_stride=2,
                                         max_frames=1600, return_scale=True)
    frames = stabilize_frames(frames, mode="euclidean", reference="mid")
    regions = build_roi_masks(frames[0].shape, scale=scale, rois=[f"cortex={roi}"], units="fraction")
    full = np.ones(frames[0].shape, bool)
    sig, amap, ref = motion_signals(frames, [full, regions[0].mask], method="flow")
    res_full = analyze_pulsatility(frames, fps, precomputed=(sig[0], amap, ref), min_bpm=30, max_bpm=200)
    res_ctx = analyze_pulsatility(frames, fps, precomputed=(sig[1], amap, ref), min_bpm=30, max_bpm=200)
    resp = decompose_respiration(res_ctx.detrended, fps)

    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read(); cap.release()
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); fr = cv2.resize(fr, (720, int(720 * fr.shape[0] / fr.shape[1])))
    with open(OUT / f"{label}_full.pkl", "wb") as fh:
        pickle.dump(dict(cortex=res_ctx, full=res_full, resp=resp, fps=fps, color=fr,
                         roi_frac=tuple(float(x) for x in roi.split(","))), fh)
    print(label, "cortex", round(res_ctx.bpm_spectral, 1), "bpm  resp", round(resp.resp_hz * 60, 1),
          "bpm  modulation", round(resp.modulation_index, 3))


if __name__ == "__main__":
    idx = int(sys.argv[4]) if len(sys.argv) > 4 else 1764
    main(sys.argv[1], sys.argv[2], sys.argv[3], idx)
