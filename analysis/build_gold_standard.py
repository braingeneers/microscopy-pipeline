"""Build the human-as-gold-standard figures (speed space and signed/displacement).

Scores the bioreactor model, plus a rate-matched sine and flat-flow straw man, by
closeness (R^2) to the real human cortex pulse; the human's own cycle-to-cycle
self-consistency is the achievable ceiling.

Speed space uses the *_full.pkl results (run_human.py / run_bioreactor.py).
Displacement space integrates the SIGNED velocity (run_signed_projection.py):
displacement ~ parenchymal pressure, so it is compared to the pressure template
directly. Expects $PULS_WORK/out/{human1,human2}_full.pkl, bio_rep{1,2,3}_full.pkl,
and signed_{human1,human2,bio1,bio2,bio3}.pkl.
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
from scipy.stats import skew
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures
from pulsatility import highpass

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work"))
OUT = WORK / "out"


def _load(name):
    return pickle.load(open(OUT / f"{name}.pkl", "rb"))


def _displacement(signed_v, fps, f0):
    x = np.cumsum(signed_v - np.mean(signed_v))      # integrate signed velocity
    x = highpass(x, fps, 12.0)                         # remove integration drift
    x = figures._bandlimit(x, fps, f0)                 # cardiac band (also strips respiration)
    return x if skew(x) >= 0 else -x                   # orient: systolic spike positive


def main():
    h1, h2 = _load("human1_full"), _load("human2b_full") if (OUT / "human2b_full.pkl").exists() else _load("human2_full")
    bf = {i: _load(f"bio_rep{i}_full") for i in (1, 2, 3)}
    humans = [dict(label="Human 1", cortex=h1["cortex"], resp=h1["resp"], fps=h1["fps"]),
              dict(label="Human 2", cortex=h2["cortex"], resp=h2["resp"], fps=h2["fps"])]
    bio = [dict(results=bf[i]["results"], fps=bf[i]["fps"]) for i in (1, 2, 3)]

    figures.gold_standard_figure(humans, bio, OUT / "gold_standard_speed")
    print("built gold_standard_speed")

    if all((OUT / f"signed_{k}.pkl").exists() for k in ("human1", "human2", "bio1", "bio2", "bio3")):
        f0h = {"human1": h1["cortex"].dominant_hz, "human2": h2["cortex"].dominant_hz}
        f0b = {f"bio{i}": bf[i]["results"]["within-vessel"].dominant_hz for i in (1, 2, 3)}
        human_sigs = []
        for k in ("human1", "human2"):
            s = _load(f"signed_{k}"); fps = s["fps"]
            human_sigs.append((_displacement(s["signed"]["cortex"], fps, f0h[k]), fps, f0h[k]))
        bio_sigs = []
        for i in (1, 2, 3):
            k = f"bio{i}"; s = _load(f"signed_{k}"); fps = s["fps"]; f0 = f0b[k]; hz = s["signed"]["housing"]
            for on in [n for n in s["names"] if len(n) == 2 and n[0] == "o"]:
                v = figures._regress_out(s["signed"][on], hz)[0]
                bio_sigs.append((_displacement(v, fps, f0), fps, f0))
        figures.gold_standard_core(
            human_sigs, bio_sigs, OUT / "gold_standard_signed",
            space_label="displacement (integral of signed flow, ~ pressure)",
            icp=figures.pressure_templates())
        print("built gold_standard_signed")
    else:
        print("(signed_*.pkl not found — run run_signed_projection.py to enable the displacement figure)")


if __name__ == "__main__":
    main()
