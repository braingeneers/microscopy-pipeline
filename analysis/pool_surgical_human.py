"""Single-run vs pooled ensemble pulse for a surgical (hands-in-frame) human.

A hand-contaminated recording yields only short (~15-25 s) hand-free runs. Each run
alone gives a noisy mean pulse; pooling the cardiac CYCLES across runs -- after
phase-warping every cycle to a common cardiac phase and normalizing each run's
amplitude -- sharpens the *mean* pulse (its standard error falls ~sqrt(N)) without
distorting its shape, as long as the runs share a reproducible rate. It does not
shrink the genuine beat-to-beat SD band, and per-cycle reproducibility (cycle-R2)
can dip slightly because pooled runs span different intra-operative contexts.

Reads $PULS_WORK/out/<key>_surgical.pkl (run_human_surgical.py) and writes
$PULS_WORK/out/<key>_pooled.{png,pdf}.

Usage:
    python pool_surgical_human.py [key]        # key defaults to human4
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures
from figures import BLUE, GREEN, PURPLE
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work"))
OUT = WORK / "out"
PER = 100


def _cycles_of(run):
    """Phase-warped, per-run amplitude-normalized cardiac cycles from one clean run."""
    z = (run["signal"] - run["signal"].mean()) / (run["signal"].std() or 1)
    return figures._cycles(z, run["fps"], run["f0"], per=PER)


def _ensemble(cyc, shift=0):
    m, sd, n = cyc.mean(0), cyc.std(0), len(cyc)
    se = sd / np.sqrt(n)
    sstot = ((cyc - cyc.mean(1, keepdims=True)) ** 2).sum(1)
    ssres = ((cyc - m) ** 2).sum(1); r2 = 1 - ssres / np.where(sstot > 0, sstot, 1)
    m, sd, se = np.roll(m, shift), np.roll(sd, shift), np.roll(se, shift)
    return m, sd, se, n, float(np.median(r2))


def main(key="human4"):
    d = pickle.load(open(OUT / f"{key}_surgical.pkl", "rb"))
    runs = d["runs"]
    if len(runs) < 2:
        print(f"{key}: need >=2 clean runs to pool (have {len(runs)})"); return
    per_run = [dict(run=r, cyc=_cycles_of(r)) for r in runs]
    per_run = [p for p in per_run if len(p["cyc"])]
    longest = max(per_run, key=lambda p: p["run"]["dur_s"])
    single = longest["cyc"]
    pooled = np.vstack([p["cyc"] for p in per_run])
    shift = int(0.15 * PER) - int(np.argmax(pooled.mean(0)))     # peak -> phase ~0.15, shared
    ms, sds, ses, ns, r2s = _ensemble(single, shift)
    mp, sdp, sep, npool, r2p = _ensemble(pooled, shift)
    ph = np.linspace(0, 1, PER); rates = [p["run"]["f0"] * 60 for p in per_run]

    print(f"{key}: {len(per_run)} runs, rate {min(rates):.0f}-{max(rates):.0f} bpm (median {np.median(rates):.0f})")
    print(f"  single longest: {longest['run']['dur_s']:.0f}s, {ns} beats, cycle-R2 {r2s:.2f}, mean SE {ses.mean():.3f}")
    print(f"  pooled:         {npool} beats from {len(per_run)} runs, cycle-R2 {r2p:.2f}, mean SE {sep.mean():.3f}")
    print(f"  SE shrink {ses.mean() / sep.mean():.2f}x (theory sqrt({npool}/{ns})={np.sqrt(npool / ns):.2f})")

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1.0, 0.75], hspace=0.42, wspace=0.22,
                  top=0.9, bottom=0.09, left=0.09, right=0.97)
    ylo = min(mp.min() - sdp.max(), ms.min() - sds.max()) * 1.05
    yhi = max(mp.max() + sdp.max(), ms.max() + sds.max()) * 1.05
    for col, (m, sd, se, n, r2, ttl) in enumerate([
            (ms, sds, ses, ns, r2s, f"Single best run\n{longest['run']['dur_s']:.0f} s, {ns} beats"),
            (mp, sdp, sep, npool, r2p, f"Pooled across {len(per_run)} clean runs\n{npool} beats")]):
        ax = fig.add_subplot(gs[0, col])
        ax.fill_between(ph, m - sd, m + sd, color=BLUE, alpha=0.15, label="±1 SD (beat-to-beat)")
        ax.fill_between(ph, m - se, m + se, color=BLUE, alpha=0.45, label="±1 SE (of the mean)")
        ax.plot(ph, m, color="#0b3d91", lw=2, label="mean pulse")
        ax.set_title(ttl); ax.set_xlabel("cardiac phase"); ax.set_ylabel("normalized tissue speed (z)")
        ax.set_ylim(ylo, yhi)
        ax.text(0.02, 0.04, f"median cycle-R² {r2:.2f}\nmean SE {se.mean():.3f}", transform=ax.transAxes,
                fontsize=9.5, va="bottom", bbox=dict(facecolor="white", edgecolor="0.8", pad=3))
        if col == 1:
            ax.legend(fontsize=8.5, loc="upper right")

    axo = fig.add_subplot(gs[1, 0])
    axo.plot(ph, ms, color=PURPLE, lw=1.6, label=f"single ({ns} beats)")
    axo.plot(ph, mp, color="#0b3d91", lw=2.2, label=f"pooled ({npool} beats)")
    axo.set_title("Mean pulse shape — single vs pooled (do they agree?)", fontsize=11)
    axo.set_xlabel("cardiac phase"); axo.set_ylabel("z"); axo.legend(fontsize=8.5)

    axr = fig.add_subplot(gs[1, 1])
    axr.hist(rates, bins=np.arange(35, 76, 3), color=GREEN, alpha=0.8)
    axr.axvline(np.median(rates), color="k", ls="--", lw=0.9)
    axr.set_title(f"Per-run rate — reproducible ~{np.median(rates):.0f} bpm", fontsize=11)
    axr.set_xlabel("bpm"); axr.set_ylabel("# runs")

    fig.suptitle(f"{key} (surgical) — pooling clean runs sharpens the mean pulse (SE) without distorting its shape",
                 y=0.965, fontsize=13.5)
    figures._save(fig, OUT / f"{key}_pooled")
    print(f"built {key}_pooled")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "human4")
