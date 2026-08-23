"""Build the three comparison figures from the pickled analysis results.

Expects $PULS_WORK/out/{bio_rep1,bio_rep2,bio_rep3,human1,human2}_full.pkl
(produced by run_bioreactor.py / run_human.py). Writes PNG+PDF to the same dir:

    ReplicateComparison3        bioreactor 3-replicate comparison
    comparison_pooled           human cortex vs bioreactor (pooled over reps)
    Human_replicate_comparison  human 1 vs human 2 (rate-matched, separate traces)
"""
import os, sys, pickle
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work"))
OUT = WORK / "out"


def load(name):
    return pickle.load(open(OUT / f"{name}_full.pkl", "rb"))


def main():
    b1, b2, b3 = load("bio_rep1"), load("bio_rep2"), load("bio_rep3")
    h1, h2 = load("human1"), load("human2")

    figures.replicate_comparison_n(
        [dict(label=f"Rep {i}", results=b["results"], fps=b["fps"],
              color=b["color"], roi_fracs=b["roi_fracs"])
         for i, b in [(1, b1), (2, b2), (3, b3)]],
        OUT / "ReplicateComparison3")

    figures.comparison_pooled_figure(
        h1["cortex"], h1["resp"],
        [dict(results=b["results"], fps=b["fps"]) for b in (b1, b2, b3)],
        h1["fps"], OUT / "comparison_pooled")

    figures.human_replicate_figure(
        [dict(label="Human 1", cortex=h1["cortex"], resp=h1["resp"], fps=h1["fps"],
              color=h1["color"], roi_frac=h1["roi_frac"]),
         dict(label="Human 2", cortex=h2["cortex"], resp=h2["resp"], fps=h2["fps"],
              color=h2["color"], roi_frac=h2["roi_frac"])],
        OUT / "Human_replicate_comparison")
    print("built ReplicateComparison3, comparison_pooled, Human_replicate_comparison (png+pdf)")


if __name__ == "__main__":
    main()
