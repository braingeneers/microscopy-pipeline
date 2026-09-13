"""Figure 2 (paper, Nature style): human cortex vs bioreactor pulsatility.

Human side pools replicates 1, 2 and 4 with EQUAL per-cycle weight (human 4 is the
surgical/hands-in-frame recording, clean runs recovered by run_human_surgical.py).
Bioreactor side pools all organoids across the 3 replicates.

Emits a self-contained bundle: small PNG preview, vector PDF, editable PPTX
(SVG -> PowerPoint via LibreOffice), and one CSV of the underlying data per panel,
all zipped. Panels:
    a  mean cardiac pulse (ensemble mean +/- SD), human vs bioreactor superimposed
    b  frequency spectrum, human vs bioreactor
    c  pulsatility magnitude      d  time above trough+1SD
    e  time below trough+1SD      f  inter-beat interval

Usage:  python paper_fig2.py            # reads $PULS_WORK/out, writes .../out/fig2_bundle
"""
import os, sys, csv, shutil, zipfile, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures
from figures import _bandlimit, _cycles, _spectrum, pulse_features, resp_correct, _pool_bioreactor, BLUE, RED
from pulsatility import highpass
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

WORK = Path(os.environ.get("PULS_WORK", "./pulsatility_work")); OUT = WORK / "out"
PER = 100; NP = 4; BPM = np.linspace(0, 260, 400)
load = lambda n: pickle.load(open(OUT / f"{n}.pkl", "rb"))


# ---- data --------------------------------------------------------------------
def _rep_stats(sig, fps, f0):
    F = pulse_features(sig, fps, f0)
    C = _cycles(sig, fps, f0, PER)
    Cn = C / (np.percentile(np.abs(C), 99) or 1) if len(C) else np.zeros((0, PER))
    f, p = _spectrum(sig, fps); spec = np.interp(BPM, f * 60, p / (p.max() or 1), left=0, right=0)
    isi = F["isi"] / np.median(F["isi"]) if len(F["isi"]) else F["isi"]
    return dict(mag=F["mag"], ta=F["ta"], tb=F["tb"], isi=isi, cyc=Cn, spec=spec, rate=f0 * 60)


def _pool_humans(reps_12, surg_runs, hp=12.0):
    reps = []
    for d in reps_12:
        ctx, resp, fps = d["cortex"], d["resp"], d["fps"]
        sig = resp_correct(_bandlimit(highpass(ctx.detrended, fps, hp), fps, ctx.dominant_hz), resp)
        reps.append(_rep_stats(sig, fps, ctx.dominant_hz))
    rs = [_rep_stats(r["signal"], r["fps"], r["f0"]) for r in surg_runs]   # human 4: pre-band-limited runs
    reps.append(dict(mag=np.concatenate([r["mag"] for r in rs]), ta=np.concatenate([r["ta"] for r in rs]),
                     tb=np.concatenate([r["tb"] for r in rs]), isi=np.concatenate([r["isi"] for r in rs]),
                     cyc=np.vstack([r["cyc"] for r in rs]), spec=np.mean([r["spec"] for r in rs], 0),
                     rate=float(np.median([r["rate"] for r in rs]))))
    cat = lambda k: np.concatenate([r[k] for r in reps])
    allc = np.vstack([r["cyc"] for r in reps])                             # equal per-cycle pool
    return dict(mag=cat("mag"), ta=cat("ta"), tb=cat("tb"), isi=cat("isi"),
                cyc_mean=allc.mean(0), cyc_sd=allc.std(0), n_cyc=len(allc),
                spec=np.mean([r["spec"] for r in reps], 0), bpm=BPM,
                rate=float(np.median([r["rate"] for r in reps])), n_h=len(reps))


def _tile(mean, sd, n):
    x = np.linspace(0, n, len(mean) * n)
    return x, np.tile(mean, n), np.tile(sd, n)


# ---- figure ------------------------------------------------------------------
def _style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5, "axes.linewidth": 0.6, "axes.titlesize": 8, "axes.titlepad": 3,
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "legend.fontsize": 6.8, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
    })


def _panel(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def _box(ax, data, ylabel, title):
    bp = ax.boxplot(data, widths=0.62, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.0), whiskerprops=dict(lw=0.6),
                    capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for patch, c in zip(bp["boxes"], (BLUE, RED)):
        patch.set_facecolor(c); patch.set_alpha(0.45); patch.set_edgecolor(c)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Human", "Bioreactor"])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.13, axis="y", lw=0.5)


def build(H, B, out):
    _style()
    hbpm, bbpm = H["rate"], float(np.median(B["rep_bpm"]))
    xh, hm, hs = _tile(H["cyc_mean"], H["cyc_sd"], NP)
    xb, bm, bs = _tile(B["cyc_mean"], B["cyc_sd"], NP)
    fig = plt.figure(figsize=(7.2, 4.3))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.5,
                  top=0.9, bottom=0.12, left=0.07, right=0.985)

    # a: superimposed mean pulse +/- SD
    axa = fig.add_subplot(gs[0, 0:2])
    axa.fill_between(xh, hm - hs, hm + hs, color=BLUE, alpha=0.15, lw=0)
    axa.fill_between(xb, bm - bs, bm + bs, color=RED, alpha=0.15, lw=0)
    axa.plot(xh, hm, color=BLUE, lw=1.3, label="Human")
    axa.plot(xb, bm, color=RED, lw=1.3, label="Bioreactor")
    axa.axhline(0, color="k", lw=0.4, alpha=0.4)
    axa.set_xlim(0, NP); axa.set_xticks(range(NP + 1)); axa.set_yticks([])
    axa.set_xlabel("Cardiac cycle (rate-standardized)"); axa.set_ylabel("Normalized motion")
    axa.set_title("Mean cardiac pulse ± SD")
    axa.legend(loc="upper left", frameon=False, handlelength=1.3, borderaxespad=0.2)
    _panel(axa, "a")

    # b: frequency spectrum
    axb = fig.add_subplot(gs[0, 2:4])
    axb.plot(H["bpm"], H["spec"] / (H["spec"].max() or 1), color=BLUE, lw=1.1, label="Human")
    axb.plot(B["bpm"], B["spec"] / (B["spec"].max() or 1), color=RED, lw=1.1, label="Bioreactor")
    for v, c in [(hbpm, BLUE), (bbpm, RED)]:
        axb.axvline(v, color=c, ls="--", lw=0.6, alpha=0.6)
    axb.set_xlim(0, 240); axb.set_yticks([]); axb.set_xlabel("Rate (bpm)")
    axb.set_ylabel("Power (norm.)"); axb.set_title("Frequency spectrum")
    axb.legend(loc="upper right", frameon=False, handlelength=1.4)
    _panel(axb, "b")

    # c-f: box plots
    specs = [("c", [H["mag"], B["mag"]], "px per frame", "Pulsatility magnitude"),
             ("d", [H["ta"], B["ta"]], "% of cycle", "Time above trough+1SD"),
             ("e", [H["tb"], B["tb"]], "% of cycle", "Time below trough+1SD"),
             ("f", [H["isi"], B["isi"]], "× median cycle", "Inter-beat interval")]
    for i, (letter, data, ylab, title) in enumerate(specs):
        ax = fig.add_subplot(gs[1, i]); _box(ax, data, ylab, title); _panel(ax, letter)

    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}_preview.png", dpi=140, bbox_inches="tight")
    fig.savefig(f"{out}_hires.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- CSV + bundle ------------------------------------------------------------
def _write_csvs(H, B, cdir):
    cdir.mkdir(parents=True, exist_ok=True)
    ph = np.linspace(0, 1, PER, endpoint=False)
    with open(cdir / "panel_a_mean_waveform.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cardiac_phase", "human_mean", "human_sd", "bioreactor_mean", "bioreactor_sd"])
        for i in range(PER):
            w.writerow([f"{ph[i]:.4f}", f"{H['cyc_mean'][i]:.6f}", f"{H['cyc_sd'][i]:.6f}",
                        f"{B['cyc_mean'][i]:.6f}", f"{B['cyc_sd'][i]:.6f}"])
    with open(cdir / "panel_b_frequency_spectrum.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rate_bpm", "human_power_norm", "bioreactor_power_norm"])
        hs = H["spec"] / (H["spec"].max() or 1); bsp = B["spec"] / (B["spec"].max() or 1)
        for i in range(len(BPM)):
            w.writerow([f"{BPM[i]:.3f}", f"{hs[i]:.6f}", f"{bsp[i]:.6f}"])
    for name, key in [("panel_c_pulsatility_magnitude", "mag"), ("panel_d_time_above_pct", "ta"),
                      ("panel_e_time_below_pct", "tb"), ("panel_f_interbeat_interval", "isi")]:
        with open(cdir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["group", "value"])
            for grp, arr in [("human", H[key]), ("bioreactor", B[key])]:
                for v in arr:
                    w.writerow([grp, f"{float(v):.6f}"])


README = (
    "Figure 2 — Human cortex vs bioreactor pulsatility\n"
    "==================================================\n\n"
    "Human = replicates 1, 2 and 4 pooled with EQUAL per-cycle weight (replicate 4 is the\n"
    "surgical/hands-in-frame recording; clean hand-free runs recovered before pooling).\n"
    "Bioreactor = all organoids pooled across 3 replicates.\n\n"
    "Files:\n"
    "  Figure2.pdf          vector figure (submission)\n"
    "  Figure2.svg          vector source (open in Illustrator/Inkscape, or Insert into\n"
    "                       PowerPoint 365 and 'Convert to Shape' for vector editing)\n"
    "  Figure2.pptx         editable slide: 300-dpi figure + editable caption text box\n"
    "  Figure2_preview.png  raster preview\n"
    "  csv/                 one file per panel with the plotted data\n\n"
    "Panels: a mean cardiac pulse ±SD (rate-standardized ensemble, superimposed);\n"
    "b frequency spectrum; c pulsatility magnitude; d time above trough+1SD;\n"
    "e time below trough+1SD; f inter-beat interval. Waveform cycles are per-cycle\n"
    "amplitude-normalized; panel a shows one mean cycle tiled for display.\n"
)


def make_pptx(png_hi, outpath, caption):
    """Editable slide: high-res figure placed full-bleed + an editable caption text
    box. (Curve-level vector editing is via the bundled SVG/PDF: PowerPoint 365 can
    Insert > Pictures the SVG then 'Convert to Shape'; SVG also opens in Illustrator/
    Inkscape. LibreOffice is non-functional in this build for direct conversion.)"""
    try:
        from PIL import Image
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError:
        print("(python-pptx not installed -> skipping .pptx; pip install python-pptx)"); return False
    w_px, h_px = Image.open(png_hi).size; dpi = 300.0
    w_in, h_in = w_px / dpi, h_px / dpi
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(w_in), Inches(h_in + 0.55)
    slide = prs.slides.add_slide(prs.slide_layouts[6])       # blank
    slide.shapes.add_picture(str(png_hi), Inches(0), Inches(0), width=Inches(w_in), height=Inches(h_in))
    tb = slide.shapes.add_textbox(Inches(0.12), Inches(h_in + 0.04), Inches(w_in - 0.24), Inches(0.48))
    tf = tb.text_frame; tf.word_wrap = True; tf.text = caption
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.size = Pt(7)
    prs.save(str(outpath))
    return Path(outpath).exists()


def main():
    h1, h2 = load("human1_full"), load("human2_full")
    srec = load("human4_surgical")
    B = _pool_bioreactor([dict(results=load(f"bio_rep{i}_full")["results"],
                               fps=load(f"bio_rep{i}_full")["fps"]) for i in (1, 2, 3)], per=PER)
    H = _pool_humans([dict(cortex=h1["cortex"], resp=h1["resp"], fps=h1["fps"]),
                      dict(cortex=h2["cortex"], resp=h2["resp"], fps=h2["fps"])], srec["runs"])

    bdir = OUT / "fig2_bundle";
    if bdir.exists(): shutil.rmtree(bdir)
    bdir.mkdir(parents=True)
    stem = bdir / "Figure2"
    build(H, B, stem)
    _write_csvs(H, B, bdir / "csv")
    (bdir / "README.txt").write_text(README)
    bbpm = float(np.median(B["rep_bpm"]))
    caption = (f"Figure 2 | Parenchymal pulsatility, human cortex vs bioreactor. Human = replicates "
               f"1, 2 and 4 pooled with equal per-cycle weight ({H['n_cyc']} beats, {H['rate']:.0f} bpm median); "
               f"bioreactor = {B['n_org']} organoids across 3 replicates ({B['n_cyc']} beats, {bbpm:.0f} bpm). "
               f"a, Mean cardiac pulse ± SD (rate-standardized ensemble, superimposed). b, Frequency spectrum. "
               f"c, Pulsatility magnitude. d,e, Time above/below trough+1SD. f, Inter-beat interval.")
    ok = make_pptx(bdir / "Figure2_hires.png", bdir / "Figure2.pptx", caption)
    (bdir / "Figure2_hires.png").unlink()                    # embed-only; keep the bundle tidy

    zpath = OUT / "Figure2_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bdir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(bdir.parent))
    print(f"human: {H['n_h']} reps, {H['n_cyc']} beats, {H['rate']:.0f} bpm | "
          f"bioreactor: {B['n_org']} organoids, {B['n_cyc']} beats, {bbpm:.0f} bpm")
    print(f"pptx: {'ok' if ok else 'MISSING'} | bundle: {zpath}")


if __name__ == "__main__":
    main()
