"""Figure 2 (paper, Nature style): human cortex vs bioreactor pulsatility.

Human side pools patients 1, 2 and 4 with EQUAL PER-PATIENT weight (patient 4 is the
surgical/hands-in-frame recording, clean runs recovered by run_human_surgical.py).
Bioreactor side pools all organoids across the 3 replicates.

Emits a self-contained bundle: small PNG preview, vector PDF, editable PPTX
(300-dpi figure + editable caption; python-pptx), and one CSV of the underlying data
per panel, all zipped. Two columns of rhythm strips -- LEFT signed velocity
(~ fluid flow), RIGHT displacement (its time-integral ~ parenchymal pressure) --
then an analysis row:
    a/b human velocity/displacement    c/d bioreactor velocity/displacement
    e/f velocity/displacement overlay  g frequency spectrum  h magnitude  i inter-beat
Both waveform columns use the signed-projection pkls (patients 1/2 + organoids); the
surgical patient 4 (no reliable directional projection) enters only g/h/i.

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
PER = 100; BPM = np.linspace(0, 260, 400)
load = lambda n: pickle.load(open(OUT / f"{n}.pkl", "rb"))


# ---- data --------------------------------------------------------------------
def _rep_stats(sig, fps, f0):
    F = pulse_features(sig, fps, f0)
    C = _cycles(sig, fps, f0, PER)
    Cn = C / (np.percentile(np.abs(C), 99) or 1) if len(C) else np.zeros((0, PER))
    f, p = _spectrum(sig, fps); spec = np.interp(BPM, f * 60, p / (p.max() or 1), left=0, right=0)
    isi = F["isi"] / np.median(F["isi"]) if len(F["isi"]) else F["isi"]
    return dict(mag=F["mag"], ta=F["ta"], tb=F["tb"], isi=isi, cyc=Cn, spec=spec, rate=f0 * 60)


def _pool_humans(reps_12, surg_runs, hp=12.0, seed=0):
    """Pool human patients 1, 2 and 4 with EQUAL PER-PATIENT weight (patient 4 is the
    surgical recording, its clean runs pooled into one patient). The mean pulse is a
    per-patient-weighted average (each patient's cycles sum to weight 1); box-plot
    metrics subsample each patient to a common count so none dominates by beat count."""
    rng = np.random.default_rng(seed)
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
    R = len(reps)
    cyc_list = [r["cyc"] for r in reps]
    allc = np.vstack(cyc_list)
    w = np.concatenate([np.full(len(c), 1.0 / (R * len(c))) for c in cyc_list])   # each patient -> weight 1
    cyc_mean = np.average(allc, axis=0, weights=w)
    cyc_sd = np.sqrt(np.average((allc - cyc_mean) ** 2, axis=0, weights=w))
    def eq(key):                                        # equal-count subsample -> per-patient-equal
        arrs = [r[key] for r in reps if len(r[key])]
        if not arrs:
            return np.array([])
        k = min(len(a) for a in arrs)
        return np.concatenate([rng.choice(a, k, replace=False) for a in arrs])
    return dict(mag=eq("mag"), ta=eq("ta"), tb=eq("tb"), isi=eq("isi"),
                cyc_mean=cyc_mean, cyc_sd=cyc_sd, n_cyc=len(allc),
                spec=np.mean([r["spec"] for r in reps], 0), bpm=BPM,
                rate=float(np.median([r["rate"] for r in reps])), n_h=R)


def _tile(mean, sd, n):
    x = np.linspace(0, n, len(mean) * n)
    return x, np.tile(mean, n), np.tile(sd, n)


# ---- signed velocity (~ flow) and its integral, displacement (~ pressure) ----
def _signed_cycles(signed_v, fps, f0):
    """Per-cycle SIGNED velocity (~ fluid flow): band-limited, oriented so the
    systolic excursion is positive, then cut into amplitude-normalized cycles."""
    from scipy.stats import skew
    x = _bandlimit(signed_v, fps, f0)
    x = x if skew(x) >= 0 else -x
    C = _cycles(x, fps, f0, PER)
    return C / (np.percentile(np.abs(C), 99) or 1) if len(C) else np.zeros((0, PER))


def _integrate_cycles(Vc):
    """Displacement (~ pressure) = time-integral of each velocity cycle. Integration
    is intrinsically low-pass, so this gives the smooth pressure-like pulse (no post
    high-pass/band-pass, unlike the earlier per-cycle-filtered version)."""
    if not len(Vc):
        return Vc
    D = np.cumsum(Vc - Vc.mean(1, keepdims=True), axis=1)
    s = np.percentile(np.abs(D), 99, axis=1, keepdims=True); s[s == 0] = 1
    return D / s


def _pool_disp(units):
    """Per-unit-equal weighted mean/SD over a list of per-unit cycle matrices."""
    units = [c for c in units if len(c)]
    U = len(units); allc = np.vstack(units)
    w = np.concatenate([np.full(len(c), 1.0 / (U * len(c))) for c in units])
    m = np.average(allc, axis=0, weights=w)
    sd = np.sqrt(np.average((allc - m) ** 2, axis=0, weights=w))
    return dict(cyc_mean=m, cyc_sd=sd, n_cyc=len(allc), n=U)


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


def _smooth(y, k=5):
    if k <= 1:
        return y
    return np.convolve(np.pad(y, k // 2, mode="edge"), np.ones(k) / k, "valid")[:len(y)]


def build(HV, BV, HD, BD, H, B, out):
    _style()
    NPw = 5                                            # beats shown per waveform strip

    def _train(P):                                     # roll peak ~1/3 into beat, tile
        sh = int(0.30 * PER) - int(np.argmax(P["cyc_mean"]))
        return _tile(np.roll(P["cyc_mean"], sh), np.roll(P["cyc_sd"], sh), NPw)
    hv_, bv_ = _train(HV), _train(BV)                  # signed-velocity trains (x, mean, sd)
    # displacement: put the human peak at ~0.3, then align the bioreactor to it by
    # circular cross-correlation so the two curves line up peak-to-peak (comparability)
    tgt = int(0.30 * PER)
    sh_h = tgt - int(np.argmax(HD["cyc_mean"]))
    ref = np.roll(HD["cyc_mean"], sh_h); ref = ref - ref.mean()
    bx = BD["cyc_mean"] - BD["cyc_mean"].mean()
    sh_b = int(np.argmax([np.dot(np.roll(bx, k), ref) for k in range(PER)]))
    hd_ = _tile(np.roll(HD["cyc_mean"], sh_h), np.roll(HD["cyc_sd"], sh_h), NPw)
    bd_ = _tile(np.roll(BD["cyc_mean"], sh_b), np.roll(BD["cyc_sd"], sh_b), NPw)

    def _lims(*tr):
        return (min((m - s).min() for _, m, s in tr) * 1.12,
                max((m + s).max() for _, m, s in tr) * 1.12)
    vlo, vhi = _lims(hv_, bv_); dlo, dhi = _lims(hd_, bd_)
    vyt = [t for t in (-1.0, 0.0, 1.0) if vlo - 0.05 <= t <= vhi + 0.05] or [0.0, 1.0]
    dyt = [t for t in (-1.0, 0.0, 1.0) if dlo - 0.05 <= t <= dhi + 0.05] or [0.0, 1.0]

    fig = plt.figure(figsize=(8.4, 6.7))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.7, 0.82], hspace=0.5,
                     top=0.94, bottom=0.07, left=0.085, right=0.98)
    gt = outer[0].subgridspec(3, 2, hspace=0.2, wspace=0.28)
    gb = outer[1].subgridspec(1, 3, wspace=0.52)

    def _strip(ax, tr, color, ylo, yhi, yt, ylabel=None):
        x, m, s = tr
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, lw=0)
        ax.plot(x, m, color=color, lw=1.5)
        ax.axhline(0, color="0.65", lw=0.5, zorder=0)
        ax.set_xlim(0, NPw); ax.set_xticks(range(NPw + 1)); ax.set_ylim(ylo, yhi); ax.set_yticks(yt)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xticklabels([])

    def _ov(ax, trh, trb, ylo, yhi, yt, ylabel=None, legend=False):
        ax.plot(trh[0], trh[1], color=BLUE, lw=1.4, label="Human")
        ax.plot(trb[0], trb[1], color=RED, lw=1.4, label="Bioreactor")
        ax.axhline(0, color="0.65", lw=0.5, zorder=0)
        ax.set_xlim(0, NPw); ax.set_xticks(range(NPw + 1)); ax.set_ylim(ylo, yhi); ax.set_yticks(yt)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xlabel("Cardiac cycles (rate-standardized)")
        if legend:
            ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.2, fontsize=7, columnspacing=1.0)

    # left column = signed velocity (~ fluid flow); right column = displacement (~ pressure)
    axa = fig.add_subplot(gt[0, 0]); _strip(axa, hv_, BLUE, vlo, vhi, vyt, "Human\n(norm.)"); axa.set_title("Signed velocity  (~ fluid flow)"); _panel(axa, "a")
    axb = fig.add_subplot(gt[0, 1]); _strip(axb, hd_, BLUE, dlo, dhi, dyt); axb.set_title("Displacement  (~ pressure)"); _panel(axb, "b")
    axc = fig.add_subplot(gt[1, 0]); _strip(axc, bv_, RED, vlo, vhi, vyt, "Bioreactor\n(norm.)"); _panel(axc, "c")
    axd = fig.add_subplot(gt[1, 1]); _strip(axd, bd_, RED, dlo, dhi, dyt); _panel(axd, "d")
    axe = fig.add_subplot(gt[2, 0]); _ov(axe, hv_, bv_, vlo, vhi, vyt, "Overlay\n(norm.)", legend=True); _panel(axe, "e")
    axf = fig.add_subplot(gt[2, 1]); _ov(axf, hd_, bd_, dlo, dhi, dyt); _panel(axf, "f")

    # analysis row: g frequency spectrum (speed), h pulsatility magnitude, i inter-beat interval
    axg = fig.add_subplot(gb[0])
    axg.plot(H["bpm"], _smooth(H["spec"] / (H["spec"].max() or 1)), color=BLUE, lw=1.3, label="Human")
    axg.plot(B["bpm"], _smooth(B["spec"] / (B["spec"].max() or 1)), color=RED, lw=1.3, label="Bioreactor")
    axg.set_xlim(0, 180); axg.set_ylim(bottom=0); axg.set_yticks([]); axg.set_xlabel("Rate (bpm)")
    axg.set_ylabel("Power"); axg.set_title("Frequency spectrum")
    axg.legend(loc="upper right", frameon=False, handlelength=1.3, fontsize=7.5)
    _panel(axg, "g")
    axh = fig.add_subplot(gb[1]); _box(axh, [H["mag"], B["mag"]], "px per frame", "Pulsatility magnitude"); _panel(axh, "h")
    axi = fig.add_subplot(gb[2]); _box(axi, [H["isi"], B["isi"]], "× median cycle", "Inter-beat interval"); _panel(axi, "i")

    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}_preview.png", dpi=140, bbox_inches="tight")
    fig.savefig(f"{out}_hires.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- CSV + bundle ------------------------------------------------------------
def _wave_csv(path, P, Q):
    ph = np.linspace(0, 1, PER, endpoint=False)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cardiac_phase", "human_mean", "human_sd", "bioreactor_mean", "bioreactor_sd"])
        for i in range(PER):
            w.writerow([f"{ph[i]:.4f}", f"{P['cyc_mean'][i]:.6f}", f"{P['cyc_sd'][i]:.6f}",
                        f"{Q['cyc_mean'][i]:.6f}", f"{Q['cyc_sd'][i]:.6f}"])


def _write_csvs(HV, BV, HD, BD, H, B, cdir):
    cdir.mkdir(parents=True, exist_ok=True)
    _wave_csv(cdir / "signed_velocity_mean_waveforms.csv", HV, BV)
    _wave_csv(cdir / "displacement_mean_waveforms.csv", HD, BD)
    with open(cdir / "frequency_spectrum.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rate_bpm", "human_power_norm", "bioreactor_power_norm"])
        hs = H["spec"] / (H["spec"].max() or 1); bsp = B["spec"] / (B["spec"].max() or 1)
        for i in range(len(BPM)):
            w.writerow([f"{BPM[i]:.3f}", f"{hs[i]:.6f}", f"{bsp[i]:.6f}"])
    for name, key in [("pulsatility_magnitude", "mag"), ("interbeat_interval", "isi")]:
        with open(cdir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["group", "value"])
            for grp, arr in [("human", H[key]), ("bioreactor", B[key])]:
                for v in arr:
                    w.writerow([grp, f"{float(v):.6f}"])


README = (
    "Figure 2 — Human cortex vs bioreactor pulsatility\n"
    "==================================================\n\n"
    "Human = patients 1, 2 and 4 pooled with EQUAL PER-PATIENT weight (patient 4 is the\n"
    "surgical/hands-in-frame recording; clean hand-free runs recovered and treated as one\n"
    "patient). The mean pulse is a per-patient-weighted average; box-plot metrics subsample\n"
    "each patient to a common count. Bioreactor = all organoids pooled across 3 replicates.\n\n"
    "Measured signal = rectified TISSUE SPEED: the mean optical-flow magnitude inside the ROI\n"
    "per frame (px/frame; a velocity magnitude, NOT displacement). It approximates\n"
    "|d/dt of parenchymal displacement|; displacement itself (~ parenchymal pressure) is the\n"
    "separate signed-projection analysis. Waveform panels are per-cycle amplitude-normalized.\n\n"
    "Files:\n"
    "  Figure2.pdf          vector figure (submission)\n"
    "  Figure2.svg          vector source (open in Illustrator/Inkscape, or Insert into\n"
    "                       PowerPoint 365 and 'Convert to Shape' for vector editing)\n"
    "  Figure2.pptx         editable slide: 300-dpi figure + editable caption text box\n"
    "  Figure2_preview.png  raster preview\n"
    "  csv/                 one file per panel with the plotted data\n\n"
    "Layout: LEFT column = signed velocity (~ fluid FLOW), RIGHT column = displacement\n"
    "(its integral, ~ PRESSURE). a,b human; c,d bioreactor; e,f overlays; g frequency\n"
    "spectrum; h pulsatility magnitude; i inter-beat interval. Waveforms are per-cycle\n"
    "amplitude-normalized, rate-standardized, tiled (5 beats). Both waveform columns pool\n"
    "the clean signed recordings (human 1/2 + organoids); the g/h/i rate-rhythm panels\n"
    "additionally include surgical patient 4.\n"
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


def _signed_pools(h1, h2, bf):
    """Signed velocity (~ flow) and its integral displacement (~ pressure) from the
    signed-projection pkls. Humans: patients 1 & 2 (clean 4K recordings, per-patient
    equal); the surgical patient 4 is not here -- directional projection is unreliable
    with hands moving through the field. Bioreactor: per organoid."""
    hu = []
    for key, full in [("human1", h1), ("human2", h2)]:
        s = load(f"signed_{key}"); hu.append(_signed_cycles(s["signed"]["cortex"], s["fps"], full["cortex"].dominant_hz))
    bu = []
    for i in (1, 2, 3):
        s = load(f"signed_bio{i}"); f0 = bf[i]["results"]["within-vessel"].dominant_hz; hz = s["signed"]["housing"]
        for on in [n for n in s["names"] if len(n) == 2 and n[0] == "o" and n[1].isdigit()]:
            bu.append(_signed_cycles(figures._regress_out(s["signed"][on], hz)[0], s["fps"], f0))
    HV, BV = _pool_disp(hu), _pool_disp(bu)                              # signed velocity
    HD = _pool_disp([_integrate_cycles(c) for c in hu])                 # displacement
    BD = _pool_disp([_integrate_cycles(c) for c in bu])
    return HV, BV, HD, BD


def main():
    h1, h2 = load("human1_full"), load("human2_full")
    srec = load("human4_surgical")
    bf = {i: load(f"bio_rep{i}_full") for i in (1, 2, 3)}
    B = _pool_bioreactor([dict(results=bf[i]["results"], fps=bf[i]["fps"]) for i in (1, 2, 3)], per=PER)
    H = _pool_humans([dict(cortex=h1["cortex"], resp=h1["resp"], fps=h1["fps"]),
                      dict(cortex=h2["cortex"], resp=h2["resp"], fps=h2["fps"])], srec["runs"])
    HV, BV, HD, BD = _signed_pools(h1, h2, bf)

    bdir = OUT / "fig2_bundle";
    if bdir.exists(): shutil.rmtree(bdir)
    bdir.mkdir(parents=True)
    stem = bdir / "Figure2"
    build(HV, BV, HD, BD, H, B, stem)
    _write_csvs(HV, BV, HD, BD, H, B, bdir / "csv")
    (bdir / "README.txt").write_text(README)
    bbpm = float(np.median(B["rep_bpm"]))
    caption = (f"Figure 2 | Parenchymal pulsatility, human cortex vs bioreactor. Left column, signed tissue "
               f"velocity (optical flow projected on the principal motion axis; correlates with net fluid "
               f"FLOW); right column, its time-integral, displacement (correlates with parenchymal PRESSURE). "
               f"Both pool the clean signed recordings -- human patients 1 & 2 (per-patient equal) and "
               f"{B['n_org']} organoids ({BD['n']} units). a-d, Human and bioreactor mean pulse ± SD "
               f"(rate-standardized, 5 beats). e,f, Overlays. g, Frequency spectrum (rate); "
               f"h, pulsatility magnitude (px/frame); i, inter-beat interval -- these pool human patients "
               f"1, 2 and 4 ({H['n_cyc']} beats, {H['rate']:.0f} bpm) and the organoids ({B['n_cyc']} beats, "
               f"{bbpm:.0f} bpm). Surgical patient 4 is rate/rhythm only (no reliable directional projection).")
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
