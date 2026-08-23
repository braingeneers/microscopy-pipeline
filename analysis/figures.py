"""Rebuilt figure module (private-data figures) for the pulsatility comparisons.
Uses the committed `pulsatility` package for the heavy lifting; this file only
does presentation + light signal math."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences, peak_widths, hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
    "axes.titleweight": "normal", "figure.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 13,
})
BLUE, RED, GREEN, PURPLE = "#1f6feb", "#d1242f", "#2da44e", "#8250df"
GOLD = "#bf8700"
CROP = 30.0


# --------------------------------------------------------------------------- #
# small signal helpers
# --------------------------------------------------------------------------- #
def _spectrum(sig, fps):
    x = np.asarray(sig, float); x = x - x.mean()
    w = np.hanning(len(x))
    p = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / fps)
    return f, p


def _bandlimit(sig, fps, f0, lo_mult=0.5, hi_mult=3.5):
    nyq = fps / 2.0
    lo = max(f0 * lo_mult, 0.05) / nyq
    hi = min(f0 * hi_mult, nyq * 0.95) / nyq
    if not (0 < lo < hi < 1):
        return np.asarray(sig, float) - np.mean(sig)
    b, a = butter(3, [lo, hi], btype="band")
    return filtfilt(b, a, np.asarray(sig, float) - np.mean(sig))


def _regress_out(y, h):
    n = min(len(y), len(h)); y = np.asarray(y, float)[:n]; h = np.asarray(h, float)[:n]
    hc = h - h.mean(); denom = float(np.dot(hc, hc))
    if denom <= 0:
        return y.copy(), 0.0
    beta = float(np.dot(hc, y - y.mean()) / denom)
    return y - beta * hc, beta


def resp_correct(sig, resp):
    """Remove the additive respiration component from a (band-limited) cardiac trace."""
    return _regress_out(sig, resp.resp_wave)[0]


def pulse_features(sig, fps, f0):
    sig = np.asarray(sig, float)
    dist = max(1, int(round(fps / f0 * 0.6)))
    sd = np.std(sig); prom = 0.3 * sd if sd > 0 else None
    spc = fps / f0                                   # samples per cycle
    pk, _ = find_peaks(sig, distance=dist, prominence=prom)
    tr, _ = find_peaks(-sig, distance=dist, prominence=prom)
    out = dict(mag=np.array([]), pw=np.array([]), tw=np.array([]), isi=np.array([]))
    if len(pk) >= 2:
        out["mag"] = peak_prominences(sig, pk)[0]
        out["pw"] = peak_widths(sig, pk, rel_height=0.5)[0] / spc
        out["isi"] = np.diff(pk) / fps
    if len(tr) >= 1:
        out["tw"] = peak_widths(-sig, tr, rel_height=0.5)[0] / spc
    return out


def xcorr_lag_ms(reference, signal, fps, max_ms=250):
    """Lag of `signal` relative to `reference`; positive => signal lags reference."""
    a = np.asarray(reference, float); b = np.asarray(signal, float)
    n = min(len(a), len(b)); a = a[:n]; b = b[:n]
    a = (a - a.mean()) / (a.std() or 1); b = (b - b.mean()) / (b.std() or 1)
    maxlag = int(round(max_ms / 1000.0 * fps))
    lags = np.arange(-maxlag, maxlag + 1)
    c = []
    for k in lags:
        c.append(np.dot(a[:n - k], b[k:]) if k >= 0 else np.dot(a[-k:], b[:n + k]))
    return float(lags[int(np.argmax(c))]) / fps * 1000.0


def pulse_train(sig, peaks, fps, npulses=15, per=48):
    peaks = np.asarray(peaks); npulses = min(npulses, len(peaks) - 1)
    segs = []
    for i in range(npulses):
        seg = sig[peaks[i]:peaks[i + 1]]
        segs.append(np.interp(np.linspace(0, 1, per), np.linspace(0, 1, len(seg)), seg))
    train = np.concatenate(segs) if segs else np.zeros(per)
    train = train / (np.percentile(np.abs(train), 99) or 1)
    return np.linspace(0, npulses, len(train)), train


def _rate_match(times, f0_hz, f_ref_bpm=60.0):
    """Uniformly scale a real time axis so the pulse rate maps to ``f_ref_bpm``.

    Unlike per-cycle standardization, this is a single linear stretch of the time
    axis (real cycle-to-cycle timing is preserved), so two signals of different
    rates can be overlaid / compared at a common cadence. One reference-second then
    equals one pulse at ``f_ref_bpm``."""
    f_ref = f_ref_bpm / 60.0
    return np.asarray(times, float) * (f0_hz / f_ref)


def _headroom_legend(ax, ncol, loc="upper center", frac=0.55):
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + frac * (hi - lo))
    ax.legend(loc=loc, ncol=ncol, framealpha=0.92, handlelength=1.7, columnspacing=1.3)


def _draw_rois(ax, img, fracs, cmap):
    ax.imshow(img); H, W = img.shape[:2]
    for nm, fr in fracs.items():
        x, y, w, h = fr; c = cmap.get(nm, "#57606a")
        ax.add_patch(Rectangle((x * W, y * H), w * W, h * H, fill=False, edgecolor=c, lw=2.5))
        ax.text(x * W + 2, y * H - 3, nm, color="white", fontsize=10,
                bbox=dict(facecolor=c, edgecolor="none", pad=1.5))
    ax.axis("off")


def _save(fig, out):
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# bioreactor replicate stats + N-replicate figure
# --------------------------------------------------------------------------- #
REP_COLORS = [BLUE, RED, GREEN, PURPLE, "#e16f24"]
ROI_CMAP = {"o1": BLUE, "o2": RED, "o3": GREEN, "o4": GOLD,
            "within-vessel": PURPLE, "housing": "#57606a", "open-field": "#0a7ea4"}


def _replicate_stats(results, fps):
    housing = results["housing"].detrended
    wv = results["within-vessel"]
    f_drive = wv.dominant_hz
    wv_sig = _bandlimit(wv.detrended, fps, f_drive)
    okeys = sorted(k for k in results if len(k) == 2 and k[0] == "o" and k[1].isdigit())
    mag, pw, isi, lags, rates, org_specs = [], [], [], [], [], []
    for k in okeys:
        base = _regress_out(results[k].detrended, housing)[0]
        sig = _bandlimit(base, fps, f_drive)
        F = pulse_features(sig, fps, f_drive)
        if len(F["mag"]):
            mag.append(F["mag"]); pw.append(F["pw"] * 100)
            if len(F["isi"]):
                isi.append(F["isi"] / np.median(F["isi"]))
        rates.append(results[k].bpm_spectral)
        lags.append(xcorr_lag_ms(wv_sig, sig, fps))
        f, p = _spectrum(sig, fps); org_specs.append((f, p / (p.max() or 1)))
    cat = lambda L: np.concatenate(L) if L else np.array([])
    fwv, pwv = _spectrum(wv_sig, fps)
    fo = org_specs[0][0]
    org_mean = np.mean([s[1] for s in org_specs], axis=0)
    rates = np.array(rates)
    locked = np.abs(rates - wv.bpm_spectral) < 0.2 * wv.bpm_spectral
    locked_rate = float(np.mean(rates[locked])) if locked.any() else float("nan")
    return dict(mag=cat(mag), pw=cat(pw), isi=cat(isi), lags=np.array(lags),
                rates=rates, wv_bpm=wv.bpm_spectral, wv_f=fwv, wv_p=pwv / (pwv.max() or 1),
                org_f=fo, org_p=org_mean, n_org=len(okeys),
                n_locked=int(locked.sum()), locked_rate=locked_rate)


def _pooled_meansd(stats, fkey, pkey, grid_bpm):
    stack = []
    for _, R in stats:
        y = np.interp(grid_bpm, R[fkey] * 60, R[pkey], left=0, right=0)
        stack.append(y / (y.max() or 1))
    stack = np.array(stack)
    return stack.mean(0), stack.std(0)


def _boxN(ax, data, labels, colors, title, ylabel):
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    flierprops=dict(marker=".", markersize=3.5, alpha=0.3))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.45); patch.set_edgecolor(c)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=12); ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.15, axis="y")


def replicate_comparison_n(reps, out):
    """reps: list of {label, results, fps, color, roi_fracs}."""
    norm = [(r["label"], r["results"], r["fps"], r.get("color"), r.get("roi_fracs")) for r in reps]
    stats = [(lab, _replicate_stats(res, fps)) for lab, res, fps, _, _ in norm]
    frames = [(color, fracs) for _, _, _, color, fracs in norm]
    cols = [REP_COLORS[i % len(REP_COLORS)] for i in range(len(stats))]
    short = [lab for lab, R in stats]
    n = len(stats)
    has_frames = any(c is not None for c, _ in frames)

    fig = plt.figure(figsize=(14, 13.6 if has_frames else 12))
    if has_frames:
        gs = GridSpec(4, 3, figure=fig, height_ratios=[0.82, 1.0, 1.0, 0.85],
                      hspace=0.30, wspace=0.24, top=0.90, bottom=0.05)
        r0 = 1
        for j, ((lab, R), (color, fracs), c) in enumerate(zip(stats, frames, cols)):
            axi = fig.add_subplot(gs[0, j]); axi.set_anchor("N")
            if color is not None and fracs is not None:
                _draw_rois(axi, color, fracs, ROI_CMAP)
            else:
                axi.axis("off")
            axi.set_title(f"{lab} — ROIs ({R['wv_bpm']:.0f} bpm perfusion)", color=c, fontsize=12)
    else:
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.0],
                      hspace=0.42, wspace=0.32, top=0.9, bottom=0.06)
        r0 = 0

    grid = np.linspace(0, 240, 480)
    wv_bpms = np.array([R["wv_bpm"] for _, R in stats])
    tot_lock = sum(R["n_locked"] for _, R in stats)
    tot_org = sum(R["n_org"] for _, R in stats)

    axv = fig.add_subplot(gs[r0, 0])
    wv_m, wv_s = _pooled_meansd(stats, "wv_f", "wv_p", grid)
    axv.fill_between(grid, wv_m - wv_s, wv_m + wv_s, color=PURPLE, alpha=0.20, lw=0, label="± SD")
    axv.plot(grid, wv_m, color=PURPLE, lw=1.8, label=f"mean of {n} reps")
    axv.set_xlim(0, 240); axv.set_ylim(bottom=0); axv.set_yticks([]); axv.set_xlabel("Rate (bpm)")
    axv.set_title("Within-vessel spectrum (pooled ± SD)")
    axv.text(0.96, 0.72, f"{wv_bpms.mean():.0f} ± {wv_bpms.std():.0f} bpm",
             transform=axv.transAxes, ha="right", color=PURPLE, fontsize=10)
    axv.legend(fontsize=8.5, loc="upper right")

    axo = fig.add_subplot(gs[r0, 1])
    o_m, o_s = _pooled_meansd(stats, "org_f", "org_p", grid)
    OC = "#bf5700"
    axo.fill_between(grid, o_m - o_s, o_m + o_s, color=OC, alpha=0.20, lw=0, label="± SD")
    axo.plot(grid, o_m, color=OC, lw=1.8, label=f"mean of {n} reps")
    axo.set_xlim(0, 240); axo.set_ylim(bottom=0); axo.set_yticks([]); axo.set_xlabel("Rate (bpm)")
    axo.set_title("Organoid spectrum (pooled ± SD)")
    axo.text(0.96, 0.72, f"{tot_lock}/{tot_org} lock", transform=axo.transAxes,
             ha="right", color=OC, fontsize=10)
    axo.legend(fontsize=8.5, loc="upper right")

    axp = fig.add_subplot(gs[r0, 2]); axp.axhline(0, color="k", lw=0.6)
    for i, ((lab, R), c) in enumerate(zip(stats, cols)):
        xs = np.full(len(R["lags"]), i) + np.linspace(-0.13, 0.13, max(len(R["lags"]), 1))
        axp.scatter(xs, R["lags"], color=c, s=42, zorder=3)
        axp.plot([i - 0.22, i + 0.22], [np.mean(R["lags"])] * 2, color=c, lw=2)
    axp.set_xticks(range(n)); axp.set_xticklabels(short)
    axp.set_ylabel("lag vs within-vessel (ms)")
    axp.set_title("Organoid → within-vessel phase lag"); axp.grid(alpha=0.15, axis="y")

    _boxN(fig.add_subplot(gs[r0 + 1, 0]), [R["mag"] for _, R in stats], short, cols,
          "Pulsatility magnitude", "px/frame")
    _boxN(fig.add_subplot(gs[r0 + 1, 1]), [R["pw"] for _, R in stats], short, cols,
          "Pulse FWHM", "% of cycle")
    _boxN(fig.add_subplot(gs[r0 + 1, 2]), [R["isi"] for _, R in stats], short, cols,
          "Inter-spike interval", "× median cycle")

    axt = fig.add_subplot(gs[r0 + 2, :]); axt.axis("off")
    rows = [[lab, f"{R['n_org']}", f"{R['wv_bpm']:.1f}", f"{R['n_locked']}/{R['n_org']} @ {R['locked_rate']:.0f}",
             f"{np.median(R['mag']):.4f}", f"{np.median(R['pw']):.0f}", f"{np.mean(R['lags']):+.0f}"]
            for lab, R in stats]
    tbl = axt.table(cellText=rows,
                    colLabels=["", "n\norganoids", "perfusion\n(bpm)", "organoids\nlocked",
                               "median\nmagnitude", "median pulse\nFWHM (%)", "mean lag\n(ms)"],
                    loc="center", cellLoc="center", bbox=[0.0, 0.28, 1.0, 0.58])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.5)
    for i, c in enumerate(cols):
        tbl[(i + 1, 0)].get_text().set_color(c)
    axt.text(0.5, 0.16, "Organoids vibration-corrected (housing regressed out), measured at the "
             "perfusion rate; box plots pool all organoids per replicate.",
             transform=axt.transAxes, ha="center", fontsize=9.5, style="italic")

    fig.suptitle(f"Bioreactor replicate comparison — {n} replicates", y=0.965, fontsize=16)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# pooled human-vs-bioreactor comparison
# --------------------------------------------------------------------------- #
def _boxpair(ax, data, title, ylabel, labels=("Human\n(resp-corr.)", "Bioreactor")):
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    flierprops=dict(marker=".", markersize=4, alpha=0.35))
    for patch, c in zip(bp["boxes"], [BLUE, RED]):
        patch.set_facecolor(c); patch.set_alpha(0.45); patch.set_edgecolor(c)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_xticklabels(list(labels), fontsize=9.5)
    ax.set_title(title, fontsize=11.5); ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(alpha=0.15, axis="y")


def _cycles(sig, fps, f0, per=48):
    """Peak-to-peak cycles resampled to `per` points each; matrix (n_cycles, per)."""
    dist = max(1, int(round(fps / f0 * 0.6)))
    pk, _ = find_peaks(sig, distance=dist, prominence=0.3 * np.std(sig) or None)
    out = []
    for i in range(len(pk) - 1):
        seg = sig[pk[i]:pk[i + 1]]
        if len(seg) >= 3:
            out.append(np.interp(np.linspace(0, 1, per), np.linspace(0, 1, len(seg)), seg))
    return np.array(out) if out else np.zeros((0, per))


def _tiled(mean_cycle, sd_cycle, npulses):
    """Tile a mean-cycle (+SD) into a continuous-train view of `npulses` pulses."""
    x = np.linspace(0, npulses, len(mean_cycle) * npulses)
    return x, np.tile(mean_cycle, npulses), np.tile(sd_cycle, npulses)


def _pool_bioreactor(bio_reps, per=48):
    bpm_grid = np.linspace(0, 260, 400)
    mag, pw, tw, isi, specs, cyc = [], [], [], [], [], []
    best = None; n_org = 0
    for k, rep in enumerate(bio_reps):
        R, fps = rep["results"], rep["fps"]
        housing = R["housing"].detrended
        fd = R["within-vessel"].dominant_hz
        okeys = sorted(kk for kk in R if len(kk) == 2 and kk[0] == "o" and kk[1].isdigit())
        for ok in okeys:
            res = R[ok]
            base = _regress_out(res.detrended, housing)[0]
            sig = _bandlimit(base, fps, fd)
            F = pulse_features(sig, fps, fd)
            if not len(F["mag"]):
                continue
            n_org += 1
            mag.append(F["mag"]); pw.append(F["pw"] * 100); tw.append(F["tw"] * 100)
            if len(F["isi"]):
                isi.append(F["isi"] / np.median(F["isi"]))
            f, p = _spectrum(sig, fps)
            specs.append(np.interp(bpm_grid, f * 60, p / (p.max() or 1), left=0, right=0))
            C = _cycles(sig, fps, fd, per)
            if len(C):
                cyc.append(C / (np.percentile(np.abs(C), 99) or 1))   # per-unit amplitude-normalized
            snr = getattr(res, "spectral_snr", 0.0)
            if best is None or snr > best[0]:
                best = (snr, sig, res.times[:len(sig)], fps, fd, f"Rep {k+1} {ok}")
    cat = lambda L: np.concatenate(L) if L else np.array([])
    spec_mean = np.mean(specs, axis=0) if specs else np.zeros_like(bpm_grid)
    allc = np.vstack(cyc) if cyc else np.zeros((1, per))
    return dict(mag=cat(mag), pw=cat(pw), tw=cat(tw), isi=cat(isi),
                bpm=bpm_grid, spec=spec_mean,
                cyc_mean=allc.mean(0), cyc_sd=allc.std(0), n_cyc=len(allc),
                rep_bpm=[r["results"]["within-vessel"].dominant_hz * 60 for r in bio_reps],
                n_org=n_org, n_rep=len(bio_reps), rep_of_best=best)


def _pool_human(human_reps, hp_bpm=12.0, per=48):
    from pulsatility import highpass
    bpm_grid = np.linspace(0, 260, 400)
    mag, pw, tw, isi, specs, cyc, rates = [], [], [], [], [], [], []
    best = None; n_h = 0
    for k, d in enumerate(human_reps):
        ctx, resp, fps = d["cortex"], d["resp"], d["fps"]
        sig = resp_correct(_bandlimit(highpass(ctx.detrended, fps, hp_bpm), fps, ctx.dominant_hz), resp)
        F = pulse_features(sig, fps, ctx.dominant_hz)
        if not len(F["mag"]):
            continue
        n_h += 1; rates.append(ctx.dominant_hz * 60)
        mag.append(F["mag"]); pw.append(F["pw"] * 100); tw.append(F["tw"] * 100)
        if len(F["isi"]):
            isi.append(F["isi"] / np.median(F["isi"]))
        f, p = _spectrum(sig, fps)
        specs.append(np.interp(bpm_grid, f * 60, p / (p.max() or 1), left=0, right=0))
        C = _cycles(sig, fps, ctx.dominant_hz, per)
        if len(C):
            cyc.append(C / (np.percentile(np.abs(C), 99) or 1))
        snr = getattr(ctx, "spectral_snr", 0.0)
        if best is None or snr > best[0]:
            best = (snr, sig, ctx.times[:len(sig)], fps, ctx.dominant_hz, d.get("label", f"Human {k+1}"))
    cat = lambda L: np.concatenate(L) if L else np.array([])
    spec_mean = np.mean(specs, axis=0) if specs else np.zeros_like(bpm_grid)
    allc = np.vstack(cyc) if cyc else np.zeros((1, per))
    return dict(mag=cat(mag), pw=cat(pw), tw=cat(tw), isi=cat(isi),
                bpm=bpm_grid, spec=spec_mean,
                cyc_mean=allc.mean(0), cyc_sd=allc.std(0), n_cyc=len(allc),
                rates=rates, n_h=n_h, rep_of_best=best)


def comparison_pooled_figure(human_reps, bio_reps, out, hp_bpm=12.0):
    """Human cortex (both replicates pooled) vs bioreactor (all replicates pooled).

    human_reps: list of {label, cortex, resp, fps}.  Every waveform panel shows the
    ensemble-averaged pulse ± SD (variability across all pooled cycles), tiled into a
    continuous-train view; both sides are pooled."""
    H = _pool_human(human_reps, hp_bpm)
    P = _pool_bioreactor(bio_reps)
    NP = 6                                          # pulses in the tiled-train view
    xh, hm, hs = _tiled(H["cyc_mean"], H["cyc_sd"], NP)
    xb, bm, bs = _tiled(P["cyc_mean"], P["cyc_sd"], NP)
    hbpm = float(np.median(H["rates"])); bbpm = float(np.median(P["rep_bpm"]))
    hlab = f"Human pooled ({H['n_h']} reps, {H['n_cyc']} cycles, {hbpm:.0f} bpm)"
    blab = f"Bioreactor pooled ({P['n_org']} organoids, {P['n_cyc']} cycles, {bbpm:.0f} bpm)"

    fig = plt.figure(figsize=(10, 15.5))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.0, 1.15, 1.2],
                  width_ratios=[2.2, 1.2], hspace=0.5, wspace=0.28, top=0.9, bottom=0.05)

    def _wave(ax, x, mean, sd, color, ylabel, title, xlabel=None):
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.20, lw=0)
        ax.plot(x, mean, color=color, lw=1.9)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_xlim(0, NP); ax.set_xticks(range(NP + 1))
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11.5)
        ax.set_xlabel(xlabel) if xlabel else ax.set_xticklabels([])

    def _spec(ax, pool, color, bpm, title, xlabel=False):
        ax.plot(pool["bpm"], pool["spec"] / (pool["spec"].max() or 1), color=color, lw=1.4)
        ax.axvline(bpm, color="k", ls="--", lw=0.8, alpha=0.55)
        ax.text(0.95, 0.82, f"{bpm:.0f} bpm", transform=ax.transAxes, ha="right", color=color)
        ax.set_xlim(0, 260); ax.set_yticks([]); ax.set_title(title, fontsize=10)
        ax.set_xlabel("Rate (bpm)") if xlabel else ax.set_xticklabels([])

    # rows 0/1: each condition's ensemble pulse ± SD (left) + pooled spectrum (right)
    _wave(fig.add_subplot(gs[0, 0]), xh, hm, hs, BLUE,
          "Human cortex\n(norm. motion)", "Ensemble-averaged pulse ± SD (rate-standardized)")
    _spec(fig.add_subplot(gs[0, 1]), H, BLUE, hbpm, f"Human pooled spectrum\n({H['n_h']} reps)")
    _wave(fig.add_subplot(gs[1, 0]), xb, bm, bs, RED,
          "Bioreactor organoid\n(norm. motion)", "",
          xlabel="Pulse number (rate-standardized)")
    _spec(fig.add_subplot(gs[1, 1]), P, RED, bbpm, f"Bioreactor pooled spectrum\n({P['n_org']} organoids)", xlabel=True)

    # row 2: overlay both ensemble pulses ± SD
    axtr = fig.add_subplot(gs[2, :])
    axtr.fill_between(xh, hm - hs, hm + hs, color=BLUE, alpha=0.16, lw=0)
    axtr.plot(xh, hm, color=BLUE, lw=1.9, label=hlab)
    axtr.fill_between(xb, bm - bs, bm + bs, color=RED, alpha=0.16, lw=0)
    axtr.plot(xb, bm, color=RED, lw=1.9, label=blab)
    axtr.axhline(0, color="k", lw=0.4, alpha=0.4)
    axtr.set_xlim(0, NP); axtr.set_xticks(range(NP + 1))
    axtr.set_xlabel("Pulse number (rate-standardized)"); axtr.set_ylabel("Normalized motion")
    axtr.set_title("Ensemble pulse ± SD — human vs bioreactor (both pooled)")
    _headroom_legend(axtr, ncol=1, frac=0.34)

    # row 3: pooled box plots
    lbls = (f"Human\n({H['n_h']} reps)", f"Bioreactor\n({P['n_rep']} reps)")
    box = gs[3, :].subgridspec(1, 4, wspace=0.55)
    _boxpair(fig.add_subplot(box[0]), [H["mag"], P["mag"]], "Pulsatility magnitude", "px/frame", lbls)
    _boxpair(fig.add_subplot(box[1]), [H["pw"], P["pw"]], "Pulse FWHM", "% of cycle", lbls)
    _boxpair(fig.add_subplot(box[2]), [H["tw"], P["tw"]], "Trough FWHM", "% of cycle", lbls)
    _boxpair(fig.add_subplot(box[3]), [H["isi"], P["isi"]], "Inter-spike interval", "× median cycle", lbls)

    fig.suptitle(f"Pulsatility: human cortex ({H['n_h']} replicates pooled) vs bioreactor "
                 f"({P['n_rep']} replicates pooled, {P['n_org']} organoids)", y=0.955, fontsize=13)
    _save(fig, out)


# --------------------------------------------------------------------------- #
# human replicate comparison (rep1 vs rep2), slow-drift high-passed
# --------------------------------------------------------------------------- #
def _human_sig(d, hp_bpm=12.0):
    from pulsatility import highpass
    ctx, resp, fps = d["cortex"], d["resp"], d["fps"]
    base = highpass(ctx.detrended, fps, hp_bpm)
    return resp_correct(_bandlimit(base, fps, ctx.dominant_hz), resp), ctx, resp, fps


def human_replicate_figure(reps, out):
    """reps: list of {label, cortex, resp, fps, color, roi_frac}."""
    prep = []
    for d in reps:
        sig, ctx, resp, fps = _human_sig(d)
        F = pulse_features(sig, fps, ctx.dominant_hz)
        prep.append(dict(lab=d["label"], sig=sig, ctx=ctx, resp=resp, fps=fps,
                         color=d.get("color"), roi=d.get("roi_frac"), F=F))
    cols = [BLUE, RED, GREEN, PURPLE][:len(prep)]
    n = len(prep)
    NP = 6                                          # pulses in the tiled ensemble-train view

    # rows: [ROI + spectrum] , one row per replicate trace, [box plots]
    fig = plt.figure(figsize=(13, 4.2 + 2.0 * n + 3.2))
    gs = GridSpec(2 + n, 3, figure=fig,
                  height_ratios=[0.9] + [0.6] * n + [1.05],
                  hspace=0.45, wspace=0.28, top=0.93, bottom=0.06)

    # row 0: ROI frames (cols 0..) + overlaid cardiac spectrum in the last col
    for j, (P, c) in enumerate(zip(prep, cols[:2])):
        axi = fig.add_subplot(gs[0, j]); axi.set_anchor("N")
        if P["color"] is not None and P["roi"] is not None:
            _draw_rois(axi, P["color"], {"cortex": P["roi"]}, {"cortex": c})
        else:
            axi.axis("off")
        axi.set_title(f"{P['lab']} — cortex ROI ({P['ctx'].dominant_hz*60:.0f} bpm)", color=c, fontsize=12)
    axs = fig.add_subplot(gs[0, 2])
    for P, c in zip(prep, cols):
        f, p = _spectrum(P["sig"], P["fps"]); m = (f * 60 <= 200)
        axs.plot(f[m] * 60, p[m] / (p[m].max() or 1), color=c, lw=1.4,
                 label=f"{P['lab']}: {P['ctx'].dominant_hz*60:.0f} bpm")
    axs.set_xlim(0, 200); axs.set_yticks([]); axs.set_xlabel("Rate (bpm)")
    axs.set_title("Cardiac rate spectrum", fontsize=12); axs.legend(fontsize=8.5)

    # rows 1..n: each replicate's ensemble-averaged pulse +/- SD in its OWN plot
    # (px/frame, so amplitude differences between patients stay visible; rate-matched
    # by showing the average cycle tiled into a continuous train)
    for P in prep:
        C = _cycles(P["sig"], P["fps"], P["ctx"].dominant_hz)
        P["cm"], P["cs"] = (C.mean(0), C.std(0)) if len(C) else (np.zeros(48), np.zeros(48))
    ymax = max(float(np.max(np.abs(P["cm"]) + P["cs"])) or 1 for P in prep) * 1.1
    for i, (P, c) in enumerate(zip(prep, cols)):
        ax = fig.add_subplot(gs[1 + i, :])
        xt, mm, ss = _tiled(P["cm"], P["cs"], NP)
        ax.fill_between(xt, mm - ss, mm + ss, color=c, alpha=0.20, lw=0)
        ax.plot(xt, mm, color=c, lw=1.6)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_xlim(0, NP); ax.set_ylim(-ymax, ymax); ax.set_xticks(range(NP + 1))
        ax.set_ylabel(f"{P['lab']}\n(px/frame)", fontsize=10.5)
        ax.set_title(f"{P['lab']} — ensemble-averaged pulse ± SD "
                     f"(rate-standardized; true rate {P['ctx'].dominant_hz*60:.0f} bpm)", fontsize=11)
        if i == n - 1:
            ax.set_xlabel("Pulse number (rate-standardized)")
        else:
            ax.set_xticklabels([])

    # last row: box plots pulse FWHM, trough FWHM, ISI
    labs = [P["lab"] for P in prep]
    def boxN(ax, data, title, ylabel):
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        flierprops=dict(marker=".", markersize=3.5, alpha=0.3))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.45); patch.set_edgecolor(c)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_xticklabels(labs, fontsize=10); ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11); ax.grid(alpha=0.15, axis="y")
    boxN(fig.add_subplot(gs[1 + n, 0]), [P["F"]["pw"] * 100 for P in prep], "Pulse FWHM", "% of cycle")
    boxN(fig.add_subplot(gs[1 + n, 1]), [P["F"]["tw"] * 100 for P in prep], "Trough FWHM", "% of cycle")
    boxN(fig.add_subplot(gs[1 + n, 2]),
         [(P["F"]["isi"] / np.median(P["F"]["isi"]) if len(P["F"]["isi"]) else P["F"]["isi"]) for P in prep],
         "Inter-spike interval", "× median cycle")

    fig.suptitle("Parenchymal pulsatility in human — replicate comparison "
                 "(slow drift removed, rate-matched)", y=0.965, fontsize=15.5)
    _save(fig, out)
