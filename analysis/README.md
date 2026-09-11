# Pulsatility analysis scripts

Reproduce the human-cortex and bioreactor pulsatility figures from the study
videos. These scripts drive the committed [`pulsatility`](../pulsatility) package;
`figures.py` holds the presentation layer (spectra, phase, box plots, ROI insets,
rate-matched / rate-standardized pulse trains).

**The source videos are private (patient / lab data) and are not included.** Point
the scripts at your own copies.

## Layout

| file | what it does |
|------|--------------|
| `figures.py` | figure module: `replicate_comparison_n`, `comparison_pooled_figure`, `human_replicate_figure`, `biological_comparison_figure`, `gold_standard_figure`/`gold_standard_core` + signal helpers (spectrum, band-limit, pulse features, rate-match, phase lag) |
| `run_bioreactor.py` | analyse one bioreactor replicate video → `bio_rep<N>_full.pkl` |
| `run_human.py` | analyse one clean (4K) human cortex video (pulsatility + respiration) → `<label>_full.pkl` |
| `run_human_surgical.py` | recover cardiac pulse from a hands-in-frame **surgical** human clip by rejecting hand/instrument transients and keeping contiguous clean runs → `<key>_surgical.pkl` |
| `run_signed_projection.py` | signed-projection (PCA) motion → `signed_<key>.pkl` for the displacement/pressure space |
| `build_comparisons.py` | build the three comparison figures from the pickles |
| `build_gold_standard.py` | human-as-gold-standard figures (speed + signed/displacement); optionally folds a surgical human in as a third, weight-controlled real human |
| `pool_surgical_human.py` | single-run vs pooled ensemble pulse for a surgical human (why pooling short clean runs sharpens the mean) → `<key>_pooled.png` |
| `analyze_slow_oscillation.py` | ROI origin test: is a slow oscillation physiological or global field drift? |

## Paths

All scripts root their I/O at `$PULS_WORK` (default `./pulsatility_work`):
videos are read from wherever you pass them; pickles and figures are written to
`$PULS_WORK/out/`.

## Example

```bash
export PULS_WORK=/path/to/work
pip install -e ".[video]"                      # from repo root

python analysis/run_bioreactor.py 1 /data/Bioreactor.MOV
python analysis/run_bioreactor.py 2 /data/Bioreactor_rep2.MOV
python analysis/run_bioreactor.py 3 /data/Bioreactor_rep3.MOV
python analysis/run_human.py human1 /data/Human.MOV   0.40,0.30,0.24,0.34
python analysis/run_human.py human2 /data/Human_2.MOV 0.436,0.400,0.247,0.440
python analysis/build_comparisons.py

# human-as-gold-standard, optionally with a surgical (hands-in-frame) human folded in
python analysis/run_human_surgical.py human4 /data/Human_surgical.MOV   # -> human4_surgical.pkl
python analysis/pool_surgical_human.py human4                            # pooling demo figure
python analysis/build_gold_standard.py                                   # surgical weight via PULS_SURG_WEIGHT (default 1.0 = equal)

python analysis/analyze_slow_oscillation.py /data/Human_2.MOV
```

## Notes

* ROI boxes are `x,y,w,h` as fractions of frame width/height; the defaults in each
  script are the boxes used for this study's videos — re-place them for new views.
* Cortex pulsatility is high-passed (via `pulsatility.highpass`) to strip slow
  global field drift, and organoids are vibration-corrected by regressing out the
  housing ROI (`pulsatility.regress_out_reference`) before pulse metrics.
