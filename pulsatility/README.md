# Pulsatility quantification

A self-contained tool that turns a brightfield / surgical-microscopy **video** of
perfused parenchyma into a **bulk pulsatility waveform** — a continuous wave
showing how the tissue moves with each pulse of the perfusing flow — plus the
pulse rate, a pulsatility index, and a spatial map of where the tissue pulses
most.

Built for a gel block perfused from above by pulsatile flow through embedded
vessels, and designed to carry over unchanged to a forthcoming neurosurgical
video of brain parenchyma.

## The idea

Each pulse of the perfusing flow briefly pressurises the embedded vessels and
nudges the surrounding parenchyma. In the video this appears as tiny, spatially
coherent **motion** of the tissue. We measure that motion frame-to-frame with
dense optical flow and take the mean flow *magnitude* per frame as a
"tissue-speed" signal — it rises and falls once per pulse, which is the
continuous wave we want.

Working from **motion** rather than raw brightness is deliberate:

* it is robust to the **oblique filming angle** (we measure in-plane tissue
  displacement, whatever the viewing geometry — a caveat, not a blocker);
* it is robust to slow **illumination drift / bleaching**, which swamp a naive
  mean-intensity signal (in the sample video the intensity signal is
  drift-dominated while the motion signal is a clean 2 Hz pulse train);
* nothing about it is gel-specific — it finds whatever coherent pulsation is
  present, so the same call works on brain parenchyma.

## Install

The tool ships as part of this repo and uses only existing dependencies
(`opencv-python`, `numpy`, `scipy`, `matplotlib`):

```bash
pip install -e .
```

## Use

```bash
mp-pulsatility -i Pulsatility_video.mp4 -o results/
```

or from Python:

```python
from pulsatility import analyze_pulsatility_video

result = analyze_pulsatility_video("Pulsatility_video.mp4", "results/")
print(result.summary())
print(result.bpm_spectral, "pulses/min")
```

### Outputs (written to the `-o` directory)

| File | Contents |
|------|----------|
| `pulsatility_analysis.png` | Headline figure: the continuous pulsatility wave, the pulse spectrum, the spatial amplitude map, and the metrics |
| `pulsatility_waveform.csv` | Per-frame signal: `time_s`, `tissue_motion_px_per_frame`, `detrended`, `pulsatility_wave`, `is_pulse_peak` |
| `pulsatility_amplitude_map.png` | Standalone "where it pulses" heatmap over a reference frame |
| `pulsatility_summary.txt` | The text metrics block |

### Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--method {flow,diff}` | `flow` | `flow` = dense optical-flow tissue speed (robust); `diff` = mean absolute frame difference (faster, but conflates brightness change with motion) |
| `--resize-width` | `320` | Downscale width before analysis. Pulsatility is a bulk, low-spatial-frequency phenomenon, so downscaling denoises and speeds up flow without losing signal. `0` keeps full resolution |
| `--min-bpm` / `--max-bpm` | `30` / `300` | Bounds of the pulse-rate search (pulses per minute). Cardiac brain rates (~40–120 bpm) sit comfortably inside the default |
| `--fps` | from file | Override the frame rate if the container metadata is wrong |
| `--frame-stride` | `1` | Analyze every Nth frame (fps scaled accordingly) |
| `--pixel-size-um` | — | If given, wave amplitudes are additionally reported in µm/s |

## Metrics

* **pulse rate (spectral)** — fundamental of the motion spectrum, in pulses/min.
* **pulse rate (peaks)** — rate from detected pulse-to-pulse intervals; agreement
  with the spectral rate is a self-consistency check.
* **rate variability (CV)** — coefficient of variation of the pulse intervals.
* **pulsatility index** — modulation depth of the tissue-motion signal,
  `(P95 − P5) / mean`; analogous in spirit to a Gosling pulsatility index.
* **wave amplitude (RMS / p2p)** — size of the pulsation in px/frame (or µm/s with
  `--pixel-size-um`).
* **spectral SNR** — fraction of in-band power concentrated at the fundamental; a
  cleanliness / confidence measure.

## Result on the sample gel video

`results/` holds the output for the provided `Pulsatility_video.mp4`
(1032×934, 30 fps, 18 s):

```
pulse rate (spectral) :  118.3 pulses/min (1.972 Hz)
pulse rate (peaks)    :  118.3 pulses/min over 35 pulses
rate variability (CV) :    7.3 %
pulsatility index     :  1.934
spectral SNR          :   78.7 % of band power at the fundamental
```

The spectrum is a single sharp peak at ~2 Hz and the spatial amplitude map
localises the pulsation to the perfused vessel regions — exactly the
perivascular signal we will want to track in brain parenchyma.

![sample analysis](results/pulsatility_analysis.png)

## Method notes & where to extend

* **Motion measure** — Farneback dense optical flow (`--method flow`). The
  spatial mean of the flow magnitude is the bulk-tissue-speed waveform;
  per-pixel temporal variance gives the amplitude map. `--method diff` is a
  cheaper fallback.
* **Detrending** — a moving-average baseline (window ≈ 1.5 × the slowest
  expected period) removes drift while preserving the pulse *shape*.
* **Frequency / wave** — Hann-windowed, zero-padded FFT locates the fundamental;
  a 2nd-order zero-phase Butterworth band-pass around it yields the clean wave;
  `scipy.signal.find_peaks` counts pulses.
* **Generalising to brain** — the algorithm is tissue-agnostic. For surgical
  video the main additions worth trialing are camera-shake stabilisation before
  flow (the surgeon/scope moves), an ROI mask to the exposed cortex, and
  narrowing `--min-bpm/--max-bpm` to the cardiac band to reject respiratory and
  handling motion.
