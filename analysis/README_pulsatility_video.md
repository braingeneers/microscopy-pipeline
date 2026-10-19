# Pulsatility side-by-side comparison video

Tools for turning two perfused-parenchyma clips (e.g. a human cortical exposure
and a hydrogel bioreactor) into a **rate-matched side-by-side video** where both
beat at a common tempo, each with its displacement (~pressure) waveform drawn
underneath. These sit alongside the `pulsatility/` package (which quantifies a
single clip); here the emphasis is a legible visual A/B.

## Scripts

| script | what it does |
|--------|--------------|
| `fetch_box_folder.py` | list/download clips from a Box shared-folder link; optional ffmpeg segment extraction |
| `score_pulsatility_clips.py` | rank clips by spectral SNR of their pulsatility, to pick the best of each kind |
| `pulsatility_sidebyside.py` | the renderer: stabilize → optical-flow signed velocity → displacement → cycle rate-match → optional crop / Eulerian magnification → side-by-side mp4 (+ GIF) |

## Dependencies

numpy, opencv-python(-headless), scipy, plus an **ffmpeg** binary. ffmpeg is
located from `$FFMPEG`, then `PATH`, then the `imageio-ffmpeg` wheel
(`pip install imageio-ffmpeg`) if present — so no system install is required.

## End-to-end example

```bash
# 1. fetch clips (downscaled 600px segments, originals dropped to save disk)
python analysis/fetch_box_folder.py https://<org>.box.com/s/XXXX \
    --out clips --segment 8 25 600 --drop-original

# 2. pick the cleanest clip of each kind
python analysis/score_pulsatility_clips.py clips/*bio*_seg600.mp4
python analysis/score_pulsatility_clips.py clips/*brain*_seg600.mp4

# 3. build the comparison (brain stabilized; gel shown unstabilized + a light
#    motion boost so its sub-pixel beat is visible; minimal labels; GIF too)
python analysis/pulsatility_sidebyside.py \
    --left  clips/brain_best.mp4 --left-label  "human brain" \
    --right clips/gel_best.mp4   --right-label "hydrogel bioreactor" \
    --no-right-stabilize --right-crop 0.06,0.02,0.66,0.82 --right-magnify 2 \
    --minimal --out sidebyside.mp4 --gif sidebyside.gif
```

## Method notes / honest caveats

- **Rate-matching.** Each clip's dominant rate is found from the signed-velocity
  spectrum; cycles are segmented at diastolic troughs, the K cleanest consecutive
  cycles are chosen, and each is resampled to a fixed number of output frames.
  That single step locks both tempo *and* phase, so the panels beat together.
- **Stabilize which side?** ECC removes whole-field translation/rotation. That is
  right when a rigid frame surrounds locally-deforming tissue (cortex). For a soft
  chamber whose visible beat is *bulk* motion, stabilization would null exactly
  what you want to see — use `--no-right-stabilize`. (Measured: on the gel clip
  stabilization removed only ~0.2 px of a 108 bpm global shift; the gel's real
  structural displacement is ~1–3 px at native 4K, i.e. sub-pixel once downscaled,
  which is why aggressive downsampling hides it.)
- **Magnification is not real-scale.** `--<side>-magnify N` is linear Eulerian
  amplification in a narrow band around the measured rate; it exaggerates a real
  but small motion for visibility. Label it, and prefer the smallest N that reads.
- **Amplitudes are normalized per panel.** The gel typically moves ~30× less than
  cortex, so the waveforms compare *shape and timing*, not magnitude.
- The real clips are the user's private footage and are **not** committed; the
  scripts regenerate everything from the source videos.
