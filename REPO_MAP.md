# Microscopy Pipeline — Repo Map & Capabilities

Agent-facing reference for the OOI organoid time-lapse microscopy pipeline. It
maps every component, annotates what it does, marks **branch points** (places to
trial alternative approaches), and lists **features to add**.

> Rename or symlink this to `CLAUDE.md` if you want it auto-loaded into the agent
> context each session.

---

## 1. Architecture at a glance

The repo has **one real implementation** and a set of **legacy shims**.

```
microscopy_pipeline/         # ← the package: single source of truth
  __init__.py                #   re-exports every op + workflows into the top namespace
  io.py                      #   all image/stack/folder I/O (PIL + numpy)
  cli.py                     #   shared argparse helpers (-i/-o convention)
  ops/                       #   20 operation modules (one capability each)
  workflows.py               #   5 task-to-task pipelines that chain ops end-to-end

preprocessing/ processing/ postprocessing/   # ← thin CLI shims only
pyproject.toml               # ← installable package (mp-<op> console scripts)
README.md                    # ← user-facing quickstart
tests/                       # ← pytest smoke suite (synthetic fixtures)
```

Every legacy script (e.g. `preprocessing/crop.py`, `processing/complexedf.py`)
is now a ~15-line shim that puts the repo root on `sys.path` and calls
`microscopy_pipeline.ops.<module>.cli()`. **Do not add logic to the shims** —
all behavior lives in `microscopy_pipeline/ops/`.

### The three-layer op contract
Every op module exposes the same shape (see `microscopy_pipeline/__init__.py`):

| Layer | Form | Purpose |
|-------|------|---------|
| Core | `op(array, **params) -> array` | Pure numpy, composable in-memory (no disk) |
| File | `op_file(in_path, out_path, **params)` | One file → one file; handles TIFF stacks |
| Folder | `op_folder(in_dir, out_dir, **params)` | Batch over a directory |
| CLI | `cli(argv=None)` | argparse wrapper used by the shim |

Because the core layer is pure, ops chain in Python without touching disk:

```python
from microscopy_pipeline import io, ops
stack = io.load_stack("raw.tif")                 # List[np.ndarray]
fused = ops.complex_edf_image(stack, levels=3)   # single 2-D array (see §2)
fused = ops.clahe(fused, clip_limit=2.0)
rgb   = ops.grey_to_color(fused, channel="green")
rgb   = ops.add_scale_bar(rgb, scale_um=100, pixel_size_um=0.5)
io.save_image(rgb, "out.png")
```

Whole multi-stage pipelines come pre-assembled in `workflows.py` (see §4).

### CLI convention
Every CLI takes `-i/--input` and `-o/--output` (file or folder, auto-detected).
Op-specific params are `--kebab-case` mirroring the keyword arg
(`--clip-limit` → `clip_limit`).

---

## 2. Shared infrastructure

### `io.py` — format & batch handling
- **Canonical in-memory types**: a **single image** is an `np.ndarray` (2-D
  grayscale uint8/uint16, or `(H,W,3)` RGB uint8); a **stack** is a
  `List[np.ndarray]` — a list of 2-D frames, **not** a 3-D `(frames,H,W)` array.
  (A 3-D array is only ever a private vectorization buffer inside one function.)
  The module header used to mis-state this as 3-D; corrected.
- **Bit-depth aware**: preserves 8-bit / 16-bit through PIL modes (`L`, `I;16`,
  `I`). `detect_bit_depth`, `_pil_to_array`, `array_to_pil`.
- **Single image**: `load_image` / `save_image`.
- **Stacks**: `load_stack` (→ list) / `save_stack` (accepts any iterable of
  frames). `save_stack` writes ImageJ-compatible multi-frame TIFFs (tag 50839)
  and auto-disables LZW for 16-bit (PIL is fragile there).
- **Folders**: `list_images` (sorted, non-recursive), `ensure_dir`,
  `batch_apply`.
- Recognized extensions: png/jpg/jpeg/bmp/tif/tiff/gif/webp/jp2.
- GIFs, CSVs and matplotlib figures are intentionally outside this layer (the
  relevant ops own that I/O).

### `cli.py` — argparse helpers
- `build_io_parser(description, input_required, output_required, epilog)`.
- `dispatch_file_or_folder(...)` — a file/folder router (fixed a shadowing bug;
  still unused — every op does its own `if src.is_dir()` dispatch instead).

---

## 3. Capability map

Grouped by pipeline stage. "Core fn" is the importable pure function.

### 3a. Preprocessing (geometry, pixels, alignment, packaging)

| Op (module) | Core fn | What it does | Key params / flags |
|---|---|---|---|
| **crop** | `crop`, `crop_stack` | Trim L/T/R/B pixels off an image (`crop`) or every frame of a stack (`crop_stack`) | `--left/--top/--right/--bottom` |
| **mask** | `mask` / `parse_color` | Replace masked pixels (black-in-mask = replace) with a color; optional Gaussian feathering for alpha-blended edges; 8/16-bit + RGB | `--mask`, `--replace-color` (name/#hex/R,G,B/scalar), `--feather-radius` |
| **remove_lone_pixels** | `find_lone_pixels`, `remove_lone_pixels`, `remove_lone_pixels_batch` | Detect persistent single-pixel hot spots (brighter than all 8 neighbors by `threshold` in ≥`min_fraction` of a group) and replace with 8-neighbor mean. Groups files by numeric prefix before first `_` | `--threshold`, `--min-fraction`, `--start-prefix` (resume) |
| **scale_colors** | `scale_colors` | Linearly remap an 8-bit band `[bottom, top]` → full 16-bit `[0, 65535]` (contrast stretch into uint16) | `--top`, `--bottom` (required) |
| **align** (ECC) | `align_pair`, `align_stack`, `align_session` | Register a z-stack/timepoint acquisition. `align_session` aligns each timepoint's base-z to a reference and/or the previous timepoint (temporal), then aligns other z-slices to the base (`{tp}_zs+{z}.png` → `{tp}_{z}.png`). `align_stack` is the in-memory counterpart: register a `List[np.ndarray]` to a reference (default frame 0) with no naming rules | `--min/max-timepoint`, `--min/max-z`, `--z-increment`, `--base-z`, `--reference-image`, `--no-temporal` |
| **pngs_to_tiff** | `pngs_to_tiff_stacks`, `group_pngs_by_x` | Bundle `{X}_{Y}.png` into one ImageJ TIFF `stack_{X}.tiff` per X, frames sorted by Y | `--bit-depth {8,16}` |
| **extract_times** | `extract_times`, `parse_time_offset` | Read file creation times of `{N}_zs*` images → `file_timestamps.csv` (elapsed seconds/HH:MM:SS, actual_time) | `--time-offset` (s / MM:SS / HH:MM:SS, signed) |

### 3b. Processing (depth fusion)

| Op (module) | Core fn | What it does | Key params / flags |
|---|---|---|---|
| **complex_edf** | `complex_edf`, `complex_edf_image` | Extended-Depth-of-Field fusion via complex wavelets: pick the per-pixel sharpest (max-magnitude) wavelet coefficients across the stack and reconstruct. `complex_edf` returns `(fused, min_z, max_z)`; `complex_edf_image` is the chain-friendly wrapper returning just the fused array cast to its target dtype. `invert=True` selects *least* sharp (legacy `invertedcomplexedf`) | `--wavelet`, `--levels`, `--bit-depth`, `--top-n` (avg N sharpest), `--invert` |
| **fuse_tiffs** | `fuse_average` | Mean-projection of all frames in a stack into one image; `--brightfield` stretches result to full range. Folder mode processes `stack_X.tiff` in numeric order | `--brightfield` |

### 3c. Postprocessing (contrast, color, annotation, export)

| Op (module) | Core fn | What it does | Key params / flags |
|---|---|---|---|
| **clahe** | `clahe` | Contrast-Limited Adaptive Histogram Equalization (grayscale, or one RGB channel) | `--clip-limit`, `--tile-grid` (N→NxN), `--channel {red,green,blue}` |
| **grey_to_color** | `grey_to_color` | Colorize grayscale by routing the gray value into one RGB channel; optional gamma. Outputs 8-bit RGB | `--channel` (required), `--gamma` |
| **scale_brightness** | `scale_brightness` | Multiply pixel values by a scalar, clip to dtype range | `--factor` (required) |
| **find_max_brightness** | `find_max_brightness` | Report max pixel value after a small Gaussian blur + abs-max ratio (for choosing `scale_brightness` factors) | `--blur-kernel` (output not required) |
| **scale_bar** | `add_scale_bar` | Draw a filled scale-bar rectangle + `"N um"` label in the lower-left | `--scale-um`, `--pixel-size-um` (required), `--bar-height`, `--margin`, `--color`, `--no-label` |
| **tiff_to_gif** | `tiff_to_gif`, `frames_to_gif` | Build an animated GIF from a TIFF stack or a folder of TIFFs (sorted by a regex capture group). Per-frame normalizes to 8-bit | `--pattern`, `--duration-ms`, `--loop` |

### 3d. Analysis (quantification & plots)

| Op (module) | Core fn | What it does | Output |
|---|---|---|---|
| **sum_fluorescence** | `sum_brightness`, `sum_fluorescence_folder` | Sum pixel intensity per `*_N_*.tif` image (`--brightfield` inverts → optical density). Optional merge of `extract_times` timestamps | CSV: `index, filename, sum_brightness`/`sum_optical_density`, `actual_time` |
| **plot_summed_fluorescence** | `plot_fluorescence` | Smoothed time-series plot of the sum CSV: moving average + t/normal confidence band, auto time-unit (min/hr/days), %-of-initial normalization | SVG |
| **identify_core** | `identify_core` | Segment a bright **core** + surrounding **outer ring** in an organoid image via percentile thresholds + heavy morphology + convexity repair (convex-hull fallback). Reports per-region area & summed intensity | per-image `*_contours.png`, `outer_regions/`, `core_analysis_results.csv` |
| **brightfield_organoid_tracker** | `find_organoid_contour`, `track_organoid`, `track_organoid_folder` | Detect the largest dark-on-bright organoid (CLAHE→blur→threshold→morphology→largest connected component), measure area/perimeter/circularity/darkness over time; timepoint parsed from filename. Folder mode accepts any image extension (via `io.list_images`), not only `.tif` | overlay PNGs, `organoid_analysis_results.csv`, `organoid_areas_summary.csv` |
| **graph_contour_data** | `graph_contour_data` | Plot the tracker CSV: area-over-time, darkness-over-time, combined (growth rate, area↔darkness correlation) + text summary | PNGs + `organoid_analysis_summary.txt` |

---

## 4. Workflows — task-to-task pipelines (`workflows.py`)

`microscopy_pipeline.workflows` pre-assembles the ops into whole pipelines so a
session runs with one call — in-memory where the ops allow, folder-in / files-out
for the disk-bound analysis ops. Import as
`from microscopy_pipeline import workflows`.

| Workflow | Chain | In → out |
|---|---|---|
| `focal_stack_to_gif` | fuse (`complex_edf_image`/`fuse_average`) → `scale_brightness` → `clahe` → `grey_to_color` → `add_scale_bar` → `frames_to_gif` | focal stacks (or `{X}_{Y}.png` folder) → animated GIF |
| `pngs_to_fused_images` | `pngs_to_tiff_stacks` → `fuse_average_file`/`complex_edf_file` | `{X}_{Y}.png` folder → per-X fused TIFF stills |
| `align_crop_colorize` | `align_stack` → `crop` → (`find_max_brightness` →) `scale_brightness` → `grey_to_color` | frame list → list of RGB stills (optional folder write) |
| `brightfield_to_growth_graphs` | `track_organoid_folder` → `graph_contour_data` | brightfield folder → growth-curve PNGs + CSVs |
| `fluorescence_to_curve` | (`extract_times` →) `sum_fluorescence_folder` → `plot_fluorescence` | fluorescence folder → sum CSV + trend SVG |

```python
from microscopy_pipeline import workflows
workflows.focal_stack_to_gif("aligned_pngs/", "movie.gif",
                             method="edf", channel="green",
                             scale_um=100, pixel_size_um=0.5)
```

Each workflow has an end-to-end smoke test in `tests/` (synthetic fixtures, no
external data). These are the **5 reference compositions**; build new ones the
same way from the §3 cores.

---

## 5. Data flow & filename conventions

The pipeline is glued by filename patterns, not by a runner. Knowing them is how
you connect stages:

```
raw  {tp}_zs+{z}.png
  ├─ extract_times ........ {N}_zs*  → file_timestamps.csv        (timestamps, runs on raws)
  └─ align ................ {tp}_zs+{z}.png → {tp}_{z}.png        (ECC registration)
        │   (optional per-frame: crop · scale_colors · clahe · mask · remove_lone_pixels · scale_brightness)
        ▼
     pngs_to_tiff ......... {X}_{Y}.png → stack_{X}.tiff          (X=timepoint, Y=z)
        ▼
     complex_edf .......... stack_{X}.tiff → stack_{X}_edf.tif    (depth fusion)
        │   (alt: fuse_tiffs → stack_{X}_superimposed.tiff)
        │   (optional post: clahe · grey_to_color · scale_bar · scale_brightness)
        ├─ tiff_to_gif ..... stack_{N}_edf.tif → animation.gif
        ├─ sum_fluorescence  *_N_*.tif → sum CSV → plot_summed_fluorescence → SVG
        └─ brightfield_organoid_tracker → CSV → graph_contour_data → PNGs
            (identify_core runs standalone on fused organoid images)
```

**Pattern gotchas** (the seams most likely to break a chain):
- `align` consumes `{tp}_zs+{z}.png`, emits `{tp}_{z}.png`.
- `pngs_to_tiff` only matches `(\d+)_(-?\d+)\.png`.
- `complex_edf` folder mode only reads `.tif/.tiff`; emits `_edf.tif`.
- `tiff_to_gif` default regex is `stack_(\d+)_edf\.tif` — pass `--pattern` otherwise.
- `sum_fluorescence` requires the index flanked by underscores: `.*_(\d+)_.*\.tiff?`.
- `remove_lone_pixels` groups by the numeric prefix **before the first `_`**.

---

## 6. Branch points — where to trial alternative approaches

Each row is a decision baked into the current code that you could swap to
experiment. These are the highest-leverage places for method comparisons.

| Where | Current approach | Alternatives worth trialing |
|---|---|---|
| **align** `warp_mode` | `MOTION_TRANSLATION` ECC | `EUCLIDEAN`/`AFFINE`/`HOMOGRAPHY`; phase correlation; feature-based (ORB/SIFT + RANSAC); optical flow; pystackreg |
| **align** strategy | temporal (prev-frame) chaining | align-all-to-fixed-reference; multi-resolution pyramid; drift correction across whole session |
| **complex_edf** focus rule | max `|complex coeff|` per pixel | variance / Tenengrad / modified-Laplacian focus measures; guided filtering; max-vs-weighted blending; per-region consistency check |
| **complex_edf** wavelet | `db3`, 3 levels | other wavelet families (`sym`, `coif`, `bior`); dual-tree complex wavelet (DTCWT); level sweep |
| **fuse_tiffs** projection | mean | max / median / std / percentile projection; weighted by focus measure |
| **clahe** | CLAHE per channel/gray | global hist-eq; gamma; retinex / homomorphic; unsharp mask |
| **remove_lone_pixels** replacement | 8-neighbor mean | median filter; temporal median across the group; inpainting; hot-pixel map subtraction |
| **identify_core** segmentation | percentile thresholds + morphology + convex-hull repair | Otsu/multi-Otsu; active contours / level sets; watershed; ML (Cellpose, StarDist) |
| **brightfield_organoid_tracker** threshold | Otsu with `otsu_adjustment` bias | adaptive (already a flag) / manual; Sauvola; edge-based (Canny+close); ML segmentation; per-frame tracking with linking (trackpy) |
| **grey_to_color** | route gray → one RGB channel | perceptual colormaps (viridis/magma) via LUT; multi-channel composites for co-stains |
| **scale_colors / scale_brightness** | linear remap / scalar × | gamma; percentile auto-contrast; histogram matching across timepoints |
| **plot_summed_fluorescence** smoothing | centered moving average + t/normal CI | LOESS; Savitzky-Golay; bootstrap CI; spline fit |
| **tiff_to_gif** normalization | **per-frame** min/max | **global** normalization across frames (per-frame causes brightness flicker in time-lapse) |
| **sum_fluorescence** region | whole-image sum | mask to organoid (reuse `identify_core`/tracker mask); background subtraction; per-region (core vs outer) |

---

## 7. Features to add

Ranked roughly by leverage for this codebase.

**Orchestration & ergonomics**
1. **Pipeline runner** — ✅ *partly done*: `workflows.py` chains ops end-to-end
   in Python (5 pipelines, §4). Still TODO: a config-driven (YAML/JSON) executor
   and an `mp run <workflow>` CLI so non-Python users can drive them.
2. **Packaging** — ✅ *done*: `pyproject.toml` installs the package and exposes
   each op as a `mp-<op>` console script. A single `mp <op>` dispatcher is still
   a nice-to-have.
3. **README + examples** — ✅ *done*: `README.md` adds a quickstart, the data
   model, and the workflow table. (This file remains the deep agent reference.)
4. **Logging & progress** — replace bare `print` with `logging` + `tqdm`
   progress bars for long folder jobs.

**Correctness & robustness**
5. **Tests + sample data** — ✅ *done*: `tests/` has end-to-end smoke tests for
   every workflow plus representation/exposure guards, on synthetic fixtures.
   Still worth adding: golden-image/round-trip tests for individual ops.
6. **Resumability** — only `remove_lone_pixels` has `--start-prefix`; add
   resume/skip-existing to all folder ops.
7. **Metadata-driven pixel size** — read pixel size / spacing from TIFF/OME tags
   instead of requiring `--pixel-size-um` by hand; preserve metadata on save.
8. **Fix `np.choose` 32-frame ceiling in `complex_edf`** — `np.choose` caps the
   choice axis at 32, so stacks with >32 z-slices will fail; switch to
   `np.take_along_axis`.
9. **`io.batch_apply` `keep_name=False`** is a no-op (both branches identical) —
   either implement renaming or drop the parameter.

**Capability**
10. **Parallelism** — `multiprocessing`/`concurrent.futures` for folder ops
    (alignment and EDF are the slow ones).
11. **Unified organoid analysis** — combine `identify_core` + `sum_fluorescence`
    + tracker into one pass that emits core/outer fluorescence *and* area over
    time (mask-aware summation).
12. **ML segmentation backend** — optional Cellpose/StarDist for core/organoid
    detection as an alternative to the morphology heuristics.
13. **GPU / large-image support** — tiled processing and optional CUDA (cv2 /
    cupy) paths for big sessions.
14. **OME-TIFF / Bio-Formats I/O** and **napari/Fiji** hand-off for inspection.
15. **GIF/MP4 export options** — global normalization toggle, MP4/H.264 output,
    timestamp overlay on frames.

---

## 8. Quick op index (module → import path)

`crop · mask · remove_lone_pixels · scale_colors · align · pngs_to_tiff ·
extract_times · complex_edf · fuse_tiffs · clahe · grey_to_color ·
scale_brightness · find_max_brightness · scale_bar · tiff_to_gif ·
sum_fluorescence · plot_summed_fluorescence · identify_core ·
brightfield_organoid_tracker · graph_contour_data`

All live in `microscopy_pipeline/ops/<module>.py` and are re-exported from the
top package namespace (`from microscopy_pipeline import ops`).
