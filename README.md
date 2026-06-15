# microscopy-pipeline

A unified, composable Python package for the OOI / Braingeneers microscopy
image-processing pipeline. What used to be a set of stand-alone scripts run one
at a time from the shell is now a single importable package with a consistent
internal data model, every operation exposed as a function, and ready-made
**task-to-task workflows**.

## Layout

```
microscopy_pipeline/
  io.py            # all image/stack load+save, in one place (numpy in, numpy out)
  cli.py           # shared -i/-o argument parser used by every op
  ops/             # one module per operation (21 ops)
  workflows.py     # named pipelines that chain ops task-to-task
preprocessing/ processing/ postprocessing/   # thin CLI shims -> ops.*.cli
```

## Canonical data representation

Every operation speaks one of two in-memory types (see `io.py`):

| Artifact | In-memory type |
|----------|----------------|
| Single image | `np.ndarray` — 2-D `(H,W)` grayscale (uint8/uint16) or `(H,W,3)` RGB (uint8, RGB order) |
| Stack / multi-frame TIFF | `List[np.ndarray]` — a list of 2-D frames (**not** a 3-D array) |

`io.load_image`/`save_image` handle single images; `io.load_stack`/`save_stack`
handle stacks. A 3-D array is used only as a private vectorization buffer inside
a function and never crosses a public boundary. GIFs, CSVs and matplotlib
figures are deliberately outside this layer (the relevant ops own that I/O).

## Three forms of every op

Each op in `microscopy_pipeline.ops` is available as:

* `op(array, **params) -> array` — pure numpy core, composable in memory;
* `op_file(in_path, out_path, **params)` — file in / file out;
* `op_folder(in_dir, out_dir, **params)` — batch over a folder;
* `op.cli()` — the command line used by the legacy shim scripts.

```python
from microscopy_pipeline import io, ops

stack = io.load_stack("raw.tif")              # List[np.ndarray]
fused = ops.complex_edf_image(stack)          # 2-D uint16
fused = ops.clahe(fused, clip_limit=2.0)
rgb   = ops.grey_to_color(fused, channel="green")
io.save_image(rgb, "out.png")
```

## Workflows

`microscopy_pipeline.workflows` chains ops into whole pipelines:

| Function | Pipeline |
|----------|----------|
| `focal_stack_to_gif` | focal stacks → EDF/mean fuse → brightness → CLAHE → colorize → scale bar → animated GIF |
| `pngs_to_fused_images` | `{X}_{Y}.png` slices → per-X TIFF stacks → fused stills |
| `align_crop_colorize` | register (ECC) → crop → brightness-correct → colorize |
| `brightfield_to_growth_graphs` | brightfield series → organoid tracking → area/darkness growth graphs |
| `fluorescence_to_curve` | fluorescence series → summed intensity CSV → smoothed trend SVG |

```python
from microscopy_pipeline import workflows

workflows.focal_stack_to_gif("aligned_pngs/", "movie.gif",
                             method="edf", channel="green",
                             scale_um=100, pixel_size_um=0.5)
```

## Data flow & filename conventions

Stages are glued together by filename patterns, so the conventions matter:

```
raw  {tp}_zs+{z}.png                      acquisition: timepoint tp, z-slice z
  → align            {tp}_zs+{z}.png → {tp}_{z}.png          (ECC registration)
  → pngs_to_tiff     {X}_{Y}.png     → stack_{X}.tiff        (one stack per X)
  → complex_edf      stack_{X}.tiff  → stack_{X}_edf.tif     (depth fusion; or fuse_tiffs)
  → tiff_to_gif / sum_fluorescence / brightfield_organoid_tracker → GIF, CSVs, plots
```

`extract_times` reads the raw `{N}_zs*` files into a timestamps CSV that the
fluorescence summation can merge in.

## OME-TIFF

Read/write OME-TIFF with the physical pixel size embedded in OME-XML, for
hand-off to Fiji/Bio-Formats, napari or QuPath. Needs the optional `tifffile`
backend (`pip install -e ".[ome]"`):

```python
from microscopy_pipeline import io
io.save_ome_tiff(frames, "stack.ome.tif", pixel_size_um=0.5)
io.read_ome_metadata("stack.ome.tif")    # {'pixel_size_um': 0.5, 'axes': 'ZYX', ...}
```

Or from the CLI: `mp-to-ome-tiff -i in/ -o out/ --pixel-size-um 0.5`. Any op that
writes to a `*.ome.tif` path emits OME-TIFF automatically, and the
`--pixel-size`-aware ops (scale bar, organoid tracker) read the calibration
straight from OME files.

## Further reference

See **[REPO_MAP.md](REPO_MAP.md)** for the full agent-facing capability map —
every op annotated, the detailed data-flow conventions, and the branch points
where alternative approaches can be trialed. **[PLAN.md](PLAN.md)** tracks the
in-progress remediation work.

## Install

```bash
pip install -e .            # editable install, with all op CLIs as mp-* commands
pip install -e ".[test]"    # plus pytest
```

After install the op CLIs are available as `mp-crop`, `mp-clahe`,
`mp-complex-edf`, `mp-brightfield-organoid-tracker`, … (one per op).

## Test

```bash
pytest -q
```

The suite builds tiny synthetic inputs (no external data) and drives every
workflow end-to-end, plus guards the canonical representation and method
exposure. Runs in a few seconds.
