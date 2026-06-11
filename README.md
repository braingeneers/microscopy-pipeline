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
