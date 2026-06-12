# microscopy_pipeline — Remediation Plan

This document is a self-contained remediation plan for the `microscopy_pipeline` package. It is written so a future session with no memory of the planning conversation can pick it up cold, re-confirm each item against the current code, and execute it. Items are grouped into phases (correctness → docs → robustness → configurable branch points → optional/deferred), each with verified locations, the problem, the fix, touch points, verification, and risk.

> **Dated note (2026-06-11):** The repository was being edited live during planning. Line numbers and snippets below are best-effort snapshots captured at planning time. **Re-confirm EACH item against the current code before editing** — open the listed files, verify the symbol/line still matches, and adjust line numbers as needed. Treat every location/line-number as a hint, not a guarantee.

---

## How to use this document

Checklist for resuming cold:

- [ ] **Install** the package in editable mode: `pip install -e ".[test]"` (or `pip install -e .` without the test extra).
- [ ] **Run the suite** to establish a green baseline: `python -m pytest -q` (expected: `22 passed`).
- [ ] **Work on a branch** — do not edit on `main`. Create a feature branch per item or per phase.
- [ ] **Pick an item** from the phases below, in phase order (Phase 1 → Phase 5).
- [ ] **Re-confirm** the item's `Where` locations against current code before editing (line numbers are best-effort snapshots).
- [ ] **Re-run the suite** after each item (`python -m pytest -q`) and confirm it stays green before moving on.

---

## Current state (already done — not in scope)

**Tests command:** `python -m pytest -q`
**Tests result:** `22 passed in 4.92s`
**Passed / Failed:** 22 passed, 0 failed

Done items:

- ✅ **Packaging:** `pyproject.toml` (setuptools build backend) declares package microscopy-pipeline v0.1.0 with runtime deps (numpy, Pillow, opencv-python, PyWavelets, pandas, matplotlib, scipy, scikit-image) and a `[test]` extra (pytest); packages = microscopy_pipeline + microscopy_pipeline.ops.
- ✅ **Console scripts:** 20 mp-* entry points wired in `[project.scripts]`, one per ops module CLI (mp-align, mp-clahe, mp-complex-edf, mp-crop, mp-brightfield-organoid-tracker, mp-extract-times, mp-find-max-brightness, mp-fuse-tiffs, mp-graph-contour-data, mp-grey-to-color, mp-identify-core, mp-mask, mp-plot-summed-fluorescence, mp-pngs-to-tiff, mp-remove-lone-pixels, mp-scale-bar, mp-scale-brightness, mp-scale-colors, mp-sum-fluorescence, mp-tiff-to-gif).
- ✅ **Orchestration layer:** `microscopy_pipeline/workflows.py` chains ops into 5 named task-to-task pipelines (`workflows.__all__ = focal_stack_to_gif, pngs_to_fused_images, align_crop_colorize, brightfield_to_growth_graphs, fluorescence_to_curve`), with in-memory array-to-array stages plus disk-bridge stages for the analysis ops.
- ✅ **Convenience cores:** ops exposes `align_stack` (ECC stack registration, used by align_crop_colorize) and `complex_edf_image` (in-memory complex-wavelet extended-depth-of-field fuse, used by focal_stack_to_gif); both confirmed importable from `microscopy_pipeline.ops`.
- ✅ **Passing test suite:** `tests/` (conftest.py + test_workflows.py + test_harmonization.py) drives all 5 workflows end-to-end on synthetic inputs and guards the canonical representation; `python -m pytest -q` => 22 passed, 0 failed in ~4.9s.

### Package layout

- `microscopy_pipeline/` top-level modules: io.py, cli.py, workflows.py, __init__.py
- `microscopy_pipeline/ops`: 20 operation modules (excluding __init__.py): align.py, brightfield_organoid_tracker.py, clahe.py, complex_edf.py, crop.py, extract_times.py, find_max_brightness.py, fuse_tiffs.py, graph_contour_data.py, grey_to_color.py, identify_core.py, mask.py, plot_summed_fluorescence.py, pngs_to_tiff.py, remove_lone_pixels.py, scale_bar.py, scale_brightness.py, scale_colors.py, sum_fluorescence.py, tiff_to_gif.py (plus __init__.py = 21 .py files total in ops/)
- `workflows.__all__` (5 names): focal_stack_to_gif, pngs_to_fused_images, align_crop_colorize, brightfield_to_growth_graphs, fluorescence_to_curve
- `tests/` files (3): conftest.py, test_workflows.py, test_harmonization.py (located at repo-root tests/, not under microscopy_pipeline/; testpaths=['tests'])
- `pyproject.toml [project.scripts]`: 20 entries (all mp-*)

> **Snapshot notes:** Snapshot is consistent and green: 22/22 tests pass. Contrary to an earlier task hint, README.md is **not** missing — a substantive 3465-byte README.md exists at the repo root (referenced by `pyproject readme = "README.md"`) documenting layout, canonical data model, the three op forms (array core / _file / _folder / cli), the 5 workflows, install (`pip install -e ".[test]"`) and test instructions. Other notes: (1) tests/ lives at the repo root rather than inside the package, with `testpaths=['tests']` and a conftest.py for shared fixtures. (2) The ops/ count is 20 operation modules; including __init__.py there are 21 .py files under ops/ (the README's "21 ops" phrasing counts the package files, not 21 distinct ops). (3) requires-python is `>=3.8` but workflows.py uses `Path.unlink(missing_ok=True)` (3.8+ OK) and `from __future__ import annotations`, so it stays 3.8-compatible. (4) The legacy preprocessing/ processing/ postprocessing/ script trees still exist (modified in git status) as thin CLI shims into ops.*.cli; the new package is additive and untracked. (5) Workflows mix pure in-memory pipelines (focal_stack_to_gif, align_crop_colorize) with disk-oriented analysis stages (brightfield_to_growth_graphs, fluorescence_to_curve, pngs_to_fused_images), as designed.

---

## Scope

**IN scope:** every item listed below (Phases 1–5).

**OUT of scope:** the EDF focus-**SELECTION** rule — i.e. experimenting with the max-magnitude selection rule and replacing/augmenting it with variance / Laplacian / DTCWT-based focus measures. That rule is deliberately left untouched.

> **Clarification:** The item `phase1-complex-edf-npchoose-32-ceiling` (`complex-edf-choose`) fixes the **`np.choose` mechanism** — a correctness bug (the 32-array-object ceiling). It does **NOT** touch the focus-selection rule. The argpartition-based max-magnitude selection is intentionally preserved; only the per-pixel gather mechanism changes.

---

## Phases

## Phase 1 - Correctness

### Replace np.choose with np.take_along_axis to lift 32-frame ceiling in complex-wavelet EDF

**Status:** confirmed_present | **Category:** correctness | **Effort:** S

**Where:**
- `microscopy_pipeline/ops/complex_edf.py:72`
- `microscopy_pipeline/ops/complex_edf.py:73`
- `microscopy_pipeline/ops/complex_edf.py:85`

**Problem:**
complex_edf() selects per-pixel top-N focus coefficients inside the per-level loop (lines 58-86) using np.choose. There are exactly three np.choose calls, all still present and all in this coefficient-selection loop: line 72 `fc += np.choose(top_indices[i], cstack)`, line 73 `fd += np.choose(top_indices[i], dstack)` (the complex-detail branch), and line 85 `fa += np.choose(top_indices[i], cstack)` (the approximation branch). In each call cstack/dstack are np.stack(..., axis=0) arrays of shape (N, H, W) where N = number of input z-slices (len(images)), and top_indices[i] is an (H, W) integer index array selecting along axis 0. np.choose treats the choice axis as separate array objects and is capped at NPY_MAXARGS=32; for N>32 it raises `ValueError: Need at least 0 and at most 32 array objects.` (empirically reproduced with numpy 1.24.4). Therefore any focal stack with more than 32 frames crashes complex_edf, and by extension the public wrappers complex_edf_image (lines 105-119, calls complex_edf directly), complex_edf_file (lines 122-131), and complex_edf_folder (lines 134-145). This is purely a mechanism limitation of np.choose; the max-magnitude selection rule (argpartition on mags at lines 64 and 80) is unaffected and out of scope.

**Fix:**
Mechanism-only swap, leaving the argpartition-based focus-selection rule untouched. Replace each np.choose(<idx>, <stack>) with np.take_along_axis(<stack>, <idx>[None], axis=0)[0], which gathers along axis 0 with no choice-count limit and returns the same (H, W) result. Concretely: line 72 -> `fc += np.take_along_axis(cstack, top_indices[i][None], axis=0)[0]`; line 73 -> `fd += np.take_along_axis(dstack, top_indices[i][None], axis=0)[0]`; line 85 -> `fa += np.take_along_axis(cstack, top_indices[i][None], axis=0)[0]`. The [None] inserts a leading length-1 axis so the index array shape (1, H, W) broadcasts against the (N, H, W) stack on axis 0, and [0] drops it back to (H, W). Verified identical to np.choose for N<=32 and dtype-preserving for both float64 (cstack approximation / cD detail) and complex128 (cstack complex-detail). No signature, no other line, and no selection logic changes.

**Touch points:**
- `microscopy_pipeline/ops/complex_edf.py:72 (complex-detail branch, cstack)`
- `microscopy_pipeline/ops/complex_edf.py:73 (complex-detail branch, dstack)`
- `microscopy_pipeline/ops/complex_edf.py:85 (approximation branch, cstack)`

**Verify:**
Synthetic >32-frame stack regression. (1) Build frames = [np.random.randint(0, 65535, (64, 64), dtype=np.uint16) for _ in range(40)] (40 > 32). (2) Pre-fix: call microscopy_pipeline.ops.complex_edf.complex_edf(frames) and confirm it raises ValueError 'Need at least 0 and at most 32 array objects.' Post-fix: confirm it returns a 3-tuple (fused, min_z, max_z) with fused.shape == frames[0].shape, fused.dtype == np.float64, and 0 <= min_z <= max_z < 40, no exception. (3) complex_edf_image(frames, bit_depth=16) returns an np.uint16 array of shape (64, 64); with bit_depth=8 returns np.uint8. (4) Also exercise top_n=3 and invert=True on the 40-frame stack to cover both the n-loop (range(n)) and the invert sign path through the same take_along_axis lines. (5) Equivalence guard: on a small N=5 stack, assert results are bitwise-identical before vs after the change (np.array_equal), confirming behaviour is unchanged for <=32 frames. The harness note: complex_edf_folder swallows per-file exceptions (line 144 `except Exception`), so test complex_edf/complex_edf_image directly rather than via the folder path to avoid masking a regression.

**Risk:**
Very low. np.take_along_axis is a drop-in elementwise gather producing identical values to np.choose for N<=32 (empirically confirmed), preserves float64 and complex128 dtypes, and the surrounding accumulation (fc/fd/fa += ...) and averaging (/n) are unchanged. Available since numpy 1.15, well below any plausible floor (env has 1.24.4). Only behavioural change is that stacks with >32 slices now succeed instead of raising. The focus-selection rule (argpartition max-magnitude) is deliberately not touched, so fused output for <=32-frame stacks is bit-identical.

---

### Add normalize={global,per-frame,none} (default global) to frames_to_gif to stop time-lapse brightness flicker

**Status:** confirmed_present | **Category:** correctness | **Effort:** S

**Where:**
- `microscopy_pipeline/ops/tiff_to_gif.py:18-24 (_normalize_to_8bit, per-array min/max)`
- `microscopy_pipeline/ops/tiff_to_gif.py:38-58 (frames_to_gif signature + body)`
- `microscopy_pipeline/ops/tiff_to_gif.py:49 (per-frame normalize call inside list comprehension)`
- `microscopy_pipeline/ops/tiff_to_gif.py:61-68 (tiff_to_gif signature)`
- `microscopy_pipeline/ops/tiff_to_gif.py:78 (tiff_to_gif -> frames_to_gif call)`
- `microscopy_pipeline/ops/tiff_to_gif.py:81-89 (cli: add_argument flags + tiff_to_gif call)`
- `microscopy_pipeline/workflows.py:139 (focal_stack_to_gif -> ops.frames_to_gif call)`
- `microscopy_pipeline/workflows.py:78-99 (focal_stack_to_gif signature)`

**Problem:**
frames_to_gif normalizes every frame independently. At tiff_to_gif.py:49 the list comprehension calls _normalize_to_8bit(np.asarray(f)) once per frame, and _normalize_to_8bit (lines 18-24) rescales using that single frame's own min/max ((a - a.min())/(a.max() - a.min())*255). Across a time-lapse each frame's intensity is stretched to fill 0-255 using its own extremes, so two frames with identical underlying content but different absolute intensities (or a stack where one frame is brighter) get mapped to different gray levels. The result is visible brightness flicker frame-to-frame. There is no way to use a single normalization window across the whole sequence: no `normalize` parameter exists on frames_to_gif (lines 38-45), tiff_to_gif (lines 61-68), the CLI (lines 81-89), or workflows.focal_stack_to_gif (lines 78-99); the workflows call at line 139 forwards only duration_ms and loop.

**Fix:**
Add a `normalize: str = "global"` keyword-only parameter to frames_to_gif (tiff_to_gif.py:38-45) accepting {"global","per-frame","none"}. Refactor _normalize_to_8bit so the scaling window can be supplied: e.g. keep per-frame behavior but add a helper that computes one (vmin, vmax) across ALL non-uint8 frames and rescales every frame with that shared window. In frames_to_gif body (replace line 49): if normalize=="global" compute vmin=min over all frames, vmax=max over all frames once, then map each frame with the shared window (frames already uint8 pass through unchanged, matching current dtype guard); if normalize=="per-frame" keep current per-array _normalize_to_8bit; if normalize=="none" skip rescaling (just cast/asarray to uint8). Validate the value and raise ValueError on unknown. Thread the parameter through every caller: (1) frames_to_gif signature at 38-45; (2) tiff_to_gif signature at 61-68 add `normalize: str = "global"` and forward it in the frames_to_gif call at line 78; (3) CLI at 81-89 add `parser.add_argument("--normalize", choices=["global","per-frame","none"], default="global", ...)` and pass `normalize=args.normalize` in the tiff_to_gif call at 88-89; (4) workflows.focal_stack_to_gif signature at 78-99 add `normalize: str = "global"` and forward `normalize=normalize` in the ops.frames_to_gif call at line 139. Keep "global" as the default everywhere so the flicker fix is on by default.

**Touch points:**
- `microscopy_pipeline/ops/tiff_to_gif.py:18-24 (_normalize_to_8bit / add shared-window helper)`
- `microscopy_pipeline/ops/tiff_to_gif.py:38-45 (frames_to_gif signature: add normalize)`
- `microscopy_pipeline/ops/tiff_to_gif.py:47-49 (body: branch on normalize)`
- `microscopy_pipeline/ops/tiff_to_gif.py:61-68 (tiff_to_gif signature: add normalize)`
- `microscopy_pipeline/ops/tiff_to_gif.py:78 (forward normalize)`
- `microscopy_pipeline/ops/tiff_to_gif.py:83-89 (CLI flag + forward)`
- `microscopy_pipeline/workflows.py:78-99 (focal_stack_to_gif signature: add normalize)`
- `microscopy_pipeline/workflows.py:139 (forward normalize to ops.frames_to_gif)`

**Verify:**
Add a unit test (e.g. tests/test_workflows.py near the existing GIF tests at lines 13-25, or a new tests/test_tiff_to_gif.py): build several np.ndarray frames with identical content (same values, non-uint8 dtype so normalization triggers) plus one extra frame scaled brighter. Call ops.frames_to_gif(frames, out, normalize="global"), reload the GIF with PIL (Image.open + seek over frames, convert to np.array), and assert the frames sharing identical content map to equal gray (np.array_equal of their pixel arrays). Contrast: assert that with normalize="per-frame" the identical-content frames would NOT be guaranteed equal once a brighter frame is present in the sequence is not directly observable per-frame, so instead also assert normalize="none" passes a uint8 frame through unchanged. Run: pytest tests/ -k gif. Existing tests test_focal_stack_to_gif_edf_in_memory and test_focal_stack_to_gif_average_from_pngs must still pass (default flips to global).

**Risk:**
Low. Default changes effective normalization from per-frame to global, which alters output pixel values of existing GIFs (intended behavior change). uint8 frames are unaffected (dtype guard preserved). Global min==max edge case must be guarded like the current a.max()>a.min() check to avoid divide-by-zero. "none" path must still yield uint8 for Image.fromarray. The two existing GIF tests assert only file existence/size, so they should remain green.

---

### Remove unused io.batch_apply and cli.dispatch_file_or_folder (delete accompanying test for the latter)

**Status:** partial | **Category:** correctness | **Effort:** S

**Where:**
- `microscopy_pipeline/io.py:238-266`
- `microscopy_pipeline/cli.py:48-62`
- `tests/test_harmonization.py:95-113`
- `REPO_MAP.md:81`
- `REPO_MAP.md:88`
- `REPO_MAP.md:250`

**Problem:**
Both helpers are dead production code. io.batch_apply (io.py:242-266) has ZERO callers anywhere in the repo (grep over ops/, workflows/, tests/, and the legacy preprocessing/processing/postprocessing shims finds only its own definition). Neither symbol is re-exported in microscopy_pipeline/__init__.py. batch_apply also contains a confirmed no-op: line 259 reads `dst = output_dir / src.name if keep_name else output_dir / src.name`, where both ternary branches are byte-for-byte identical, so the keep_name flag has no effect at all. cli.dispatch_file_or_folder (cli.py:48-62) likewise has NO production callers (no op module routes through it), so it is dead in the package. HOWEVER the item's premise that it is fully unused is only partial: tests/test_harmonization.py:97-113 (test_dispatch_file_or_folder) imports and exercises it directly (the test's own comment at line 95 even labels it 'was buggy/unused'). Deleting the function without also removing that test will break pytest, so the two deletions are not symmetric.

**Fix:**
Delete io.batch_apply: remove io.py lines 238-266 (the section-header comment block 'Generic batch helper used by op CLIs.' at 238-240 plus the full function body 242-266). Nothing imports Tuple solely for this signature elsewhere, but verify the `Tuple` import on io.py:26 is unused after removal and drop it from the typing import if so. Delete cli.dispatch_file_or_folder: remove cli.py lines 48-62 (the function and its blank-line separator). Because that function's only reference is the test, ALSO delete tests/test_harmonization.py lines 95-113 (the '--- Shared CLI dispatch helper' comment banner plus test_dispatch_file_or_folder) so pytest stays green; otherwise the test's `from microscopy_pipeline.cli import dispatch_file_or_folder` import fails at collection. After removal, check whether the `Callable` import on cli.py:17 (typing) is still used; it is not used elsewhere in cli.py, so drop `Callable` from that import line too. Optionally update REPO_MAP.md references (lines 81, 88, 250) that document these helpers.

**Touch points:**
- `microscopy_pipeline/io.py`
- `microscopy_pipeline/cli.py`
- `tests/test_harmonization.py`

**Verify:**
After edits: `grep -rn 'batch_apply\|dispatch_file_or_folder' .` returns zero matches under microscopy_pipeline/ and tests/ (only REPO_MAP.md prose if left untouched). Then run `python -m pytest tests/` and confirm it stays green (test_dispatch_file_or_folder must be removed, not left dangling, or collection will ImportError). Also confirm `python -c "import microscopy_pipeline.io, microscopy_pipeline.cli"` imports cleanly (catches a stale unused Tuple/Callable import if you trim them).

**Risk:**
Low. Both functions are dead production code with no importers and no re-export. The only live reference is the dedicated unit test for dispatch_file_or_folder, which must be deleted in the same change. Minor care needed when trimming the now-unused `Tuple` (io.py:26) and `Callable` (cli.py:17) typing imports so as not to remove names still used elsewhere in those modules.

---

## Phase 2 - Docs

### Write README.md referenced by pyproject readme="README.md"

**Status:** already_done | **Category:** docs | **Effort:** S

**Where:**
- `d:\Documents\Braingeneers\OOI scripts\microscopy-pipeline\README.md`
- `d:\Documents\Braingeneers\OOI scripts\microscopy-pipeline\pyproject.toml:9`
- `d:\Documents\Braingeneers\OOI scripts\microscopy-pipeline\microscopy_pipeline\workflows.py:36-42`
- `d:\Documents\Braingeneers\OOI scripts\microscopy-pipeline\pyproject.toml:28-50`

**Problem:**
The item's premise is false against the CURRENT tree: README.md DOES exist at repo root (d:\...\README.md, 3465 bytes, written this session at 15:18) and pyproject.toml line 9 sets readme = "README.md", so the readme reference already resolves and `pip install -e .` will NOT warn or fail for a missing readme. The existing README already covers every section the item requested: a "what it is" intro (lines 1-7), Install with `pip install -e .` and `pip install -e ".[test]"` (lines 73-78), the mp-* console commands description (lines 80-81), a Workflows table documenting all 5 workflows one line each (lines 54-71) -- focal_stack_to_gif, pngs_to_fused_images, align_crop_colorize, brightfield_to_growth_graphs, fluorescence_to_curve, which exactly match microscopy_pipeline/workflows.py __all__ (lines 36-42) -- plus Layout, canonical data representation, three-forms-of-every-op, and Test sections. The 20 mp-* console scripts in pyproject [project.scripts] (lines 31-50) match the documented mp-crop/mp-clahe/mp-complex-edf/etc. The ONE requested element that is absent: the README contains no pointer to REPO_MAP.md (Grep for 'REPO_MAP' in README.md returns no matches), even though REPO_MAP.md exists at repo root (18633 bytes) and is the intended deep agent reference. Also the README's filename-convention section is lighter than the item's requested 'raw {tp}_zs+{z}.png -> align -> pngs_to_tiff -> EDF/fuse -> analysis' flow (that detailed flow lives in REPO_MAP.md section 5, not in README).

**Fix:**
No new README is needed -- it already exists and satisfies the core of the item. The only net-new work is a small enhancement: add a one-line pointer to REPO_MAP.md (e.g. a "See REPO_MAP.md for the full agent-facing capability map and branch points" line, naturally under the Layout or Workflows section), and optionally add a short filename/data-conventions blurb summarizing the raw {tp}_zs+{z}.png -> align -> pngs_to_tiff -> EDF/fuse -> analysis flow (or just cross-reference REPO_MAP.md section 5 which already documents it). If the plan only required the file to exist and resolve for packaging, mark complete as-is; the REPO_MAP.md pointer is the single concrete addition called out by the item that is currently missing.

**Touch points:**
- `d:\Documents\Braingeneers\OOI scripts\microscopy-pipeline\README.md`

**Verify:**
`pip install -e .` already succeeds with respect to the readme reference because README.md exists and pyproject.toml:9 points to it (no missing-file warning). To verify the one gap: Grep 'REPO_MAP' over README.md returns no matches (confirmed). After adding the pointer, re-Grep should find it. Optionally run `pip install -e .` (deps: numpy, Pillow, opencv-python, PyWavelets, pandas, matplotlib, scipy, scikit-image) and confirm no readme-related warning and that mp-* console scripts are created.

**Risk:**
Very low. README is pure documentation; editing it cannot break code, imports, tests, or packaging. The only risk is doc drift if the workflow list or mp-* command set later changes -- currently they are consistent with workflows.py __all__ and pyproject [project.scripts].

---

## Phase 3 - Robustness

### Replace print() with module logging + add -v/--verbose and optional tqdm progress

**Status:** confirmed_present | **Category:** robustness | **Effort:** M

**Where:**
- `microscopy_pipeline/io.py:264`
- `microscopy_pipeline/ops/align.py:93`
- `microscopy_pipeline/ops/complex_edf.py:130`
- `microscopy_pipeline/ops/complex_edf.py:145`
- `microscopy_pipeline/ops/extract_times.py:98`
- `microscopy_pipeline/ops/find_max_brightness.py:43`
- `microscopy_pipeline/ops/fuse_tiffs.py:65`
- `microscopy_pipeline/ops/identify_core.py:181`
- `microscopy_pipeline/ops/identify_core.py:227`
- `microscopy_pipeline/ops/pngs_to_tiff.py:73`
- `microscopy_pipeline/ops/sum_fluorescence.py:100`
- `microscopy_pipeline/ops/plot_summed_fluorescence.py:164`
- `microscopy_pipeline/cli.py:20`

**Problem:**
The package emits all user-facing diagnostics via raw print(), confirmed at 12 call sites across 9 files (grep 'print(' in microscopy_pipeline): io.py:264 (batch_apply failure), ops/align.py:93 (missing base image), ops/complex_edf.py:130 (z range) and :145 (failure), ops/extract_times.py:98 (rows written), ops/find_max_brightness.py:43 (max/ratio), ops/fuse_tiffs.py:65 (fused name), ops/identify_core.py:181 (no regions) and :227 (processed count), ops/pngs_to_tiff.py:73 (wrote slices), ops/sum_fluorescence.py:100 (rows written), ops/plot_summed_fluorescence.py:164 (plot saved). There is no module logger and no way to control verbosity: a fresh grep for 'import logging|getLogger|logging\.|tqdm|logger\.' over microscopy_pipeline returns zero matches. cli.build_io_parser (cli.py:20-45) adds only -i/--input and -o/--output, so there is no -v/--verbose knob. Long folder loops (align_session, complex_edf_folder, the io.batch_apply-backed *_folder ops, sum_fluorescence_folder, track_organoid_folder, identify_core_folder) print intermittently or run silently with no progress indication; error prints at io.py:264, complex_edf.py:145 and align.py:93 swallow exceptions into stdout, hiding stack context. This is a robustness/observability gap, not a crash bug.

**Fix:**
Introduce a single module logger logging.getLogger("microscopy_pipeline") (define once, e.g. as LOGGER in io.py or __init__.py, and import where needed). Replace each of the 12 print() calls with the appropriate level: informational summaries (complex_edf.py:130, extract_times.py:98, find_max_brightness.py:43, fuse_tiffs.py:65, identify_core.py:227, pngs_to_tiff.py:73, sum_fluorescence.py:100, plot_summed_fluorescence.py:164) -> logger.info; diagnostic/failure paths (io.py:264, align.py:93, complex_edf.py:145, identify_core.py:181) -> logger.warning (or logger.exception inside the except blocks at io.py:264 / complex_edf.py:145 to retain traceback). Add a -v/--verbose flag (action='count', default=0) to build_io_parser in cli.py:20-45 and a small helper (e.g. configure_logging(verbosity) in cli.py) that maps count -> level (0=WARNING, 1=INFO, 2=DEBUG) via logging.basicConfig and sets the microscopy_pipeline logger level; call this helper at the top of each op's cli() right after parser.parse_args (each op cli already follows the build_io_parser -> parse_args -> *_file/*_folder dispatch pattern, e.g. complex_edf.py:158-165). For progress, wrap the explicit file loops in the named folder ops (align.py align_session loop at :86, complex_edf_folder loop at :137, io.batch_apply loop at :258 which backs clahe/crop/grey_to_color/mask/scale_* folder ops, sum_fluorescence_folder loop at :73, identify_core_folder loop at :195, track_organoid_folder loop at :177) with tqdm using a guarded optional import (try: from tqdm import tqdm except ImportError: fallback to identity wrapper) so the new dependency stays optional; optionally also add tqdm to the project's optional-dependencies in pyproject.toml. Keep return values and CSV/image side effects unchanged.

**Touch points:**
- `microscopy_pipeline/cli.py`
- `microscopy_pipeline/io.py`
- `microscopy_pipeline/ops/align.py`
- `microscopy_pipeline/ops/complex_edf.py`
- `microscopy_pipeline/ops/extract_times.py`
- `microscopy_pipeline/ops/find_max_brightness.py`
- `microscopy_pipeline/ops/fuse_tiffs.py`
- `microscopy_pipeline/ops/identify_core.py`
- `microscopy_pipeline/ops/pngs_to_tiff.py`
- `microscopy_pipeline/ops/sum_fluorescence.py`
- `microscopy_pipeline/ops/plot_summed_fluorescence.py`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py`
- `pyproject.toml`

**Verify:**
1) Run pytest from repo root (`python -m pytest -q`) and confirm it stays green (current baseline: 22 passed in ~5s). 2) Confirm no print( remain: grep 'print(' over microscopy_pipeline should return zero hits (or only intentional non-logging cases). 3) -v smoke check: invoke an op cli with -v and confirm INFO-level log lines appear, e.g. `python -m microscopy_pipeline.ops.extract_times -i <dir> -o out.csv -v` (or via the mp-extract-times entry point) emits the 'wrote N rows' message at INFO and runs without -v silently at WARNING. 4) If a folder op is run on a multi-file directory with tqdm installed, confirm a progress bar renders; with tqdm absent, confirm it still runs (graceful fallback). 5) Optionally add a small unit test asserting logger name == 'microscopy_pipeline' and that caplog captures an expected record.

**Risk:**
Low-to-moderate. Behavioral risk: tests or downstream code may assert on stdout (captured print output) rather than logging; switching to logger.info would break such assertions and require caplog/caplog usage instead (current tests pass and do not appear to rely on these prints, but confirm during edit). tqdm is not a current dependency (verified: not in pyproject.toml dependencies), so the optional-import guard is essential to avoid ImportError when tqdm is absent. Calling logging.basicConfig in every op cli() is benign for single-shot CLI use but is a global side effect; guard against double-configuration if cli() functions are composed. No changes to numeric/image outputs, so analysis results are unaffected.

---

### Generalize resumability (--skip-existing) across folder ops

**Status:** net_new | **Category:** robustness | **Effort:** M

**Where:**
- `microscopy_pipeline/ops/remove_lone_pixels.py:78-103 (remove_lone_pixels_folder; start_prefix loop at 92-98)`
- `microscopy_pipeline/ops/remove_lone_pixels.py:115,119-121 (--start-prefix CLI - the only existing resumability mechanism)`
- `microscopy_pipeline/io.py:223-266 (list_images / batch_apply - home for shared helper; batch_apply currently unused by any op)`
- `microscopy_pipeline/ops/crop.py:37-41 (crop_folder)`
- `microscopy_pipeline/ops/clahe.py:60-64 (clahe_folder)`
- `microscopy_pipeline/ops/scale_brightness.py:30-33 (scale_brightness_folder)`
- `microscopy_pipeline/ops/scale_colors.py:29-33 (scale_colors_folder; renames to .png)`
- `microscopy_pipeline/ops/grey_to_color.py:52-55 (grey_to_color_folder)`
- `microscopy_pipeline/ops/mask.py:150-154 (mask_folder)`
- `microscopy_pipeline/ops/scale_bar.py:69-75 (add_scale_bar_folder)`
- `microscopy_pipeline/ops/fuse_tiffs.py:52-65 (fuse_average_folder; _superimposed suffix)`
- `microscopy_pipeline/ops/complex_edf.py:134-145 (complex_edf_folder; _edf.tif suffix)`
- `microscopy_pipeline/ops/identify_core.py:188-228 (identify_core_folder; per-file PNGs + aggregate CSV)`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py:162-215 (track_organoid_folder; per-file _contour + aggregate CSVs)`
- `microscopy_pipeline/ops/sum_fluorescence.py:52-101 (sum_fluorescence_folder; single CSV output)`
- `microscopy_pipeline/workflows.py:147-189 (pngs_to_fused_images)`
- `microscopy_pipeline/workflows.py:245-272 (brightfield_to_growth_graphs)`
- `microscopy_pipeline/workflows.py:279-312 (fluorescence_to_curve)`

**Problem:**
Resumability exists in exactly ONE place: remove_lone_pixels_folder's `start_prefix` param (remove_lone_pixels.py:80,92-98) exposed as --start-prefix (line 115). It is positional/manual (operator must know the prefix to resume from) and op-specific. The other 13 *_folder functions unconditionally recompute and rewrite every output on every run; re-running an interrupted folder op redoes all completed work. REPO_MAP.md:244 already flags this as a planned task ("resume/skip-existing to all folder ops"); no skip_existing code exists anywhere (grep confirms). The 14 folder ops fall into three output shapes that a generic helper must handle: (a) per-file writers keeping the source name -- crop, clahe, scale_brightness, mask, add_scale_bar, grey_to_color; (b) per-file writers that rename/suffix -- scale_colors (->.png, scale_colors.py:32), fuse_average (_superimposed, fuse_tiffs.py:61-63), complex_edf (_edf.tif, complex_edf.py:140); (c) batched/aggregate writers where a single per-group or per-folder artifact gates the work -- remove_lone_pixels (per-prefix group of files), identify_core (per-file _outer_region.png + core_analysis_results.csv), track_organoid (per-file _contour + organoid_analysis_results.csv + summary), sum_fluorescence (one CSV). A naive 'skip if same-named output exists' helper is wrong for (b)/(c) because the output name is not src.name. io.batch_apply (io.py:242-266) is the only shared loop abstraction but is currently unused by any op.

**Fix:**
Add a shared helper in io.py near list_images/batch_apply, e.g. `def should_skip(out_path, *, skip_existing) -> bool: return bool(skip_existing) and Path(out_path).exists()`, plus optionally fold a `skip_existing: bool = False` kwarg into `batch_apply` (io.py:242) so it consults the predicate before calling func and counts skips. Then thread a `skip_existing: bool = False` keyword through every *_folder signature and have each loop compute its real `out` path FIRST (each op already does this: crop.py:41, scale_colors.py:32, fuse_tiffs.py:63, complex_edf.py:140, etc.) and `continue`/skip when io.should_skip(out, skip_existing=skip_existing). For per-file ops (a)+(b) this is a per-iteration guard on the actual destination name (respecting renames/suffixes). For batched ops: remove_lone_pixels_folder -- skip a prefix group only when ALL files' outputs exist (compose with existing start_prefix, do not remove it); identify_core_folder / track_organoid_folder / sum_fluorescence_folder -- guard on the aggregate CSV (e.g. organoid_analysis_results.csv at brightfield_organoid_tracker.py:202, core_analysis_results.csv at identify_core.py:226, output_csv at sum_fluorescence.py:92) so re-running with the CSV present is a no-op, or guard per-file overlays where applicable. Add `parser.add_argument('--skip-existing', action='store_true', help='Skip files whose output already exists.')` to each op's cli() (mirroring the --start-prefix add at remove_lone_pixels.py:115) and pass args.skip_existing through (mirror lines 119-121). Surface a matching `skip_existing` kwarg on the three folder-driving workflows -- pngs_to_fused_images (workflows.py:147; guard the `out` at line 172 before fuse_average_file/complex_edf_file), brightfield_to_growth_graphs (workflows.py:245; forward into track_organoid_folder), fluorescence_to_curve (workflows.py:279; forward into sum_fluorescence_folder) -- forwarding to the underlying ops. Update __init__.py docstrings only if signatures are re-exported (they are at ops/__init__.py:14-52; no list change needed since names are unchanged).

**Touch points:**
- `microscopy_pipeline/io.py (add should_skip helper; optionally extend batch_apply with skip_existing)`
- `microscopy_pipeline/ops/crop.py (crop_folder + cli)`
- `microscopy_pipeline/ops/clahe.py (clahe_folder + cli)`
- `microscopy_pipeline/ops/scale_brightness.py (scale_brightness_folder + cli)`
- `microscopy_pipeline/ops/scale_colors.py (scale_colors_folder + cli)`
- `microscopy_pipeline/ops/grey_to_color.py (grey_to_color_folder + cli)`
- `microscopy_pipeline/ops/mask.py (mask_folder + cli)`
- `microscopy_pipeline/ops/scale_bar.py (add_scale_bar_folder + cli)`
- `microscopy_pipeline/ops/fuse_tiffs.py (fuse_average_folder + cli)`
- `microscopy_pipeline/ops/complex_edf.py (complex_edf_folder + cli)`
- `microscopy_pipeline/ops/remove_lone_pixels.py (remove_lone_pixels_folder + cli; compose with existing start_prefix)`
- `microscopy_pipeline/ops/identify_core.py (identify_core_folder + cli; aggregate-CSV / per-file guard)`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py (track_organoid_folder + cli; aggregate-CSV guard)`
- `microscopy_pipeline/ops/sum_fluorescence.py (sum_fluorescence_folder + cli; output-CSV guard)`
- `microscopy_pipeline/workflows.py (pngs_to_fused_images, brightfield_to_growth_graphs, fluorescence_to_curve)`
- `tests/test_workflows.py (and any per-op tests asserting no-rewrite behaviour)`

**Verify:**
Run a folder op twice into the same output dir, the second time with --skip-existing, and confirm no files are rewritten: capture mtimes after run 1, run again with --skip-existing, assert all output mtimes unchanged (e.g. `python -m pytest` a new test that records st_mtime_ns of every output, re-invokes the *_folder fn with skip_existing=True, and asserts equality). Concretely for a per-file op: `crop_folder(inp, out); m={p:p.stat().st_mtime_ns for p in Path(out).iterdir()}; crop_folder(inp, out, skip_existing=True); assert {p:p.stat().st_mtime_ns for p in Path(out).iterdir()}==m`. For aggregate ops (track_organoid/identify_core/sum_fluorescence) assert the CSV mtime is unchanged on the second skip_existing run. Also confirm `python -m microscopy_pipeline.ops.crop --help` (and peers) shows --skip-existing, and that omitting the flag preserves current overwrite behaviour. Run existing suite (pytest) to ensure no regressions.

**Risk:**
Batched ops are the trap: skipping per-file in remove_lone_pixels/identify_core/track_organoid is unsound because those write aggregate CSVs that summarize ALL files -- a partial skip would emit a CSV missing skipped rows (or, for find_lone_pixels, change the per-group statistics). Must gate at group/CSV granularity, not per-file, for those four. scale_colors/fuse_average/complex_edf rename outputs, so a same-name check is wrong -- must check the actual destination (.png/_superimposed/_edf.tif). sum_fluorescence and the graph/plot stages overwrite a single artifact; skip_existing there means skip the whole stage. Threading a new kwarg through 14 ops + 3 workflows is broad and mechanical; easy to miss one cli() wiring (compare against the remove_lone_pixels --start-prefix pattern). Low semantic risk to the default path since the flag defaults False.

---

### Metadata-driven pixel size: read/preserve TIFF resolution tags and use as --pixel-size fallback

**Status:** net_new | **Category:** robustness | **Effort:** M

**Where:**
- `microscopy_pipeline/io.py:168-216 (save_stack, no resolution tags)`
- `microscopy_pipeline/io.py:144-148 (save_image)`
- `microscopy_pipeline/io.py:130-141 (load_image)`
- `microscopy_pipeline/io.py:155-165 (load_stack)`
- `microscopy_pipeline/ops/scale_bar.py:55-66 (add_scale_bar_file, cv2.imread drops tags)`
- `microscopy_pipeline/ops/scale_bar.py:78-90 (cli, --pixel-size-um required=True line 81)`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py:162-215 (track_organoid_folder), 218-249 (cli, --pixel-size-um line 222)`
- `microscopy_pipeline/ops/graph_contour_data.py:126-150, 153-168 (--pixel-size line 157)`
- `microscopy_pipeline/workflows.py:245-272 (brightfield_to_growth_graphs)`

**Problem:**
No part of the package reads or preserves the physical pixel size stored in TIFF resolution tags (XResolution=282, YResolution=283, ResolutionUnit=296). Confirmed by a round-trip test: after io.save_stack writes a stack, tags 282/283/296 are all absent (info dpi reads (1,1)). io.py has zero resolution/pixel-size helpers (grep for resolution/dpi/tag_v2 finds nothing in io.py except the unrelated ImageJ tiffinfo block at line 214). save_stack (168-216) writes ImageJ tags 50839/270/305/269 plus optional extra_tags but never the resolution tags. save_image (144-148) only forwards **save_kwargs. load_image (130-141) and load_stack (155-165) return bare ndarrays, discarding all metadata. Consequently every consumer demands the pixel size be re-typed on the command line: scale_bar.cli makes --pixel-size-um required=True (scale_bar.py:81) and its file path reads via cv2.imread (scale_bar.py:57) which strips TIFF tags entirely; brightfield_organoid_tracker.cli defaults --pixel-size-um to None (line 222) so area_um2/perimeter_um are silently zeroed (tracker lines 187-192) when omitted; graph_contour_data --pixel-size defaults None (line 157) so um2 plots are skipped. There is no way to carry a known calibration through the pipeline, and any micron calibration on input TIFFs is lost on every save. identify_core does not use pixel size (verified: no pixel/resolution refs) and is out of scope. Both PIL 10.4.0 and tifffile 2023.7.10 are installed.

**Fix:**
Add two io helpers and wire fallbacks. (1) io.read_pixel_size_um(path) -> Optional[float]: open with PIL, read tag_v2[282]/[283] (RATIONAL pixels-per-unit) and [296] ResolutionUnit (2=inch, 3=cm; 1=none -> unknown); convert to micrometres per pixel as 1/(res_per_unit) * unit_to_um (cm->1e4, inch->2.54e4), averaging X/Y; return None when tags absent, unit is None(1), or value is non-finite/<=0. Use the PIL constants (X_RESOLUTION=282, Y_RESOLUTION=283, RESOLUTION_UNIT=296) rather than magic numbers. (2) io.resolution_tags_for_pixel_size(pixel_size_um) -> dict building {282:(num,den), 283:(num,den), 296:3} as cm-based RATIONALs (res_per_cm = 1e4/pixel_size_um), so the value survives PIL/ImageJ/Fiji. (3) save_stack: accept pixel_size_um: Optional[float]=None and merge resolution_tags_for_pixel_size into tiff_tags before the tiffinfo assignment at line 213-214 (extra_tags must still win); also accept an optional source_path to copy-through existing resolution when pixel_size_um not given. (4) save_image: when path is a TIFF and pixel_size_um given, pass dpi/resolution via tiffinfo using array_to_pil's image. (5) Fallback wiring: in scale_bar.cli, make --pixel-size-um optional (default=None) and, when not supplied, call io.read_pixel_size_um(input) per file, erroring only if still None; add_scale_bar_file should switch its read to io.load_image (or additionally read tags) since cv2.imread cannot see TIFF tags. (6) brightfield_organoid_tracker.track_organoid_folder/track_organoid_file: when pixel_size_um is None, fall back to io.read_pixel_size_um(src) so area_um2/perimeter_um populate; this also feeds graph_contour_data indirectly via the CSV. (7) workflows.brightfield_to_growth_graphs: leave threading as-is; the tracker fallback now supplies pixel size when the arg is None. Keep graph_contour_data signature unchanged (CSV-driven, no image to inspect).

**Touch points:**
- `microscopy_pipeline/io.py: add read_pixel_size_um() and resolution_tags_for_pixel_size(); extend save_stack (168-216) and save_image (144-148) signatures with pixel_size_um/source_path and write resolution tags`
- `microscopy_pipeline/ops/scale_bar.py: cli() make --pixel-size-um optional with metadata fallback; add_scale_bar_file read tags via io instead of relying on cv2.imread`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py: track_organoid_folder/track_organoid_file fall back to io.read_pixel_size_um when pixel_size_um is None`
- `microscopy_pipeline/workflows.py: relies on tracker fallback (likely no signature change; verify brightfield_to_growth_graphs still passes through)`
- `tests: new round-trip test (write TIFF with known resolution via save_stack/save_image, read back via read_pixel_size_um); test scale_bar/tracker metadata fallback`

**Verify:**
Round-trip unit test: build a known stack, call io.save_stack(frames, path, pixel_size_um=0.5); reopen with PIL and assert tag_v2 contains 282/283/296 and that io.read_pixel_size_um(path) returns approx 0.5 (within 1e-3). Negative test: a TIFF saved without pixel_size_um (or with ResolutionUnit=1) -> read_pixel_size_um returns None. Fallback test: run scale_bar.cli on that calibrated TIFF WITHOUT --pixel-size-um and confirm a bar is drawn (bar_px == round(scale_um/0.5)); run tracker without --pixel-size-um on a calibrated TIFF and assert area_um2 > 0 in the CSV. Sanity: existing ImageJ tags (50839/270/305/269) still present after adding resolution tags. Current baseline (pre-fix) confirmed: tags 282/283/296 absent after save_stack.

**Risk:**
RATIONAL tag encoding via PIL must be (numerator, denominator) tuples; passing a bare float can raise or truncate. ResolutionUnit semantics: choosing cm(3) vs inch(2) must be consistent between writer and reader or values are off by 2.54x. ImageJ also encodes spacing inside the 50839/270 ImageJ string (unit=, spacing=) which is separate from resolution tags; to avoid contradicting the resolution tags, keep ImageJ string unit-agnostic or align it. cv2.imread in add_scale_bar_file silently ignores TIFF tags, so the fallback MUST route reads through PIL/io or it will appear to work yet never find the calibration. read_pixel_size_um must guard against den==0, missing keys, and ResolutionUnit==1 to avoid div-by-zero / bogus tiny pixel sizes. extra_tags precedence in save_stack must not be clobbered by injected resolution tags.

---

### Optional --jobs parallelism for embarrassingly-parallel per-file folder ops

**Status:** net_new | **Category:** robustness | **Effort:** M

**Where:**
- `microscopy_pipeline/io.py:242-266 (batch_apply - existing but unused helper, natural home for parallel map)`
- `microscopy_pipeline/ops/crop.py:37-41 (crop_folder)`
- `microscopy_pipeline/ops/clahe.py:60-64 (clahe_folder)`
- `microscopy_pipeline/ops/mask.py:150-154 (mask_folder)`
- `microscopy_pipeline/ops/scale_brightness.py:30-33 (scale_brightness_folder)`
- `microscopy_pipeline/ops/scale_colors.py:29-33 (scale_colors_folder)`
- `microscopy_pipeline/ops/grey_to_color.py:52-55 (grey_to_color_folder)`
- `microscopy_pipeline/ops/scale_bar.py:69-75 (add_scale_bar_folder)`
- `microscopy_pipeline/ops/complex_edf.py:134-145 (complex_edf_folder; iterates sorted(Path.iterdir()))`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py:162-215 (track_organoid_folder; per-file compute + ordered CSV aggregation)`
- `microscopy_pipeline/ops/identify_core.py:188-228 (identify_core_folder; per-file compute + ordered CSV aggregation)`
- `microscopy_pipeline/ops/align.py:63-121 (align_session - MUST stay serial: temporal dependency)`
- `microscopy_pipeline/ops/remove_lone_pixels.py:78-103 (remove_lone_pixels_folder - MUST stay serial: cross-image grouping)`
- `microscopy_pipeline/cli.py:20-45 (build_io_parser - add shared --jobs flag here)`
- `tests/conftest.py (synthetic fixtures: brightfield_dir, fluorescence_dir, focal_pngs)`

**Problem:**
No folder op in the package runs in parallel; every `*_folder` function is a hand-rolled serial loop of the form `for src in io.list_images(input_dir): <op>_file(str(src), str(dst), **kwargs)`. Alignment and EDF are noted as the slow stages (REPO_MAP.md:254), and the existing `io.batch_apply` helper (io.py:242) is unused by these ops. There is currently NO concurrency anywhere in the package (grep for concurrent.futures/multiprocessing/--jobs/max_workers finds only the REPO_MAP wishlist line). Two ops must NOT be parallelized: `align_session` (align.py:101-103 reads `output_dir/{tp-1}_{base_z}.png` written by the previous timepoint iteration -> hard temporal dependency) and `remove_lone_pixels_folder` (remove_lone_pixels.py:84,100-101 calls `find_lone_pixels` over a whole prefix group -> cross-image reduction, not per-file). Two analysis ops (`track_organoid_folder`, `identify_core_folder`) are per-file independent in compute but accumulate ordered result lists (`detailed`/`rows`) and emit aggregate CSVs after the loop, so any parallel map must restore deterministic input order before writing the CSV.

**Fix:**
1) Add a shared `--jobs N` flag (default 1 = serial) to `build_io_parser` in cli.py so every op CLI inherits it consistently, plus a `jobs=1` keyword on each parallelizable `*_folder` function. 2) Add a single shared helper in io.py (extend/replace the unused `batch_apply`, e.g. `parallel_map(files, func, *, jobs=1)`) that: with jobs<=1 runs the existing serial loop unchanged; with jobs>1 submits `(src, dst)` tasks to a `concurrent.futures.ProcessPoolExecutor(max_workers=jobs)` and collects results indexed by submission order so output ordering is deterministic (iterate `executor.map(...)` over the already-sorted `io.list_images` list, OR submit with enumerate and reorder by index). Preserve the existing per-file try/except so one bad file does not kill the pool. 3) Refactor the 8 pure per-file ops (crop, clahe, mask, scale_brightness, scale_colors, grey_to_color, scale_bar, complex_edf) to route their loop through the helper, passing a top-level picklable callable (module-level `*_file` function bound via `functools.partial` on the kwargs - closures are NOT picklable under ProcessPoolExecutor on Windows spawn). 4) For track_organoid_folder and identify_core_folder, parallelize ONLY the per-file `*_file` calls, return results tagged with their input index, sort by that index before building `detailed`/`rows`, then write the CSVs exactly as today (byte-identical ordering). 5) Leave align_session and remove_lone_pixels_folder serial; do NOT add --jobs to their CLIs (or accept-and-ignore with a note). 6) Guard jobs against `os.cpu_count()` and validate `jobs >= 1` in the CLI.

**Touch points:**
- `microscopy_pipeline/io.py (add parallel_map / rework batch_apply; import concurrent.futures, functools, os)`
- `microscopy_pipeline/cli.py (add --jobs to build_io_parser)`
- `microscopy_pipeline/ops/crop.py`
- `microscopy_pipeline/ops/clahe.py`
- `microscopy_pipeline/ops/mask.py`
- `microscopy_pipeline/ops/scale_brightness.py`
- `microscopy_pipeline/ops/scale_colors.py`
- `microscopy_pipeline/ops/grey_to_color.py`
- `microscopy_pipeline/ops/scale_bar.py`
- `microscopy_pipeline/ops/complex_edf.py`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py`
- `microscopy_pipeline/ops/identify_core.py`
- `tests/ (new test_parallel.py asserting jobs=2 == serial on fixtures)`

**Verify:**
Add a pytest (e.g. tests/test_parallel.py) that runs each parallelizable folder op twice on a conftest fixture (brightfield_dir for tracker, fluorescence_dir / a small png dir for crop/clahe/scale_brightness/scale_colors/grey_to_color/scale_bar/complex_edf, identify_core on a disk fixture) with jobs=1 and jobs=2 into separate temp output dirs, then assert byte-for-byte equality of every output image (np.array_equal on loaded arrays) AND identical CSV contents/row order for track_organoid_folder and identify_core_folder. Confirm align_session and remove_lone_pixels_folder either reject --jobs>1 or ignore it. Run `python -m pytest tests/ -q` (Python 3.8 here; concurrent.futures import confirmed available) and ensure no regressions in existing test_workflows.py / test_harmonization.py. Manually sanity-check picklability by actually exercising jobs=2 (ProcessPoolExecutor uses spawn on Windows, so closures/lambdas would raise PicklingError - the test must use the real ProcessPoolExecutor path, not a thread pool).

**Risk:**
ProcessPoolExecutor on Windows uses spawn: any non-top-level/lambda task callable or unpicklable kwarg raises PicklingError, and the whole `microscopy_pipeline` import (cv2, matplotlib) re-runs in each worker, so import side effects must stay clean. Process startup overhead can make jobs>1 SLOWER than serial on the tiny synthetic fixtures - the verification asserts equality, not speedup. Determinism risk: naive `as_completed` collection would scramble CSV row order in track_organoid/identify_core; must reorder by input index. Per-file exception handling currently prints and continues; the pooled version must preserve that (a worker exception surfaces at result-collection time, not at submission). Default must remain jobs=1 to avoid changing existing behavior/perf for current callers and workflows.py.

---

## Phase 4 - Configurable branch points

### Expose alignment warp mode (translation/euclidean/affine/homography) through align_session, cli, and workflow

**Status:** confirmed_present | **Category:** branch-point | **Effort:** M

**Where:**
- `microscopy_pipeline/ops/align.py:21-24 (align_pair signature with warp_mode default cv2.MOTION_TRANSLATION)`
- `microscopy_pipeline/ops/align.py:38-40 (align_stack already threads warp_mode)`
- `microscopy_pipeline/ops/align.py:63-75 (align_session signature — NO warp_mode)`
- `microscopy_pipeline/ops/align.py:98,105,120 (align_pair calls inside align_session use default translation)`
- `microscopy_pipeline/ops/align.py:124-151 (cli — no --warp-mode arg)`
- `microscopy_pipeline/workflows.py:196-221 (align_crop_colorize calls ops.align_stack without warp_mode)`

**Problem:**
align_pair (align.py:21-24) already accepts warp_mode: int = cv2.MOTION_TRANSLATION and dispatches correctly (np.eye(3,3) + warpPerspective for MOTION_HOMOGRAPHY at lines 28,33-34, else np.eye(2,3) + warpAffine). align_stack (align.py:38-40, 56-57) also already threads warp_mode through to align_pair. However the higher layers are locked to translation-only: align_session (align.py:63-75) has no warp_mode parameter and its three internal align_pair calls (lines 98, 105, 120) all use the default; the cli (align.py:124-151) exposes no --warp-mode flag; and workflows.align_crop_colorize (workflows.py:221) calls ops.align_stack(frames, reference=reference) with no warp_mode. So users running the directory session or CLI cannot choose euclidean/affine/homography even though the underlying ECC solver supports them. Confirmed cv2 constants: MOTION_TRANSLATION=0, EUCLIDEAN=1, AFFINE=2, HOMOGRAPHY=3. This is a configurability branch point, distinct from the excluded EDF focus rule.

**Fix:**
Add a module-level mapping in align.py, e.g. WARP_MODES = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN, "affine": cv2.MOTION_AFFINE, "homography": cv2.MOTION_HOMOGRAPHY}. Add warp_mode: int = cv2.MOTION_TRANSLATION keyword param to align_session (line 63-75 signature) and pass warp_mode=warp_mode into all three internal align_pair calls (lines 98, 105, 120). In cli (lines 124-151) add parser.add_argument("--warp-mode", choices=list(WARP_MODES), default="translation", ...) and pass warp_mode=WARP_MODES[args.warp_mode] in the align_session(...) call (lines 140-151). Optionally thread warp_mode through workflows.align_crop_colorize (workflows.py:196-221) into the ops.align_stack call at line 221 for parity. Keep translation as the default everywhere so existing behavior is unchanged.

**Touch points:**
- `microscopy_pipeline/ops/align.py: add WARP_MODES dict near imports (after line 18)`
- `microscopy_pipeline/ops/align.py: align_session signature + 3 align_pair calls (lines 63-75, 98, 105, 120)`
- `microscopy_pipeline/ops/align.py: cli add_argument + align_session call (lines 129-139, 140-151)`
- `microscopy_pipeline/workflows.py: optional warp_mode param + align_stack call (lines 196, 221)`

**Verify:**
drift_frames fixture (tests/conftest.py:52-56) produces four 64x64 uint8 disks drifting +2px in x per frame. Add/extend a test asserting align_stack(drift_frames, warp_mode=cv2.MOTION_EUCLIDEAN) returns frames whose disk centroids are realigned to the reference (first frame), and a CLI smoke test that --warp-mode euclidean is accepted and maps to cv2.MOTION_EUCLIDEAN. Existing test_align_crop_colorize (tests/test_workflows.py:51-60) must still pass with the translation default. Run: python -m pytest tests/test_workflows.py -q.

**Risk:**
Low. Default stays cv2.MOTION_TRANSLATION so existing pipelines/tests are unaffected. Non-translation modes (especially homography) may fail to converge on sparse/synthetic frames and raise cv2.error — align_stack already catches cv2.error (lines 58-59) and returns the frame unchanged, but align_session has no such guard, so euclidean/affine/homography could now raise inside a session run; consider whether to leave that as-is (caller-visible failure) or wrap. choices= validation prevents invalid CLI strings.

---

### Add --projection {mean,max,median,std,percentile} to fuse_tiffs (mean default, fuse_average back-compat alias)

**Status:** confirmed_present | **Category:** branch-point | **Effort:** M

**Where:**
- `microscopy_pipeline/ops/fuse_tiffs.py:17-41 (fuse_average core, mean only)`
- `microscopy_pipeline/ops/fuse_tiffs.py:44-49 (fuse_average_file)`
- `microscopy_pipeline/ops/fuse_tiffs.py:52-65 (fuse_average_folder)`
- `microscopy_pipeline/ops/fuse_tiffs.py:68-84 (cli)`
- `microscopy_pipeline/ops/__init__.py:32,71 (exports)`
- `microscopy_pipeline/workflows.py:63-71 (_fuse_stack)`
- `microscopy_pipeline/workflows.py:147-189 (pngs_to_fused_images)`

**Problem:**
fuse_average in microscopy_pipeline/ops/fuse_tiffs.py (lines 17-41) hard-codes a MEAN projection: it accumulates summed += f.astype(acc_dtype) over all frames (lines 31-33) then divides averaged = summed / len(frames) (line 34). There is no way to collapse a focal stack with any other statistic. The only CLI knob is --brightfield (line 73); there is no --projection flag. The folder wrapper hard-bakes the projection name into the output suffix ('_superimposed', line 61). Both workflow call sites are mean-locked: _fuse_stack (workflows.py:69-70) calls ops.fuse_average for method=='average', and pngs_to_fused_images (workflows.py:173-174) calls ops.fuse_average_file for method=='average'. For high-dynamic-range focal stacks a MAX projection (or median/std/percentile) is often the desired fusion mode; the package offers none.

**Fix:**
Refactor fuse_tiffs.py to a generic fuse_project(frames, *, projection='mean', percentile=90.0, brightfield=False, bit_depth=None) core that stacks frames and dispatches over a {mean,max,median,std,percentile} map onto float, then applies the existing brightfield range-stretch (lines 36-39), clip, and dtype cast (lines 40-41). Keep mean as the DEFAULT so current behavior is unchanged. Add fuse_average(frames, **kw) as a thin back-compat alias delegating with projection='mean' (preserve its signature so existing callers and ops.__init__ export at __init__.py:32/71 keep working). Thread a projection (and percentile) kwarg through fuse_average_file and fuse_average_folder (and derive the output suffix from the projection name instead of the literal '_superimposed' at line 61, keeping '_superimposed' for mean for back-compat). In cli() add parser.add_argument('--projection', choices=['mean','max','median','std','percentile'], default='mean') and an optional '--percentile' float (default 90), passing them through both dispatch branches (lines 78,80). In workflows.py thread a projection kwarg through _fuse_stack (extend the method=='average' branch at line 70 to pass projection) and through pngs_to_fused_images (pass projection to ops.fuse_average_file at line 174); keep method='average'+projection='mean' as defaults so focal_stack_to_gif/pngs_to_fused_images behavior is unchanged.

**Touch points:**
- `microscopy_pipeline/ops/fuse_tiffs.py (core fuse_project + fuse_average alias, _file/_folder wrappers, cli --projection/--percentile)`
- `microscopy_pipeline/ops/__init__.py (export fuse_project alongside fuse_average*, add to __all__)`
- `microscopy_pipeline/workflows.py (_fuse_stack projection kwarg; pngs_to_fused_images projection kwarg + docstring)`
- `tests/test_workflows.py (extend test_pngs_to_fused_images_average / add projection assertion)`

**Verify:**
Build a focal stack with a bright disk (conftest make_disk val=50000 bg=2000, as in the focal_stacks fixture) and one dim frame. Confirm fuse_project(frames, projection='max') yields disk-region values >= the per-frame max (= the brightest frame) while fuse_project(frames, projection='mean') yields the average; assert max_result[disk].mean() > mean_result[disk].mean() on the bright disk. Confirm fuse_average(frames) still equals fuse_project(frames, projection='mean') (back-compat). Run CLI with --projection max on a stack TIFF and confirm output differs from default; run existing pytest tests/test_workflows.py to confirm mean-default workflows are unchanged.

**Risk:**
Low-to-moderate. Main risk is breaking the existing fuse_average public API (exported in ops/__init__.py and called at workflows.py:70,174) and the '_superimposed' output-suffix contract — mitigated by keeping fuse_average as an alias, mean as default, and the mean suffix unchanged. std/percentile projections produce float intermediates that must be clipped/cast to the target bit depth (reuse existing clip+cast at lines 40-41) to avoid overflow/dtype surprises; std output is low-magnitude and may look near-black, which is expected, not a bug.

---

### Add region-restricted fluorescence summation (--mask / --region) with optional background subtraction

**Status:** confirmed_present | **Category:** capability | **Effort:** M

**Where:**
- `microscopy_pipeline/ops/sum_fluorescence.py:20-33 (sum_brightness)`
- `microscopy_pipeline/ops/sum_fluorescence.py:52-101 (sum_fluorescence_folder)`
- `microscopy_pipeline/ops/sum_fluorescence.py:104-114 (cli)`
- `microscopy_pipeline/ops/identify_core.py:25-146 (identify_core returns core_mask/outer_mask)`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py:97-138 (track_organoid returns contour), :115-116 (cv2.fillPoly mask)`
- `microscopy_pipeline/ops/mask.py:61-77 (_load_mask_array helper)`

**Problem:**
sum_brightness and sum_fluorescence_folder sum the WHOLE image only. sum_brightness(image, *, brightfield, max_value) (sum_fluorescence.py:20-33) does np.sum over the entire array; the folder loop (lines 73-90) calls it on the full frame with no spatial restriction. The CLI (lines 104-114) exposes only --brightfield and --timestamps. There is no way to sum fluorescence within an organoid region nor to subtract background, even though identify_core (identify_core.py:25) already produces core_mask/outer_mask boolean arrays and track_organoid (brightfield_organoid_tracker.py:97) produces a fillable contour. This blocks the unified organoid analysis goal (organoid area from the tracker + masked intensity over time from fluorescence).

**Fix:**
1) Extend sum_brightness to accept an optional region mask: add a `mask: Optional[np.ndarray] = None` (and optional `background: Optional[float]` for per-pixel background subtraction) kwarg; when provided, validate mask.shape[:2] == image.shape[:2], coerce to boolean, optionally subtract background before clipping at 0, and sum only arr[mask]. Keep current whole-image behavior when mask is None. 2) In sum_fluorescence_folder add a `mask` parameter accepting either a single mask path/array (applied to every frame) or a callable/source that derives a per-frame mask. For --region organoid, derive the mask per frame by calling identify_core(arr) and OR-ing core_mask|outer_mask, or by calling track_organoid(arr) and rasterizing its contour with cv2.fillPoly (mirror brightfield_organoid_tracker.py:115-116); for --region whole, behave as today. Reuse mask.py:_load_mask_array (lines 61-77) for loading/shape-validating a static --mask PATH so shape-mismatch handling stays consistent. Emit both the masked sum and (optionally) the masked area to the CSV so area+intensity-over-time live in one table; add a background_subtraction column/option (e.g. subtract median of the non-mask region). 3) In cli() add mutually-informing flags: `--mask PATH` (static mask file) and `--region {whole,organoid}` (default whole); optionally `--subtract-background`. Validate that --mask and --region organoid are not both set. Update ops/__init__.py exports only if a new public helper is added (sum_brightness/sum_fluorescence_folder already exported at __init__.py:43).

**Touch points:**
- `microscopy_pipeline/ops/sum_fluorescence.py: extend sum_brightness signature/body, sum_fluorescence_folder signature/loop, cli() argparse`
- `microscopy_pipeline/ops/identify_core.py: reuse identify_core() for organoid-region masks (no change, import)`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py: reuse track_organoid()/contour rasterization for organoid-region masks (no change, import)`
- `microscopy_pipeline/ops/mask.py: reuse _load_mask_array() for --mask PATH loading/validation`
- `microscopy_pipeline/workflows.py: optionally thread mask/region through fluorescence_to_curve`
- `tests/test_workflows.py: add masked-sum test alongside test_fluorescence_to_curve (lines 93-103)`

**Verify:**
Synthetic check: build an array that is nonzero everywhere (e.g. uniform value or a bright disk on a bright field) and a boolean mask covering a strict sub-region. Assert sum_brightness(img, mask=submask) < sum_brightness(img) (masked sum strictly less than whole-image sum) and equals np.sum(img[submask]). For the folder path, write a couple of *_N_*.tif frames plus a mask PNG, run sum_fluorescence_folder(..., mask=mask_path) (or --region organoid), and assert each row's masked sum is < the whole-image sum for the same frame and that a masked-area column is populated. Confirm background subtraction lowers the sum further when enabled. Run pytest tests/test_workflows.py and ensure existing assertions (sum_brightness/sum_optical_density present, rising trend) still pass with default whole-region behavior.

**Risk:**
Per-frame organoid mask derivation (identify_core/track_organoid) is heavy and can return None / empty mask -> must fall back gracefully (e.g. skip frame or record 0 with a flag) without crashing the batch. Mask vs image shape mismatches must raise clearly (reuse _load_mask_array's ValueError). brightfield mode inverts pixels (max_value - arr) before summing, so masking/background subtraction must happen in the same value space to stay consistent. Must preserve default whole-image behavior so existing tests and the fluorescence_to_curve workflow are unchanged. CSV schema changes (new columns) could break downstream plot_summed_fluorescence/graph consumers if they assume fixed columns.

---

### Optional MP4 export selected by extension + per-frame timestamp/label overlay

**Status:** net_new | **Category:** capability | **Effort:** M

**Where:**
- `microscopy_pipeline/ops/tiff_to_gif.py:38-58 (frames_to_gif)`
- `microscopy_pipeline/ops/tiff_to_gif.py:61-78 (tiff_to_gif)`
- `microscopy_pipeline/ops/tiff_to_gif.py:81-89 (cli)`
- `microscopy_pipeline/workflows.py:78-140 (focal_stack_to_gif, write at line 139)`
- `microscopy_pipeline/ops/__init__.py:36,72 (tiff_to_gif/frames_to_gif exports)`
- `pyproject.toml:25-26 (optional-dependencies)`
- `REPO_MAP.md:264-265 (planned item #15)`

**Problem:**
tiff_to_gif.py only writes animated GIF: both frames_to_gif (lines 38-58) and tiff_to_gif (lines 61-78) hardcode PIL Image.save(save_all=True, ...), and cli() (lines 81-89) exposes only --pattern/--duration-ms/--loop with no format or overlay option. workflows.focal_stack_to_gif hardcodes ops.frames_to_gif(...) at workflows.py:139 and documents output as a ".gif" path (lines 114, 138-139). There is no MP4/H.264 output path, no encoder (imageio/ffmpeg) is declared anywhere (pyproject.toml dependencies lines 14-23 and optional-dependencies lines 25-26 list none; a repo-wide grep for imageio|ffmpeg|mp4|ImageDraw|ImageFont|VideoWriter finds only the REPO_MAP.md:264 planning note), and there is no per-frame timestamp/label overlay. This is the planned capability REPO_MAP.md item #15 (lines 264-265: MP4/H.264 output + timestamp overlay) and ties to the gif-flicker normalization item since both touch the shared frame-normalization path (_normalize_to_8bit at lines 18-24).

**Fix:**
Add format dispatch keyed on output extension in the writer layer. (1) In tiff_to_gif.py add a frames_to_video(frames, output_path, *, fps/duration_ms, ...) helper that lazily imports imageio (try: import imageio[.v3]; except ImportError: raise a clear RuntimeError naming the missing optional dependency, e.g. "MP4 output requires 'imageio[ffmpeg]'; pip install microscopy-pipeline[video]"); reuse _normalize_to_8bit + RGB conversion so MP4 and GIF share identical frame prep (the tie-in to gif-flicker normalization). (2) Add a small dispatcher (e.g. write_animation or extend frames_to_gif's callers) that picks GIF vs MP4 from Path(output_path).suffix.lower(): .gif -> existing PIL path (default), .mp4/.mov/.avi -> frames_to_video. (3) Add an optional per-frame overlay: a labels: Optional[Sequence[str]] (or a callable) parameter plus a helper that draws text per frame, following the existing scale_bar.py precedent (cv2.putText/cv2.rectangle, ops/scale_bar.py:46-49) or PIL ImageDraw/ImageFont, applied uniformly before encoding so it works for both GIF and MP4. (4) Thread the new params through tiff_to_gif() and add CLI flags (--fps or reuse --duration-ms, --label/--timestamp). (5) In workflows.focal_stack_to_gif, route the line 139 write through the new dispatcher so an .mp4 output_path is honored, plumb an optional per-frame label/timestamp param, and update the docstring (currently ".gif" at lines 114/138). (6) Declare the optional dependency in pyproject.toml optional-dependencies (e.g. video = ["imageio[ffmpeg]"]). Keep GIF the default and behavior unchanged when no overlay/MP4 is requested; fail gracefully with a clear message when the encoder is unavailable.

**Touch points:**
- `microscopy_pipeline/ops/tiff_to_gif.py: add frames_to_video + extension dispatch + overlay helper; extend frames_to_gif/tiff_to_gif/cli signatures and flags`
- `microscopy_pipeline/workflows.py: route focal_stack_to_gif write (line 139) through the dispatcher, add optional label/timestamp param, update docstring (lines 114,138)`
- `microscopy_pipeline/ops/__init__.py: export any new public symbol (e.g. frames_to_video) alongside lines 36/72`
- `pyproject.toml: add optional-dependencies entry for the video encoder (after line 26)`
- `tests/test_workflows.py: add an .mp4 case (skip/xfail when imageio unavailable) mirroring lines 13-26`
- `REPO_MAP.md:264-265: mark item #15 progress`

**Verify:**
With the encoder installed, call workflows.focal_stack_to_gif(..., out.mp4) (or ops.tiff_to_gif with an .mp4 output) and assert the .mp4 file exists with size > 0, mirroring the existing GIF smoke tests at tests/test_workflows.py:13-26. With imageio absent, assert a clear RuntimeError/ImportError naming the missing optional dependency is raised (no silent failure, no partial file). Confirm GIF output is unchanged (default extension still produces a working GIF) and that requesting a timestamp/label overlay produces frames with the drawn text (e.g. visually or via a non-empty file plus no exception). Run pytest tests/test_workflows.py.

**Risk:**
imageio[ffmpeg] is heavyweight and platform-sensitive, so it must stay an optional extra and the import must be lazy to avoid breaking the GIF-only default install and the existing tests. The overlay must share the exact _normalize_to_8bit/RGB path used by GIF so MP4 and GIF stay visually consistent (coupling to the gif-flicker normalization item — uncoordinated changes could diverge). Extension dispatch must not silently mis-handle unknown suffixes (raise a clear error). Font handling for PIL ImageDraw (default vs truetype) can be environment-fragile; the cv2.putText precedent in scale_bar.py avoids that. MP4 even-dimension constraints (some H.264 encoders require width/height divisible by 2) may need padding to avoid encoder errors.

---

## Phase 5 - Optional (confirm need first)

### Heavy/optional segmentation + I/O capabilities - defer unless concretely needed

**Status:** net_new | **Category:** capability | **Effort:** L

**Where:**
- `microscopy_pipeline/ops/identify_core.py`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py`

**Problem:**
The two segmentation ops currently rely entirely on classical morphology/threshold heuristics with no pluggable backend. identify_core.identify_core (microscopy_pipeline/ops/identify_core.py:25-146) computes core/outer masks from percentile thresholds plus skimage morphology (remove_small_objects, binary_closing/opening, binary_fill_holes, ConvexHull fallback). brightfield_organoid_tracker.find_organoid_contour (microscopy_pipeline/ops/brightfield_organoid_tracker.py:36-84) uses CLAHE + Otsu/adaptive cv2.threshold + connectedComponentsWithStats + findContours, wrapped by track_organoid (line 97). All I/O is centralized in microscopy_pipeline/io.py (PIL-based load_image/load_stack/save_stack; IMAGE_EXTS/TIFF_EXTS at io.py:33-34) with no OME-TIFF/Bio-Formats support and no napari hand-off. Dependencies are a flat list in pyproject.toml:14-23 with only a 'test' extra at pyproject.toml:25-26. Three larger capability efforts have been floated; each pulls in heavy dependencies (deep-learning runtimes, GPU/CUDA, JVM/Java for Bio-Formats, Qt for napari) that would substantially bloat install size and CI for a package whose current deps are pure CPU/numpy. There is no demonstrated requirement for any of them today; this item exists to document the integration points and explicitly DEFER.

**Fix:**
Design-only; do NOT implement now. Document three deferred efforts as ONE backlog item, gated on a concrete need: (1) ML segmentation backend (Cellpose/StarDist) as an alternative to the morphology heuristics. Integration point: introduce a backend seam at identify_core.identify_core (identify_core.py:25) and brightfield_organoid_tracker.find_organoid_contour (brightfield_organoid_tracker.py:36) / track_organoid (line 97) -- e.g. a backend='morphology'|'cellpose'|'stardist' parameter threaded through the *_file/*_folder callers and the cli() arg parsers (identify_core.py:231, brightfield_organoid_tracker.py:218), with the ML path imported lazily and guarded behind a new optional-dependency extra (e.g. [project.optional-dependencies] ml = ['cellpose'] / segml = ['stardist','tensorflow'] in pyproject.toml:25). Heavy deps: torch or tensorflow, model weights download. (2) GPU / tiled large-image processing paths: optional CuPy/GPU acceleration and a tiled/chunked path (e.g. dask-image / zarr block processing) for whole-slide images exceeding RAM, plumbed through io.py loaders and the per-image op loops; gated behind a 'gpu'/'large' extra. Heavy deps: cupy (CUDA toolkit), dask, zarr. (3) OME-TIFF / Bio-Formats I/O + napari hand-off: add OME-TIFF read/write (tifffile/ome-types) and optional Bio-Formats (python-bioformats, needs a JVM) at the io.py layer alongside the existing PIL path, plus an optional napari viewer hand-off for interactive QC of masks/contours; gated behind 'omero'/'viz' extras. Heavy deps: tifffile, ome-types, JPype/JVM, napari + Qt. RECOMMENDATION: defer all three; revisit only when a concrete dataset/accuracy/scale requirement appears. If ever pursued, each must stay behind an optional extra and a lazy import so the core install remains pure-CPU and lightweight.

**Touch points:**
- `microscopy_pipeline/ops/identify_core.py`
- `microscopy_pipeline/ops/brightfield_organoid_tracker.py`
- `microscopy_pipeline/io.py`
- `pyproject.toml`

**Verify:**
n/a (design only) - no code change in this item; nothing to run or test. If/when implemented later, each backend/path must be exercised only with its optional extra installed and degrade cleanly (clear error) when the extra is absent.

**Risk:**
Scope creep and dependency bloat: Cellpose/StarDist pull torch/tensorflow + model weights, GPU paths pull CUDA/CuPy, Bio-Formats needs a JVM, and napari pulls Qt -- collectively dwarfing the current pure-CPU dependency set (pyproject.toml:14-23) and inflating install size, CI time, and maintenance. Implementing speculatively without a concrete need risks unused, hard-to-test code paths and platform-specific breakage. Mitigation: keep deferred; isolate behind optional extras + lazy imports if ever pursued.

---

## Suggested order & effort

| Item | Phase | Effort |
| --- | --- | --- |
| Replace np.choose with np.take_along_axis to lift 32-frame ceiling in complex-wavelet EDF | Phase 1 - Correctness | S |
| Add normalize={global,per-frame,none} (default global) to frames_to_gif to stop time-lapse brightness flicker | Phase 1 - Correctness | S |
| Remove unused io.batch_apply and cli.dispatch_file_or_folder (delete accompanying test for the latter) | Phase 1 - Correctness | S |
| Write README.md referenced by pyproject readme="README.md" | Phase 2 - Docs | S |
| Replace print() with module logging + add -v/--verbose and optional tqdm progress | Phase 3 - Robustness | M |
| Generalize resumability (--skip-existing) across folder ops | Phase 3 - Robustness | M |
| Metadata-driven pixel size: read/preserve TIFF resolution tags and use as --pixel-size fallback | Phase 3 - Robustness | M |
| Optional --jobs parallelism for embarrassingly-parallel per-file folder ops | Phase 3 - Robustness | M |
| Expose alignment warp mode (translation/euclidean/affine/homography) through align_session, cli, and workflow | Phase 4 - Configurable branch points | M |
| Add --projection {mean,max,median,std,percentile} to fuse_tiffs (mean default, fuse_average back-compat alias) | Phase 4 - Configurable branch points | M |
| Add region-restricted fluorescence summation (--mask / --region) with optional background subtraction | Phase 4 - Configurable branch points | M |
| Optional MP4 export selected by extension + per-frame timestamp/label overlay | Phase 4 - Configurable branch points | M |
| Heavy/optional segmentation + I/O capabilities - defer unless concretely needed | Phase 5 - Optional (confirm need first) | L |

**Recommendation:** Land **Phase 1 (correctness)** and **Phase 2 (README)** first — they are low-effort and either fix bugs or close a doc gap. Then take **Phase 3 (robustness)**, then **Phase 4 (configurable branch points)**. **Confirm a concrete need before touching Phase 5** (heavy/optional capabilities) — keep it deferred behind optional extras unless a real dataset/accuracy/scale requirement appears.

---

## Commands quick-reference

```bash
# Install (with test extra)
pip install -e ".[test]"

# Install (runtime only)
pip install -e .

# Run the test suite
python -m pytest -q

# Build a distribution (uses the setuptools backend declared in pyproject.toml)
python -m build

# Invoke an example op CLI (console scripts created on install)
mp-crop -i in/ -o out/ --left 10
```

Invoke an example workflow from Python:

```python
from microscopy_pipeline import workflows

# Fuse a focal stack (a dir of {X}_{Y}.png) and write an animated GIF.
# First positional arg `stacks` accepts either a directory path or a list of stacks.
workflows.focal_stack_to_gif("focal_pngs/", "out.gif")
```

---

See **REPO_MAP.md** at the repo root for the full agent-facing capability map, the detailed data-flow conventions (raw `{tp}_zs+{z}.png` → align → pngs_to_tiff → EDF/fuse → analysis), and branch points.
