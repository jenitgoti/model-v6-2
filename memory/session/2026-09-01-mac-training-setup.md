# Session 2026-09-01 (part 2) — Split_Data audit and Mac training setup

## Dataset

Full measured audit written to `memory/dataset-split-data-audit.md`. Headlines:
6,160 images (not 45,000); `Rectangle_concave` has 680 training polygons
(0.47%); test objects are 0.48-0.76x the linear size of train objects and
2.4x denser per image; no train/val or train/test leakage (verified with a
within-split control — an 8x8 aHash matches 71.8% of train against itself and
is useless, a 16x16 dHash gives 0.0% cross-split); 44 images are
byte-identical between val and test; ~9% of images have overlapping polygons.

Prepared arrays: `Split_Data/instance_npy_512_letterbox_v6` (~9.7 GB).

## Real bug found: the model could not be built

`C3k2(x, ..., name="foo")` creates an internal layer `foo_concat`. The builder
also created explicit skip-connection layers named `semantic_p2_concat`,
`semantic_p1_concat`, `semantic_full_concat`, `instance_p2_concat` and
`instance_p1_concat` — colliding with the C3k2 blocks of the same base name.
Keras 3.15 rejects duplicate operation names, so
`build_model_v6_2_instance()` raised `ValueError` and **the model has never
been constructible in this environment**. Renamed the five explicit layers to
`*_skip_concat`. Model now builds: 460 layers, 6,418,851 parameters.

`run_selftest` now builds the model and asserts no duplicate layer names, so
this cannot regress silently.

## Environment

- `.venv` with `tensorflow==2.21.0` + `keras 3.15.1` (matches the V5 run).
- **`tensorflow-metal` has no Python 3.13 wheel**, so this Mac is CPU-only
  (M3, 8 cores, 16 GB unified). `USE_MIXED_PRECISION` and `REQUIRE_GPU` must
  be off for CPU.
- Paths come from `PCB_DATASET_ROOT` / `PCB_MODEL_OUTPUT_DIR`.
- Self-test passes on real TensorFlow, all 9 checks.

## Also added

`ZOOM_OUT_RANGE = (0.45, 1.00)` at p=0.5, set from the measured train->test
scale ratios rather than by taste.

## Benchmark results (measured, M3 Air 16 GB)

| config | s/image | h/epoch | 300 epochs |
|---|---:|---:|---:|
| CPU batch 1 | **3.03** | 3.9 | 49 d |
| CPU batch 2 | 3.28 | 4.3 | 53 d |
| CPU batch 4 | 4.83 | 6.3 | 79 d |
| Metal GPU batch 2 | **7.54** | 9.8 | 123 d |

**tensorflow-metal is 2.5x SLOWER than the CPU for this model** — many small
convs plus custom losses using top_k/cumsum are its worst case. Do not enable
it. Batching also hurts: 16 GB unified memory is the limit.

Data loading is 0.012 s/image, 0.4% of a step, so the model is the entire
cost and only less compute (resolution or architecture) makes it faster.

To use Metal at all you need Python 3.12 (`.venv-metal`, brew python@3.12)
with `tensorflow==2.18.0` — metal 1.2.0's dylib does not link against TF 2.21,
and TF 2.16.2 pins numpy<2 which breaks opencv/scipy. Kept for reference only.

## Two more real bugs found and fixed

1. **Duplicate layer names** (above) — model was unbuildable.
2. **`add_panel_title` sized its header from the FITTED font scale**, so a
   long title that got shrunk produced a shorter panel and the row could not
   be concatenated (`Mode 0 preview panel shapes disagree: {(544,512,3),
   (545,512,3), (542,512,3)}`). It aborted `train()` before epoch 1, and the
   same code runs in `ValidationPreviewCallback` every 10 epochs, so it would
   also have killed a multi-day run mid-flight. Header is now sized from the
   panel geometry only. `ValidationPreviewCallback` is additionally wrapped in
   `_NonFatalPreviewCallback` — a cosmetic preview must never end a long run.
3. `tensorboard` was missing from the venv (TBNotInstalledError at epoch 1).

## Training is RUNNING

Started 2026-09-01 ~18:40 local, `nohup .venv/bin/python -u train_mac.py`,
log at `runs/train_mac.log`, output `runs/scratch_mac/`.
Scratch (random init), 512 px, batch 1 x 4 accumulation (effective 4, matching
the RTX config), float32, `EPOCHS=120`.

**Why 120 and not 300:** the cosine schedule anneals over `EPOCHS`, so
stopping a 300-epoch run early leaves the LR high and the model un-annealed.
120 completes the whole schedule in ~20 days and still clears the 95-mode
augmentation cycle. Change `m.EPOCHS` in `train_mac.py` to go back to 300
(~50 days).

Verified in a preflight on a 24-image slice: builds, compiles, preflight
validation passes, trains, instance evaluation + checkpoint save, previews,
TensorBoard, BackupAndRestore, and TRAINING_REPORT.md with per-class tables.

## Next step

- Progress: `cat runs/scratch_mac/TRAINING_REPORT.md` (rewritten every epoch),
  milestone snapshots in `runs/scratch_mac/reports/epoch_NNNN.md`.
- Resume after any interruption: rerun `train_mac.py`; BackupAndRestore
  continues from the last epoch.
- Everything is uncommitted. Consider committing before the run ends.

Commit at session start: dbdf607 (working tree uncommitted).

## X.0 resolved — the transfer path was broken, not merely unverified

`ProgressiveTransferAdamW.__init__` set instance attributes before calling
`super().__init__()`. Keras 3's `BaseOptimizer.__setattr__` calls
`_check_super_called()` and raises, so **`RUN_MODE="transfer"` (the committed
default) crashed on optimizer construction** under Keras 3.15 / TF 2.21 — the
exact stack the V5 model was trained on. Fixed: all validation now runs on
local names, nothing touches `self` until after `super().__init__()`.

The freeze mechanism itself is sound. Keras 3.15's TF trainer calls
`optimizer.apply_gradients` (`keras/src/backend/tensorflow/trainer.py:84`), so
the gradient-scaling override is on the call path. Verified empirically:
multiplier 0.0 leaves a transferred group bit-identical while the new head
trains; 1.0 lets it train. That probe is `run_selftest` check 8.

## Launchers

- `train_mac.py`  — CPU, batch 1 x 4, float32, EPOCHS=120. Currently running.
- `train_wsl.py`  — RTX box. `selftest | prepare | transfer | scratch | report`,
  batch 2 x 2, mixed_float16, workers 8. Paths from `PCB_DATASET_ROOT`,
  `PCB_MODEL_OUTPUT_DIR`, `PCB_V5_CHECKPOINT`, `PCB_TRANSFER_OUTPUT_DIR`.
  Refuses `transfer` if the V5 checkpoint is missing rather than silently
  falling back to random initialization.
