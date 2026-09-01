---
name: dataset-split-data-audit
description: Measured audit of the Split_Data PCB dataset — size, class balance, object scale shift across splits, leakage and label overlap
metadata:
  type: project
---

Measured 2026-09-01 from `Split_Data` after running `RUN_MODE="prepare"`.
Everything here is counted, not estimated. See [[v6-2-uplift-plan]].

**Size.** 6,160 images (train 4,680 / val 1,048 / test 432), 13 GB source.
Not 45,000 — that was only the v5 folder name. Source images vary from
1024x1536 to 2588x1940, so letterbox bars occupy up to a third of the canvas.

**Class balance (train polygons, 151,828 total, 32.4 per image):**
| class | polygons | share | train pixel freq | resulting weight |
|---|---:|---:|---:|---:|
| circle_full (4) | 76,612 | 52.6% | 3.96% | 4.64 |
| Rectangle (1) | 57,857 | 39.7% | 7.19% | 3.44 |
| circle (3) | 12,000 | 8.2% | 2.29% | 6.10 |
| **Rectangle_concave (2)** | **680** | **0.47%** | 1.42% | 7.74 |
Foreground is 14.9% of pixels. No class reaches the 8.0 weight cap, which
confirms the plan's P2.5 was a non-issue.

**Rectangle_concave is the structural limit.** 680 training polygons, and
they are large (median 9,104 px, ~108 px equivalent diameter). Large and rare
is the hard combination. It scored 54% precision in v5 and will stay the
worst class until it gets more labels. No architecture change substitutes.

**Object scale shifts hard across splits.** Median equivalent diameter:
| class | train | val | test | train->test |
|---|---:|---:|---:|---:|
| Rectangle | 35.7 | 27.7 | 21.9 | 0.61x |
| circle | 33.8 | 33.3 | 16.1 | 0.48x |
| circle_full | 21.0 | 17.5 | 16.0 | 0.76x |
Objects per image also rises: train 32.4, val 46.2, **test 77.5**. The test
split is denser and substantially more zoomed-out than train. **This is why
the zoom-out augmentation matters most here** — V5 zoom-to-fill is always
>= 1.0, so training literally never showed the model a test-scale object.
`ZOOM_OUT_RANGE = (0.45, 1.00)` at p=0.5 is set from these numbers.

**Leakage: the split boundary is clean.**
- Exact (md5) duplicates across train/val and train/test: **none**.
- 44 images are byte-identical between **val and test** (10% of test). Since
  decoder thresholds are tuned on val, that 10% of test is contaminated.
- Within-split exact duplicates: train 24, val 16, test 4.
- Near-duplicate scan: an 8x8 aHash reports 42% train-val matches, but it
  matches **71.8% of train against train itself**, so it is not
  discriminative and that number is an artifact. A 16x16 dHash gives
  **0.0%** train-val and train-test against 20.6% within-train. **Always run
  the within-split control before believing a near-duplicate number.**
- Filenames collide across splits (all `sample_val_NNNNNN`) only because each
  split was numbered from its own counter. It is not duplication.

**Label overlap (a later polygon overwrites an earlier one):**
| split | images with overlap | pixels claimed twice | instances erased |
|---|---:|---:|---:|
| train | 434 / 4,680 (9.3%) | 292,656 | 104 |
| val | 84 / 1,048 (8.0%) | 20,780 | 4 |
| test | 84 / 432 (19.4%) | 15,164 | 16 |
Small per image (~674 px, 0.26% of a 512x512 canvas) so not disqualifying,
but 104 training instances are erased outright and the test split has twice
the overlap rate of train.

**Prepared arrays** live in `Split_Data/instance_npy_512_letterbox_v6`
(~9.7 GB). Regenerate with `PCB_DATASET_ROOT=<...> RUN_MODE="prepare"`.
