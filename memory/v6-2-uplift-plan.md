---
name: v6-2-uplift-plan
description: Model V6.2 instance-seg uplift plan — phased work order (P0 measure → P6 protocol) from the audit artifact
metadata: 
  node_type: memory
  type: project
  originSessionId: b6c391f3-1d5f-4034-96e9-6b10167752ad
  modified: 2026-08-31T17:09:01.649Z
---

Uplift plan for `model_v6_2.py` (16,561 lines, 4-class instance seg, 512² letterboxed, RTX PRO 1000 8GB, batch 2×2 accum, scratch-init only). Line refs are against commit dbdf607. **mAP figures are reasoned estimates, not measurements — Phase 0 replaces them with numbers.**

**STATUS (2026-09-01):** P1.1, P2.2 (corrected), P2.3, P2.4, P3.1, P3.3, P6.1 and part of P6.2 are implemented in `model_v6_2.py`, plus a `selftest` run mode and a label-overlap audit in `prepare`. Three items below (P2.1, P2.2 as originally written, P2.5) were **wrong** and are corrected in place — read those corrections before acting on anything else here.

**Why:** Mask mAP is a product of 6 stages. Stage status: semantic=healthy, centres=healthy, boundary=healthy, offsets=starved (~0.1% of shared-trunk gradient), decoder=fragments (51px assignment cap dumps rims into a fallback that mints new IDs), ranking=saturated (mean semantic prob 0.97–0.99 for real objects and phantoms alike). Decoder + ranking are pure post-processing — fixable against the existing checkpoint, no retraining. Start there.

**How to apply:** Do the phases in order; each step makes the next measurable.

**P0 — measure first (½ day, blocks everything)**
- P0.1 Oracle ablation harness (4h): substitute GT for one head at a time, re-run eval. `decode_instances` takes plain numpy prob arrays; feed it `build_all_targets` output. Row that matters: GT sem+centre+offset should give ~1.0; if not, decoder is broken by construction (see P2.2, `model_v6_2.py:5290`).
- P0.2 Cache val outputs to disk as float16 (1h): ~2.4MB/image, ~1.2GB for 500. Makes 200-config threshold sweeps run in numpy in seconds.

**P1 — free wins vs current checkpoint (1 day, no retrain)**
- P1.1 (highest ROI) Rank instances by centre peak score, not mean semantic prob. `find_center_peaks` returns peak score (`:4416`), `decode_instances` discards it. Thread `score` through `add_partitioned_instances` (~`:4691`) into an `instance_scores` dict; no-centres fallback gets a deliberately low score (0.05). Combine at scoring (`evaluate_model_arrays` ~`:7535`) as `sqrt(centre_score) * confidence` (try plain product too). `decode_instances` now returns 4 values — update call sites `:7512`, `:13477`, `:15902`. Est +3 to +10.
- P1.2 Fit the 5 decoder thresholds (`:275–284`: SEMANTIC_CONFIDENCE_THRESHOLD, CENTER_CONFIDENCE_THRESHOLD, CENTER_NMS_RADIUS, MIN_INSTANCE_AREA, BOUNDARY_CONFIDENCE_THRESHOLD) via random search (200 draws), score mAP50-95 over cached tensors. Split val in half: fit on one, verify on other. Ranges: semantic 0.20–0.60, centre 0.05–0.40, nms_radius 3–9, min_area 8–60, boundary 0.30–0.70. Est +2 to +5.
- P1.3 Dihedral-8 TTA (insert at `:15888` and `:7392`). D4 = 4 rotations × mirror, exact symmetrisation for rectangles/circles. **Trap: offset head is a vector field** — must rotate each (dx,dy) pair, not just un-rotate the raster. `rot90(raster,-1)` sends (dx,dy)→(-dy,dx); mirror flips dx only. Average semantic in logit space, centre/boundary in prob space. Validate inverses by enabling transforms one at a time checking instance F1. 8× inference — final eval/test only. Est +1 to +3.

**P2 — fix training (1 day; P2.1 & P2.2 blocking)**
- ~~P2.1 Scale offset loss to head-pixel units.~~ **WRONG — do not do this.** The claim confuses loss *magnitude* with *gradient* magnitude. `MaskedOffsetPixelHuberLoss` uses `delta = 1/256`, so for any error above 1/256 the loss sits in its linear (L1) regime where `d(loss)/d(prediction) = 1.0` — the same order as the semantic loss. The head is not starved. V5's `training_log.csv` confirms convergence to `offset_loss ≈ 8.5e-4`, which back-solves to ~0.6 head-pixel error. Multiplying by `head_width` would give the offset head 256× everyone else's gradient.
- ~~P2.2 Delete the assignment distance cap.~~ **WRONG as written — the cap is not a size limit.** `nearest_center_assignments` measures the distance from the pixel's **projected** position (`pixel + predicted offset`) to the centre, not from the pixel itself; `xs`/`ys` are passed only for length validation. It is an offset-residual outlier filter and is object-size independent. **The real defect in that block is the fallback**, which dumped every unassigned pixel of a class into one mask and minted a new instance per connected component ≥ `MIN_INSTANCE_AREA` (13 px = a 4×4 blob). **Done differently:** `MIN_COMPONENT_AREA_FRACTION = 0.10` now rejects any component smaller than 10% of the largest one in the same candidate, applied at candidate scope (a per-partition floor cannot work — `boundary_partition` puts a disconnected fragment in a partition of its own). The cap is kept and its comment now says what it measures.
- P2.3 (retrain) Threaded data loading: `InstanceArraySequence.__init__` `:1298` — `super().__init__(workers=8 if training else 2, use_multiprocessing=False, max_queue_size=16)`. cv2/numpy release GIL; threads avoid pickling memmaps. Then crop per-instance loops: `sanitize_semantic_and_instances` (`:742`) and `build_center_and_offset_targets` (`:810`) allocate full 512² bool per instance — use `cv2.connectedComponentsWithStats` bbox and work in the crop. Speed 3–5×.
- P2.4 (retrain) Subsample Lovász sort. `tf.math.top_k(errors, k=all)` at `:1601` = full sort of 524,288 elements ×4/step in-graph on GPU, likely dominant step cost. `foreground_lovasz_softmax_loss` (`:1566`): add `max_pixels=65536`, uniform random sample via `tf.random.shuffle(tf.range(pixel_count))[:keep]`.
- ~~P2.5 Check class-weight cap.~~ **Not an issue for this data.** V5's `training_config.json` records the actually-computed weights: `[1.0, 2.82, 5.97, 4.69, 3.71]`. Nothing reaches the 8.0 cap, so it never binds.

**P3 — augmentation matching data symmetry (1 day, retrain)**
- P3.1 Dihedral symmetry, orthogonal to the 95-mode cycle. No flips/90° rotations anywhere in the file; both lossless for this data. **Do NOT restructure the cycle** — AUGMENTATION_CYCLE_LENGTH=95 is load-bearing (early stopping, epoch schedule, config validator, `:873`). Add separate RNG stream `augmentation_rng_for_sample(sample_index, salt=1)` (SeedSequence), apply rot90/flip in `__getitem__` after `augment_training_sample`, before `build_all_targets` (`:1329`). Mirror caveat: safe for shape classification; drop mirror the day an orientation output is added. Est +2 to +5.
- P3.2 (biggest augmentation lever) Copy-paste augmentation (Ghiasi et al. 2021). Per-pixel instance IDs already stored → extraction free. New `copy_paste(image, semantic, instance, donor, rng, max_objects=3)`, call from `__getitem__` before `build_all_targets`. **Feather the paste edge** (`cv2.GaussianBlur` alpha) so a hard cut never becomes the boundary cue. `sanitize_semantic_and_instances` handles occlusion remnants. Est +3 to +8.
- P3.3 Zoom-out scale aug. `zoom_to_fill_affine_matrix` (`:966`) always ≥1.0 → model never sees objects smaller than labelled. Add zoom-out branch 0.7–1.0, pad with 114 (match letterbox value). Est +1 to +3.

**P4 — replace centre+offset grouping (1–7 days, architecture decision)**
Centre+offset is from Panoptic-DeepLab (2,975 finely-labelled Cityscapes images, crowds/occlusion). This task has neither. It's the most data-hungry head and buys nothing.
- **Run first:** `shape_fit_report` — fit rotated rect + circle to every polygon, report mask IoU per class. 10 min, decides the option.
- **Option C (start here, ~1 day, low risk):** distance transform + watershed. Swap 2-ch offset head for 1 ch of per-instance-normalised inner distance (distance-to-boundary / instance max, [0,1], scale-free). One scalar field << two-vector field. Reuses nearest-core assignment in `boundary_partition` (`:4527`) — substitute distance-transform cores for ridge cores. Deletes 51px cap + fallback by construction.
- **Option B (if C plateaus, ~3–4 days):** StarDist radial distances. Object prob + 32 radial distances to boundary; instances via NMS over object prob. Circles/rectangles are star-convex → lossless. Home domain microscopy (labels as scarce as here). Check Rectangle_concave first: rasterise GT polygons from centroids with 32 rays, need IoU >0.95.
- **Option A (highest ceiling, ~1 week, if fit report >~0.95):** shape-parametric heads. Predict `(cx,cy,r,r_inner)` circles, `(cx,cy,w,h,sinθ,cosθ)` rectangles + notch params for concave class, rasterise analytically. 4–9 regression targets/object — most sample-efficient. Masks analytically perfect → IoU 0.9+.
- **Recommendation:** Do C now, run fit report in parallel. Pass → A; fail → B. Keep the semantic head throughout (healthiest component, gives class assignment free).

**P5 — spend the unlabelled data (1–2 weeks, largest long-run lever)**
DATASET_ROOT points at `.../45000 images/Split_Data`. If most are unlabelled, this beats P1–P4 combined. Neither item violates scratch-init.
- P5.1 Pseudo-labelling: train → predict on unlabelled pool → keep only high-confidence instances → add to training set → retrain, 2–3 rounds. Depends on P1.1 (honest ranking makes "high confidence" mean something). Keep the bar high (wrong pseudo-label worse than none, errors compound). Always keep real labels at fixed ratio. **Never pseudo-label from val/test.**
- P5.2 Self-supervised encoder pretraining — honours the no-ImageNet/no-downloaded-weights docstring (nothing downloaded, encoder still random init, only sees own data). Masked reconstruction: mask ~60% of 16×16 patches, light decoder, **reconstruct the Sobel edge map** not raw pixels (forces structural over photometric; Sobel already in `FixedSobelFeatures`). 256², batch 32, ~20 epochs over 45k images ≈ 12–20h. Load encoder into `build_model_v6_2_instance`, fine-tune. Machinery already exists: `initialize_model_v6_2_from_v5` (shape validation + progressive unfreezing).

**P6 — make the reported number mean something (2 days, needed for YOLO claim)**
- P6.1 Select on the metric you report. Currently checkpoint on foreground mIoU + instance F1@0.50, report mAP50-95. F1@0.50 is indifferent to mask tightness >0.5. Add mAP50-95 to `InstanceF1Checkpoint`, select on it (already computed in `mask_average_precision_report` — plumbing change).
- P6.2 Stop `max` over candidates on one val split. `compare_candidate_models` (`:14751`) picks winner of 3–4 checkpoints on same val split — selection noise can exceed real differences (fitting val, looks like progress). Use k-fold CV or a selection-only holdout that never informs a threshold/checkpoint.
- P6.3 Ensemble the folds: 5 folds → average semantic/centre/boundary logits at inference. With dihedral TTA = 40 forward passes/image (still seconds). Composes with EMA (time dimension) — folds do it in data dimension.
- P6.4 Error bars: bootstrap test set 1,000× → 95% interval. Overlapping intervals with YOLO = not beaten yet.

**Evidence from the V5 run (measured, not estimated)**
- Val: TP 34,898 / **FP 6,244** / FN 1,830 over 864 images. Precision 84.8%, recall 95.0%. **The problem is false positives, not misses** — trade recall for precision, never the reverse.
- GT instance counts: circle_full 24,028 (65%), circle 6,396, Rectangle 6,044, **Rectangle_concave 260 (0.7%)**. Its 54% precision is a labelling-volume problem; no decoder or architecture change fixes ~1.5k training examples.
- `training_log.csv` epoch 356: train fg-mIoU 0.965 vs val 0.826; train loss 0.102 vs val 1.361; **train centre loss 0.032 vs val 1.032 (32×)**. The centre head memorises. That gap, not the offset head, is where the accuracy is.

**Loose ends (optional, ~afternoon each)**
- X.1 Move decoders to half resolution. `semantic_full` + boundary stack run C3k2 at 512×512 (`:3745–3775`) — the memory hog forcing batch 2 + accum. Predict at 1/2 + bilinear upsample → ~4× less activation memory → batch 8, no accum, faster steps. Cost: slightly softer boundaries.
- ~~X.0 Verify `ProgressiveTransferAdamW` is not a silent no-op.~~ **RESOLVED 2026-09-01, and it was worse than a no-op.** `__init__` assigned `self.transfer_variable_groups` and friends *before* `super().__init__()`; Keras 3's `BaseOptimizer.__setattr__` rejects that, so `RUN_MODE="transfer"` — the committed default — crashed on optimizer construction under Keras 3.15. Fixed by running every validation on locals and assigning nothing to `self` until after `super().__init__()`. Keras 3.15's TF trainer *does* call `optimizer.apply_gradients` (`keras/src/backend/tensorflow/trainer.py:84`), so the override is on the call path. Verified empirically both ways (multiplier 0.0 leaves a group bit-identical while the new head trains; 1.0 lets it train) and that probe is now `run_selftest` check 8.
- X.3 Overlapping polygons. `rasterize_instances` writes polygons in file order and a later one overwrites an earlier one in both the semantic and instance maps, so nested or overlapping labels silently hole or erase the earlier instance. `prepare` now counts and reports this; a non-zero count invalidates every metric downstream.
- X.2 Letterbox bars rotate with content. Bars baked in at prepare time then rotated by augmentation → training sees diagonal grey wedges that never occur at inference (always axis-aligned edges). Low impact if source aspect ratios consistent. To fix: letterbox metadata in `{split}_records.json` — crop to content region before augmenting, re-letterbox after.
