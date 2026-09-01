# Session 2026-09-01 — V6.2 examination and uplift implementation

## What was done

Read `model_v6_2.py` end to end plus the `model-v5-instance` GitHub repo
(`training_log.csv`, `training_config.json`, `val_instance_evaluation.json`)
for measured evidence rather than estimates.

**Correction to the uplift plan:** three items were based on misreadings and
are struck through in `memory/v6-2-uplift-plan.md`. P2.1 (scale offset loss by
`head_width`) would have given the offset head 256× the gradient of every
other head — the Huber delta of 1/256 puts it in its linear regime where the
gradient is already 1.0. P2.2 (delete the 51 px cap) misread the cap as an
object-size limit; it measures the *projected* offset residual. P2.5
(class-weight cap) does not bind on this data.

## Changes to `model_v6_2.py` (+1015 / -99)

Decoder and ranking, no retrain needed:
- Centre-peak score threaded through `add_partitioned_instances` into a new
  fourth return value from `decode_instances`; `summarise_instances` now
  ranks by `sqrt(centre score) x mean semantic prob`. Previously every
  instance scored 0.97-0.99, so mask AP had no ranking signal at all.
- `FALLBACK_INSTANCE_SCORE = 0.05` so centre-less recoveries rank last.
- `MIN_COMPONENT_AREA_FRACTION = 0.10` rejects rim fragments at *candidate*
  scope (per-partition does not work: `boundary_partition` isolates a
  disconnected fragment into its own partition).
- `CENTER_NMS_RADIUS` is now unambiguously in instance-head pixels (2 = the
  4 full-res px V5 was tuned to); the silent rescale in `decode_instances`
  is gone.

Training, needs a retrain in a fresh output directory:
- Dihedral stream: 8 exact flips/quarter-turns on its own salted RNG,
  applied before `build_all_targets` so the offset field is regenerated
  rather than rotated. The 95-mode cycle is untouched.
- Zoom-out branch (`ZOOM_OUT_RANGE = (0.70, 1.00)`, p = 0.35 on realistic
  variants). V5 zoom-to-fill is always >= 1.0, so training only ever saw
  objects *larger* than labelled while inference never zooms.
- Warp now pads with `LETTERBOX_FILL_VALUE` instead of `BORDER_REPLICATE`
  (unreachable while zoom-to-fill covers the frame, required for zoom-out).
- Threaded loading (`workers=8`) and `LOVASZ_MAX_PIXELS = 65536`.

Honesty of the reported number:
- `INSTANCE_SELECTION_METRIC = "mask_map50_95"`; both `InstanceF1Checkpoint`
  and `compare_candidate_models` now select on it. History records carry the
  metric name and refuse to resume a best score set under a different one.
- Evaluation targets pass through `sanitize_semantic_and_instances`, matching
  what training actually optimised.
- `prepare` counts overlapping polygons and erased instances and warns.

Verification:
- New `RUN_MODE = "selftest"`: a miniature oracle ablation on synthetic data,
  no dataset/GPU/checkpoint. Feeds ground-truth targets straight into the
  decoder and requires IoU >= 0.95 back, plus checks on ranking, fragment
  rejection, edge-adjacent object separation, `boundary_partition`
  losslessness, all 8 dihedral transforms, zoom-out, and the batch loader.
- Mutation-tested: 8 of 10 seeded defects are caught. Verified here against a
  stubbed TensorFlow (numpy/cv2 only); **it has not been run on the real
  TF 2.21 box.** Run `RUN_MODE = "selftest"` there first.

## Next step

1. `RUN_MODE = "selftest"` on the training box.
2. `RUN_MODE = "prepare"` — read the overlap warning. Non-zero means the
   labels need fixing before any metric is meaningful.
3. `RUN_MODE = "evaluate"` on the existing checkpoint to measure the decoder
   changes alone, before spending a retrain.
4. Then retrain into a **fresh** `MODEL_OUTPUT_DIR` (the augmentation
   distribution changed, so the old run cannot be resumed into).
5. Not done, still open: X.0 (verify the transfer freeze is not a no-op),
   P1.2 (threshold search), P1.3 (dihedral TTA), P4 (`shape_fit_report`,
   then the shape-parametric heads), P5 (the unlabelled pool).

Commit at session start: dbdf607 (uncommitted working tree).
