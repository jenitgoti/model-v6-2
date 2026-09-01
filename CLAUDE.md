# CLAUDE.md — model-v6-2

Single-file project: `model_v6_2.py` (~16.5k lines, 4-class instance segmentation, 512² letterboxed, RTX PRO 1000 8GB, scratch-init only).

## Memory

- `memory/v6-2-uplift-plan.md` — the phased uplift plan (P0 measure → P6 protocol). Read this before proposing model changes; line refs are against commit dbdf607.
- `memory/dataset-split-data-audit.md` — measured audit of `Split_Data` (size, class balance, object-scale shift, leakage, label overlap). Read before interpreting any metric.
- `memory/session/` — one short file per working session. **At the end of each session, add a `YYYY-MM-DD-<slug>.md` entry** with: what was done, files touched, current commit, and next step. Keep each to a few bullets.

## Session log

| Date | Session | Short info |
|------|---------|-----------|
| 2026-08-31 | [uplift-plan-intake](memory/session/2026-08-31-uplift-plan-intake.md) | Ingested audit/uplift plan into `memory/`. No code changes. Repo at dbdf607. |
| 2026-09-01 | [v6-2-uplift-implementation](memory/session/2026-09-01-v6-2-uplift-implementation.md) | Full read of `model_v6_2.py` + v5 evidence. Implemented centre-score ranking, fragment rejection, dihedral + zoom-out augmentation, mAP-based selection, `selftest` mode. Corrected 3 wrong plan items. |
| 2026-09-01 | [mac-training-setup](memory/session/2026-09-01-mac-training-setup.md) | Audited `Split_Data`. Fixed a layer-name collision that made the model unbuildable. Installed TF 2.21 venv (CPU-only, no Metal wheel for py3.13). Benchmarking. |
