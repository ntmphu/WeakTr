# Working on the WeakTr reproduction

Read `docs/PROJECT_CONTEXT.md` and `docs/REPRODUCTION_PLAN.md` before work.

## Objective and boundaries

- Reproduce both stages of original DeiT-S WeakTr on VOC 2012, target 74.0% validation mIoU, with official training/evaluation/checkpoint-selection behavior and one seed.
- Use ImageNet initialization; authors' trained checkpoints are evaluation references. Never silently substitute their pseudo-labels or shorten schedules.
- Default compute is Kaggle only, up to 60 user-reported hours subject to availability. Reserve 12 hours for recovery/evaluation. **Explicit user authorization is required before spending Colab compute units.**
- Train on runtime-local disk, preserve outputs between sessions, and verify transferred checkpoint hashes. Keep GPU work out of CPU-only preparation.
- Do not claim training, compatibility, or metrics are verified unless actually measured. Record pending work honestly.

## Repository workflow

- Work on `reproduction/voc-deit-small`; push to `origin` (`ntmphu/WeakTr`). `upstream` is `hustvl/WeakTr`.
- Preserve upstream attribution and license. Never commit datasets, checkpoints, generated CAM/mask collections, or credentials.
- Keep small configs, experiment summaries, and selected figures reviewable. Check ignore rules before adding files.
- Preserve the external walkthrough copy until the repository copy's paths have been verified.

## Verification and experiment records

- Before changes: `git status --short --branch` and `git remote -v`.
- Before committing: `git diff --check`; run focused tests for checkpoint recovery, data validation, and IoU once implemented. Document exact runnable test commands here as tools are added.
- CPU checks: `python -m pytest tests -q`. GPU acceptance: `python -m reproduction.smoke --pretrained /absolute/path/deit_small_patch16_224.pth`; then both stages' full-size pilots and interrupted recovery comparisons with `python -m reproduction.compare BASELINE/checkpoint.pth RESUMED/checkpoint.pth`. See `docs/KAGGLE.md`.
- GPU acceptance requires both stages' forward/backward checks and an interrupted-versus-uninterrupted recovery check, followed by author-checkpoint evaluation.
- Record Git commit, complete resolved config, seed, dataset-list hashes, environment/GPU, elapsed time, and checkpoint provenance per run.
- Update the plan's checklist/results after verified milestones. Preserve official ground-truth usage for CAM evaluation/checkpoint selection in descriptions.
- User workflow clarification: importing the notebook or cloning code plus attached private input/saved-run datasets is the default. Keep one notebook configuration cell, make the pilot optional, and restore checkpoints automatically. Do not impose manual GPU-acceptance/forecast flags before normal training. Verification evidence remains required before claiming runtime compatibility/results; preserve the compute and official-schedule boundaries.

## Handoff state

Implementation resumed at the user's request on 2026-11-09. Full recovery helpers, trainer integration, optional EVA dependencies, data/transfer checks, records, staged Kaggle notebook, and walkthrough import are implemented. Focused CPU verification passes; Kaggle environment acceptance, dataset preflight on real data, both GPU pilots/recovery checks, reference evaluation and full reproduction runs remain pending. Continue from `docs/PROJECT_CONTEXT.md` and `docs/KAGGLE.md`; do not infer GPU acceptance from CPU tests.

2026-11-13 update: the user reports both GPU smoke checks passed and supplied a complete Phase-1 60-epoch log (best selection CAM mIoU 65.8367%). The upload notebook now defaults to `START_FROM='cams'`, verifies completed Phase-1 snapshots, and continues through integrated CAM evaluation/pseudo-labels and Phase 2; `RUN_SMOKE` and `PHASE2_PILOT` are optional. All 38 CPU tests pass; notebook schema/code compilation pass. Storage migration and real GPU resume equivalence, complete pseudo-labels, Phase 2, and final metrics still need remote evidence. Preserve old notebook outputs in ignored artifacts before clearing the upload notebook.
