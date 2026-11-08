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
- GPU acceptance requires both stages' forward/backward checks and an interrupted-versus-uninterrupted recovery check, followed by author-checkpoint evaluation.
- Record Git commit, complete resolved config, seed, dataset-list hashes, environment/GPU, elapsed time, and checkpoint provenance per run.
- Update the plan's checklist/results after verified milestones. Preserve official ground-truth usage for CAM evaluation/checkpoint selection in descriptions.

## Handoff state

The user paused implementation on 2026-11-09 to resume the following day. Documentation is saved; implementation and GPU runs remain pending. Resume only when requested.
