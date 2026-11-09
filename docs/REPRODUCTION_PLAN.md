# WeakTr VOC Reproduction: Implementation Plan

Status: implementation resumed on 2026-11-09 (Asia/Saigon). CPU recovery/data/IoU/transfer tools, trainer fixes, notebook workflow and walkthrough import are implemented and locally checked. GPU/runtime acceptance remains pending. No training has been launched and no reproduction results have been measured.

## Summary

Reproduce both stages of the original DeiT-S WeakTr pipeline, targeting the repository's reported **74.0% validation mIoU on PASCAL VOC 2012**, within one week. Use the official training and evaluation schedule, one random seed, and ImageNet-pretrained initialization. Authors' trained checkpoints serve only as evaluation references.

Budget: up to **60 user-reported Kaggle GPU hours across separate runtimes**, subject to actual availability. **Colab spending defaults to zero** and requires explicit user authorization. Do not silently shorten training or substitute authors' pseudo-labels if time runs short.

Work in `ntmphu/WeakTr`, branch `reproduction/voc-deit-small`. Push changes to `origin`; use `upstream` to reference the authors' code.

## Project context and repository setup

- Maintain `AGENTS.md` with the objective, remotes, compute constraints, verification commands, and experiment-recording rules.
- Maintain `docs/PROJECT_CONTEXT.md` with baseline identity, upstream commit, current implementation, known issues, dataset protocol, artifact locations, and learning resources.
- Maintain this plan with run commands, a completion checklist, measured budget, and results as implementation progresses.
- Bring the external walkthrough notebook into the repository root and verify repository-relative imports. Preserve its external copy until verification; document the external cache location.
- Extend Git exclusions for model weights, datasets, CAM arrays, generated masks, experiment outputs, and credentials. Preserve small configurations, metrics, selected report figures, upstream attribution, and license.

## Implementation changes

### Runtime and launch workflow

- Provide a Kaggle notebook with separate setup, preflight, Phase-1 training, pseudo-label generation, Phase-2 training, and evaluation sections. Long training requires explicit cell execution.
- Prepare data/packages with the accelerator disabled where possible. Capture Python, PyTorch, CUDA, GPU, and package versions; pin the environment after both stages pass compatibility checks.
- Make dependencies for unused DINOv2/EVA backbones optional so DeiT-S does not require their specialized packages.
- Expose dataset root, output directory, pretrained-weight path, and resume checkpoint as notebook parameters. Start with one GPU and preserve official effective batch sizes.
- Record Git commit, resolved settings, seed, dataset-list hashes, GPU, elapsed time, and checkpoint provenance for every run.

### Training correctness and recovery

- Implement full Phase-1 training resume and repair Phase-2 best-checkpoint restoration.
- Save model, optimizer, scheduler, AMP scaler when applicable, completed epoch, best metrics, configuration, and random-generator states, including any dedicated data-loader generator.
- Write checkpoints atomically; retain latest, previous, and best checkpoints rather than every epoch.
- Resume at epoch boundaries. Reject incompatible training checkpoints clearly. Weights-only author checkpoints remain usable for evaluation and explicitly identified initialization.
- Preserve official architecture, losses, training, evaluation frequency, and checkpoint-selection behavior while making compatibility changes.

### Data and experiment protocol

- Validate **10,582 training IDs** and **1,449 validation IDs**, checking duplicates, overlap, missing images, and annotation dimensions.
- Check class IDs `0–20` and ignore label `255`; verify Phase 2 receives generated pseudo-labels as training annotations.
- Preserve and document official ground-truth-mask use for CAM evaluation/checkpoint selection; do not describe selection as annotation-free.
- Phase 1: 60 epochs, effective batch size 64, input size 224, reduction 8, max pooling, supplied LR `6e-4`, weight decay `0.03`. Record internally scaled LR.
- Generate training CAMs at scales `1.0` and `1.2`, then use the documented CRF procedure and threshold argument `43` (converted by the evaluator to 0.43).
- Phase 2: 100 epochs, batch size 4, SGD, base LR `1e-4`, encoder LR multiplier `0.1`, polynomial scheduling, gradient-clipping settings `1.2` and `120`. Capture resolved image sizes and augmentations.
- Run documented final multiscale evaluation and CRF. Report raw and post-CRF validation scores separately, per-class IoU, and qualitative examples.

### Artifact persistence

- Train and write checkpoints on Kaggle's local working disk; avoid remote storage in the per-batch path.
- Stop sessions at epoch boundaries and preserve artifacts in saved notebook-version outputs. Verify outputs before depending on them in another session.
- Use local downloads as transfer backups. The user's 400 GB Drive can hold an additional archive uploaded from the local machine, avoiding GPU time spent uploading.
- Include checkpoint hashes and a manifest with every transfer. An active runtime's files alone are not durable backups.

## Execution schedule and budget gates

Day numbers are relative to the start of implementation/training, not claims of completed work.

| Period | Work | Gate |
| --- | --- | --- |
| Day 1 | Context, environment, dataset checks, resume fixes | Both stages load data and complete a training step |
| Day 2 | GPU pilot, author-checkpoint evaluation, restart test | Measure runtimes and verify evaluation |
| Days 2–3 | Phase 1 and pseudo-label generation | Valid masks for all training IDs; recorded CAM quality |
| Days 3–6 | Phase 2 across resumable sessions | Full schedule and persistent checkpoints |
| Days 6–7 | Final evaluation, visualizations, report | Reproducible commands, metrics, documented differences |

Reserve **12 of the reported 60 hours** for recovery and final evaluation. Forecast the pipeline using representative training epochs, scheduled evaluation, and a timed CAM/CRF subset before long runs.

If the forecast exceeds available budget, report the shortfall and preserve progress. Further spending or scope changes require a user decision. Exclude full ablations, extra backbones, hyperparameter sweeps, and repeated seeds. Study the walkthrough and inspect attention/CAM examples locally while training runs.

## Verification and acceptance

- Data: split counts/disjointness, valid masks and dimensions, correct pseudo-label annotation directory.
- Compatibility: both stages import, initialize pretrained weights, and complete forward/backward steps in the recorded Kaggle environment.
- Recovery: controlled interruption restores epoch, optimizer, scheduler, scaler, and best metrics; compare with an uninterrupted short run within appropriate numerical tolerance.
- Evaluation: hand-constructed-mask IoU checks including ignored pixels; author-checkpoint evaluation to separate pipeline problems from training differences.
- Artifacts: restore a saved checkpoint in a fresh session, verify its hash, and check complete mask coverage.
- Final: complete both schedules; compare measured validation mIoU against 74.0%, with absolute gap and all deviations. A gap over one percentage point triggers a configuration/evaluation audit before retraining.

The deliverable is a repeatable experiment with honest measured results. Matching the published score is the target, not an assumed outcome.

## Current checklist and results

- [x] Fork connected and reproduction branch pushed.
- [x] Plan, context, and agent guidance saved.
- [x] Candidate isolated environment and Kaggle notebook implemented (GPU compatibility pending).
- [x] Resume fixes and verification tools implemented/CPU-tested (GPU integration acceptance pending).
- [x] Walkthrough imported, paths verified, external copy preserved, and Git exclusions extended.
- [ ] Dataset preflight and GPU compatibility checks passed.
- [ ] Runtime forecast and restart test completed.
- [ ] Phase 1, CAMs, and pseudo-labels completed.
- [ ] Phase 2 completed.
- [ ] Final evaluation and report completed.

| Measurement | Value |
| --- | --- |
| Local focused tests | 11 passed; CPU only, `python -m pytest tests/test_reproduction.py -q --tb=short` |
| Notebook/static validation | Both notebooks validate; every code cell compiles; old outputs cleared |
| Measured GPU hours | Not measured |
| Phase-1 CAM mIoU | Not measured; README reference 69.4% |
| Raw validation mIoU | Not measured |
| Final post-CRF validation mIoU | Not measured; target 74.0% |

Runnable preparation/pilot/full-stage commands are in [KAGGLE.md](KAGGLE.md) and `WeakTr_Kaggle.ipynb`. Local focused verification: `python -m pytest tests/test_reproduction.py -q`. Static checks validate both notebooks and compile their code cells. The upstream `docs/Training.md` and `docs/Evaluate.md` remain the protocol references. The Kaggle environment and complete training/evaluation workflow still require measured GPU acceptance; candidate dependency versions must not be described as validated pins.
