# WeakTr VOC Reproduction: Implementation Plan

Status updated 2026-11-15 (Asia/Saigon): Phase 1 completed all 60 epochs; multiscale CAMs/pseudo-labels are generated. The latest supplied log confirms successful Phase-2 continuation from 29 to 59 completed epochs, resume ZIP export and successful notebook output writing. Best epoch-validation mIoU is 70.97%, latest 70.39%. There are 41 epochs remaining; final 100-epoch segmentation evaluation remains pending.

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

2026-11-15 final notebook delivery: prepared default 89->100 training, full multiscale/CRF evaluation, structured raw/CRF mean and per-class scores, a checked final report, and final ZIP export. The minimum-epoch guard rejects old 59-epoch input. The final report rejects incomplete training/evaluations, checkpoint hash mismatches, missing prediction masks and partial validation coverage; final evaluation failures still trigger ZIP preservation. The full suite passes 55 CPU tests (`PYTHONPATH=artifacts/pytest74`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, inherited `WEAKTR_RUNTIME`, `python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp artifacts/pytest-final-session-verified`). Notebook schema and all code cells compile; modified evaluation/helper Python modules compile. This is implementation/static/CPU validation only: the latest reviewed real run remains 59 completed epochs; the 89-epoch input and final raw/CRF scores still require remote evidence. Estimated final training/validation is 3.7 hours plus setup and unmeasured final inference/CRF runtime.

2026-11-15 run review: epochs 29-58 completed, latest checkpoint summary 59/100, new ZIP exported, notebook/HTML conversion succeeded. Best epoch-validation mIoU 70.97% at zero-based epoch 56; latest 70.39% at epoch 58. Last-ten median epoch duration 1203.3 seconds; 41 remaining epochs forecast 13.70 hours before setup/final evaluation. Keep the single-T4 30-epoch chunk: next session reaches 89 completed epochs (~10.03 training/validation hours plus setup/export), following session trains 11 epochs (~3.68 hours plus setup/final evaluation). Controlled GPU recovery equivalence and final metrics are still pending. This review changes only experiment status/forecast, so no new model tests were run.

2026-11-14 capacity adjustment: changed the notebook default from 24 to 30 additional Phase-2 epochs per invocation. The measured 1237 seconds/epoch projects to 10.3 hours for 30 epochs, plus the observed 37.4-minute setup, leaving approximately 64 minutes under 12 hours. Forty epochs would exceed 13.7 hours before setup. From 29 completed epochs, endpoints become 59/89/100; still three additional sessions. Full 100-epoch scheduler/recovery settings remain unchanged. This is a count-based plan, not an elapsed-time guarantee across different GPUs.

2026-11-14 timeout recovery: logged 29 completed Phase-2 epochs, best/latest raw epoch-validation mIoU 69.53%; the 30th was interrupted. The old ZIP export predates Phase 2. Active artifact folder recovery now supports missing export manifests by computing transfer hashes; original data/checkpoints are not substituted. The updated notebook defaults to Phase 2, discovers a unique active artifact folder with Phase-1 snapshots, Phase-2 snapshot and pseudo-labels, and refuses to silently restart from zero if recovery files are missing. Phase-2 invocations are capped at 24 additional epochs, retaining the original 100-epoch scheduler, and final evaluation waits for a completed 100-epoch checkpoint. The full 45-test suite passed with pinned pytest 7.4.4 and inherited `WEAKTR_RUNTIME` (`python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp artifacts/pytest-timeout-recovery-verified`, isolated pytest on `PYTHONPATH`, plugin autoload disabled); a subsequent focused storage suite verifies actual CPU checkpoint-epoch reading. Notebook schema/code compilation pass. Session timing forecast: 1237 seconds/epoch, approximately 24.4 hours for 71 remaining epochs, excluding setup and final evaluation; a 24-epoch chunk is approximately 8.3 hours at that measured speed.

2026-11-13 resume notebook delivery: rebuilt the upload notebook for the completed Phase-1 run, with checkpoint metadata validation, default Phase-1 skip, integrated multiscale CAM evaluation, pseudo-label preservation, optional Phase-2 pilot, and full Phase-2/final-evaluation routes. Existing complete environments are reused automatically. Prior local outputs were backed up outside Git; upload outputs are cleared. Schema/code compilation pass and all 38 CPU tests pass with pinned pytest 7.4.4 and inherited `WEAKTR_RUNTIME` (`python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp artifacts/pytest-resume-notebook-final`, isolated pytest on `PYTHONPATH`, plugin autoload disabled). Tests verify stage ordering never calls Phase 1 in default/Phase-2/evaluation resume routes. Remote execution of the new notebook remains pending.

2026-11-13 preparation test isolation: the new Kaggle log confirms trainer/CRF imports passed in scratch, then 28 tests passed and the environment-repair test failed because it inherited `WEAKTR_RUNTIME` and expected its mock interpreter under `tmp_path`. The test now controls that variable and covers both default and explicit scratch layouts. All 30 tests pass locally with `WEAKTR_RUNTIME=/kaggle/tmp/weaktr-runtime`, pinned pytest 7.4.4 and plugin autoload disabled (`python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp artifacts/pytest-inherited-runtime-verified`). No trainer, installed runtime, checkpoint or resume behavior changed. The existing installed Kaggle environment can be reused with `prepare --skip-install`; final Kaggle preflight rerun remains pending.

2026-11-13 Phase-1 results (user-provided log and manifest): all 60 epochs 0-59 are logged consecutively. Best selection CAM mIoU: 65.83671942199474% at epoch 33, threshold 0.42. Last CAM mIoU: 64.39329263999188%; last classification mAP: 95.26802583583123%. Manifest lists latest/previous/best-classification/best-CAM snapshots, but binary hashes and latest checkpoint epoch remain unverified. Phase-1 schedule is complete in the log. Next steps are best-checkpoint multiscale CAM generation, CRF pseudo-labels, the Phase-2 pilot/full schedule, and final evaluation. The 69.4% upstream CAM reference needs matching postprocessing/protocol before a direct comparison; 74.0% is a separate final validation segmentation target.

2026-11-13 storage failure: the user supplied nbconvert's final `OSError: [Errno 28] No space left on device`. Temporary runtime/data/CAM/probability storage is now separate from durable Kaggle outputs, with migration of known old temporary paths and storage reporting. Checkpoints and pseudo-labels stay in `/kaggle/working`. Pip/uv download caching is disabled for new installs; existing download caches can be purged before recovery. Resume ZIP export no longer creates an extra unpacked payload copy, and failures clean partial ZIPs while preserving the prior export. All 29 CPU tests pass with pinned pytest 7.4.4 (`PYTHONPATH=artifacts/pytest74`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp artifacts/pytest-storage-verified`), including migration, scratch producer/consumer paths, saved-run roundtrip, and simulated disk-full export. Kaggle migration, actual free capacity, checkpoint availability, and completion of the failed run remain pending confirmation. Official training schedules and checkpoint identity are unchanged.

2026-11-13 GPU smoke report: Phase-1 execution reached the AMP gradient check, which failed on its first attempt. Initial loss-scale overflow is a possible cause, not yet confirmed from the remote log. Smoke acceptance now permits at most 20 GradScaler backoffs and requires a finite, unskipped update, with scale/attempt diagnostics. Persistent overflow and nonfinite forward losses remain failures. All 25 CPU tests pass with isolated pytest 7.4.4, including simulated skipped-update recovery and persistent-overflow rejection (`PYTHONPATH=artifacts/pytest74`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp artifacts/pytest-amp-calibration-verified`). Real CUDA rerun and both-stage acceptance remain pending. Trainer settings are unchanged.

2026-11-13 environment recovery: a subsequent user log shows `PermissionError` while launching the existing `weaktr-env/bin/python`, before pip starts. Setup now probes the interpreter rather than checking only that its path exists, recreates an unusable or wrong-version environment with uv, and verifies Python 3.10 before installing packages. All 21 CPU tests pass locally with isolated pytest 7.4.4 (`PYTHONPATH=artifacts/pytest74`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp artifacts/pytest-env-repair-verified`), including a simulated permission-denied interpreter. Actual environment repair on Kaggle and GPU execution remain pending.

2026-11-13 Kaggle setup report: both trainer imports, `mmcv.ops`, and CRF imports passed. The CPU suite then stopped with 9 passes and 11 setup errors because pytest 7.4.4 interpreted the test helper named `setup` as a legacy module hook. Renamed the helper to `make_training_state`; the committed suite plus this fix passes all 20 tests locally using isolated pytest 7.4.4. Verification used a copy at `artifacts/pytest74-validation/test_reproduction.py` containing the committed test module plus only the rename, together with `tests/test_preparation.py`, leaving unrelated working changes intact. Kaggle preparation rerun, real-data preflight in Kaggle, and GPU checks remain pending.

Workflow clarification (2026-11-09): the user expects notebook import or Git clone plus private Kaggle input/saved-run datasets. Setup and checkpoint restoration are automatic; a first-run pilot is optional. Normal training does not depend on manual acceptance/forecast variables. GPU verification and reference-evaluation results remain pending evidence, not assumed achievements. Original schedules and compute limits remain in force.

- [x] Fork connected and reproduction branch pushed.
- [x] Plan, context, and agent guidance saved.
- [x] Candidate isolated environment and Kaggle notebook implemented (GPU compatibility pending).
- [x] Resume fixes and verification tools implemented/CPU-tested (GPU integration acceptance pending).
- [x] Walkthrough imported, paths verified, external copy preserved, and Git exclusions extended.
- [ ] Dataset preflight and GPU compatibility checks passed.
- [ ] Runtime forecast and restart test completed.
- [x] Phase-1 60-epoch training completed according to the supplied log (saved checkpoint bytes still need verification).
- [x] Multiscale CAMs and CRF pseudo-labels generated; subsequent Phase-2 training is logged.
- [ ] Phase 2 completed.
- [ ] Final evaluation and report completed.

| Measurement | Value |
| --- | --- |
| Local focused tests | 20 passed; CPU only, `python -m pytest tests -q --tb=short`, including input/saved-run transfer round trips and interrupted restore |
| Notebook/static validation | Both notebooks validate; every code cell compiles; old outputs cleared |
| Local VOC data validation | All 10,582 train and 1,449 val pairs pass decoding, dimensions, class IDs, split counts and disjointness; Kaggle transfer/preflight pending |
| Measured GPU hours | Not measured |
| Phase-1 CAM mIoU | User log: best selection 65.8367% at epoch 33; final epoch 64.3933%. README reference 69.4% requires matching CAM evaluation/postprocessing |
| Raw validation mIoU | Not measured |
| Final post-CRF validation mIoU | Not measured; target 74.0% |

Runnable preparation/pilot/full-stage commands are in [KAGGLE.md](KAGGLE.md) and `WeakTr_Kaggle.ipynb`. Local focused verification: `python -m pytest tests/test_reproduction.py -q`. Static checks validate both notebooks and compile their code cells. The upstream `docs/Training.md` and `docs/Evaluate.md` remain the protocol references. The Kaggle environment and complete training/evaluation workflow still require measured GPU acceptance; candidate dependency versions must not be described as validated pins.
