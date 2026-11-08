# Project context and handoff

Updated: 2026-11-09, Asia/Saigon.

## Goal and decisions

Reproduce **WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation**, original DeiT-S variant, on PASCAL VOC 2012. Target the reported 74.0% validation mIoU. Train both stages from ImageNet initialization, preserving official schedules and checkpoint selection; one seed. Authors' trained checkpoints are evaluation references.

- Paper: https://arxiv.org/abs/2304.01184
- Fork (`origin`): https://github.com/ntmphu/WeakTr
- Authors (`upstream`): https://github.com/hustvl/WeakTr
- Branch: `reproduction/voc-deit-small`
- Inspected upstream commit: `d2f66aa7cff49828d6c1c3771b35af3a1c829097`
- Budget: user reports 30 Kaggle GPU hours on each of two accounts; availability must be checked per runtime. Colab's approximately 300 CU should be preserved; no spending without explicit authorization.
- Timeline: approximately one week. Preserve 12 of the reported 60 Kaggle hours for recovery/final evaluation.
- Fidelity: official evaluation frequency, not reduced-frequency selection.
- Storage: local runtime disk while training, saved Kaggle outputs and local backups between sessions; optional archive on the user's 400 GB Drive.

## Current state

Fork wiring and branch publication are complete. Default push remote is `origin`. Initial inspection is complete. The user asked to save the plan and stop for the night before implementation proceeded.

Only documentation has been added in this handoff. No training code fixes, environment changes, GPU jobs, data downloads, or measured reproduction results are complete. See [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md) for the agreed implementation and checklist.

## Files and learning resources

Workspace: `F:\Courses\CV nâng cao\Project`; repository: its `WeakTr` subdirectory.

- `../WeakTr_explained.ipynb`: existing walkthrough, still outside the repository. Copy to repository root during implementation and verify paths before removing any external copy.
- `../weaktr_cache/`: external cache containing files named `deit_small_patch16_224.pth` and `weaktr_voc_phase1.pth`, plus example images. Names are not provenance verification; validate before use.
- `../WeakTr paper.pdf`: local paper copy.
- `docs/Training.md`, `docs/Evaluate.md`, `docs/Install.md`, `docs/prepare_dataset.md`: upstream instructions.
- `main.py`, `models.py`, `AAF.py`, `engine.py`: Phase 1 and CAM generation.
- `OnlineRetraining/segm/train.py` and `model/decoder.py`: Phase 2 and gradient clipping.

## Findings to address next

1. Phase 1's `--resume` loads weights for CAM generation but does not implement normal full-state training resume. Latest checkpoints currently omit optimizer/scheduler/scaler/RNG state. The dedicated data-loader generator also needs persistence.
2. Phase 2's resume block indexes `checkpoint_best_path` as a dictionary and suppresses the error. Its latest checkpoint is saved before validation; new complete snapshots should capture the completed epoch's selection state.
3. `OnlineRetraining/segm/model/factory.py` imports Apex unconditionally; `model/blocks.py` imports xformers unconditionally. Make optional-backbone dependencies lazy without changing DeiT-S computations.
4. Phase 1 scales LR by effective batch size / 512. With the documented batch size and supplied LR, the scaled LR is `7.5e-5`.
5. Phase 2 CLI defaults to evaluation every epoch, overriding the YAML's `eval_freq: 8`. Preserve resolved CLI behavior. YAML VOC dimensions: resize 520, crop/window 480, stride 320.
6. Phase 2's default annotation path is ground truth. The reproduction launcher must explicitly select generated pseudo-labels and verify this before training.
7. Official Phase-1 commands differ from some paper appendix hyperparameters. Use the official VOC repository commands and record differences.
8. Existing Git exclusions include broad `*.json` and `*results*` patterns. Ensure intended small manifests/metrics can be tracked while excluding heavy artifacts and secrets.

Local Python inspection found torch, NumPy, Pillow, pytest, and nbformat installed, but not timm. This does not establish GPU availability or end-to-end compatibility. No Kaggle runtime is connected or verified.

## Dataset and evaluation protocol

Repository lists contain 10,582 training and 1,449 validation IDs. Full image/mask existence and integrity checks remain pending. Semantic masks use 21 classes including background; ignore label 255. Pixel ground truth is used for evaluation and official CAM checkpoint selection, not as Phase-2 training targets. Compare intermediate CAM quality and final segmentation quality separately.

## Next session

Read agent guidance and the saved plan, inspect the working tree, then implement environment isolation, full checkpoint recovery, data/metric tests, and the staged Kaggle notebook. Verify locally before pushing. Leave GPU results marked pending until measured; do not imply that saving this plan completed the reproduction.
