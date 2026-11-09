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

Implementation resumed on 2026-11-09 at the user's request. The reproduction now has shared atomic full-state checkpoints, both trainers' resume integration, single-GPU evaluation repairs, lazy EVA dependencies, VOC validation, annotation fingerprints, experiment records, transfer verification, a staged launcher, GPU smoke/comparison commands, candidate isolated Kaggle setup, and both notebooks in the repository. Default push remote is `origin`.

Focused CPU tests cover interrupted-versus-uninterrupted recovery, failed writes/previous checkpoints, invalid masks/splits, official CAM IoU with ignored/absent classes, and corrupted transfers. Notebook schemas and all code cells compile, and walkthrough paths are verified. These are CPU/static checks; no real dataset preflight, Kaggle compatibility, GPU training, reference evaluation or reproduction metric has been measured. See [KAGGLE.md](KAGGLE.md) for commands and [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md) for pending acceptance.

## Files and learning resources

Workspace: `F:\Courses\CV nâng cao\Project`; repository: its `WeakTr` subdirectory.

- `WeakTr_explained.ipynb`: imported walkthrough with old outputs cleared and repository paths verified. Its external source `../WeakTr_explained.ipynb` remains preserved. Default cache remains the sibling `../weaktr_cache`, overridable with `WEAKTR_CACHE`. Full execution is pending.
- `WeakTr_Kaggle.ipynb`: staged runtime workflow; long jobs are disabled until deliberately executed.
- `../weaktr_cache/`: external cache containing files named `deit_small_patch16_224.pth` and `weaktr_voc_phase1.pth`, plus example images. Names are not provenance verification; validate before use.
- `../WeakTr paper.pdf`: local paper copy.
- `docs/Training.md`, `docs/Evaluate.md`, `docs/Install.md`, `docs/prepare_dataset.md`: upstream instructions.
- `main.py`, `models.py`, `AAF.py`, `engine.py`: Phase 1 and CAM generation.
- `OnlineRetraining/segm/train.py` and `model/decoder.py`: Phase 2 and gradient clipping.

## Inspection findings and implementation status

1. Phase 1 now saves/restores optimizer, scheduler, scaler, best metrics, configuration and RNG, including its dedicated data-loader generator. Legacy weights-only training recovery is rejected; author checkpoints remain evaluation references.
2. Phase 2 restores best metrics from the full latest snapshot, saved after validation. Both stages retain latest/previous/best snapshots with atomic writes. Recovery is restricted to one training GPU and completed epochs; GPU equivalence remains pending.
3. Apex/xformers imports are lazy within EVA code paths. DeiT-S calculations are unchanged. Phase-2 single-GPU metrics no longer require an initialized process group, and the removed NumPy float alias is replaced.
4. Phase 1 scales LR by effective batch size / 512. With the documented batch size and supplied LR, the scaled LR is `7.5e-5`.
5. Phase 2 CLI defaults to evaluation every epoch, overriding the YAML's `eval_freq: 8`. Preserve resolved CLI behavior. YAML VOC dimensions: resize 520, crop/window 480, stride 320.
6. Phase 2's upstream default annotation path is ground truth. The reproduction launcher now explicitly selects generated pseudo-labels, validates them, and fingerprints their contents for recovery.
7. Official Phase-1 commands differ from some paper appendix hyperparameters. Use the official VOC repository commands and record differences.
8. Broad JSON/results exclusions were replaced with explicit artifact/credential exclusions. Small configs/manifests/metrics can be tracked outside generated artifact directories; licenses and attribution are preserved.
9. Final evaluation intentionally defaults to `backbone=multi`, combining two decoder-layer predictions with weight 0.8; the launcher preserves this, separately from the single-output decoder used during epoch validation.

The external ImageNet cache file was hashed during implementation: `cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f`, matching the published DeiT-S filename prefix. The author WeakTr cache's provenance and evaluation remain pending.

Local Python inspection found torch, NumPy, Pillow, pytest, and nbformat installed, but not timm. This does not establish GPU availability or end-to-end compatibility. No Kaggle runtime is connected or verified.

## Dataset and evaluation protocol

Repository lists contain 10,582 training and 1,449 validation IDs. Full image/mask existence and integrity checks remain pending. Semantic masks use 21 classes including background; ignore label 255. Pixel ground truth is used for evaluation and official CAM checkpoint selection, not as Phase-2 training targets. Compare intermediate CAM quality and final segmentation quality separately.

## Next session

Use `docs/KAGGLE.md` and `WeakTr_Kaggle.ipynb` to prepare the isolated candidate environment and real VOC data with the accelerator disabled. Then verify both stages' GPU steps and official batch sizes, controlled recovery, author-checkpoint evaluation and runtime forecast before long training. No Kaggle runtime is currently connected. Preserve the compute boundary and leave all GPU metrics pending until measured.
