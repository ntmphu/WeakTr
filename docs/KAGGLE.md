# Kaggle execution and recovery

Implementation is ready for runtime acceptance, which is **pending**. Local verification uses Python 3.12, PyTorch 2.3.1+cpu and NumPy 1.26.4. It does not establish Kaggle/CUDA compatibility. No datasets were downloaded, training launched, or reproduction metrics measured during implementation.

Open `WeakTr_Kaggle.ipynb`. Its setup/preflight/training/evaluation cells have independent switches, initially disabled. Clone this branch into `/kaggle/working/WeakTr`; attach VOC data and the official ImageNet DeiT-S checkpoint. Prepare with the accelerator disabled. The notebook creates an isolated Python 3.10 environment with candidate Torch 1.13.1+cu117/TorchVision 0.14.1+cu117 and `reproduction/requirements-kaggle.txt`. These versions are candidates until both stages pass; preserve the resolved package freeze, including the pydensecrf source commit. The initial CPU mmcv package choice still needs dataset import/operator acceptance. Apex/xformers are loaded only for EVA backbones; DeiT-S computations are unchanged.

The walkthrough is now `WeakTr_explained.ipynb`; its educational reimplementation is separate from training acceptance. Its cache defaults to the external sibling `../weaktr_cache`, overridable with `WEAKTR_CACHE`. The external notebook remains preserved. Repository-relative `models.py`, `voc12/cls_labels.npy`, `img/` and dataset candidate paths have been checked; full notebook execution remains pending.

## Dataset layout

Set `DATA` to the parent directory of `voc12`, with this layout on runtime-local disk:

```text
data/voc12/VOCdevkit/VOC2012/
  JPEGImages/
  SegmentationClass/
  SegmentationClassAug/
  ImageSets/Segmentation/trainaug.txt
  ImageSets/Segmentation/val.txt
```

Follow upstream `docs/prepare_dataset.md`; obtain the data through the supported dataset source. Preflight checks counts, duplicates, split overlap, exact runtime-list order, image/mask decoding and dimensions, and labels 0–20/255. Phase 2 validates generated pseudo-labels and refuses the standard GT directories. Its recovery config fingerprints training annotations, so moving the directory is allowed but changing its masks rejects recovery. Do not rename GT as pseudo-labels. Ground truth is used for official CAM selection/evaluation and Phase-2 validation.

## Commands

Run from repository root, using the isolated environment's Python. Substitute absolute runtime paths for the example paths. `--dry-run` prints argv and working directories without loading data or starting training.

```bash
python -m pytest tests/test_reproduction.py -q
python -m reproduction.launch preflight --data /kaggle/working/data --output /kaggle/working/weaktr-artifacts --pretrained /kaggle/input/weaktr-inputs/deit_small_patch16_224.pth
python -m reproduction.smoke --pretrained /kaggle/input/weaktr-inputs/deit_small_patch16_224.pth --record /kaggle/working/weaktr-artifacts/gpu-smoke.json
python -m reproduction.launch phase1 --data /kaggle/working/data --output /kaggle/working/weaktr-artifacts --pretrained /kaggle/input/weaktr-inputs/deit_small_patch16_224.pth --stop-after-epoch 1
```

All launcher stages accept `--data`, `--output`, `--pretrained`, `--seed` (504) and optional `--resume`. Invoke `phase1`, `cams`, `pseudo`, `phase2`, `evaluate`, `crf` separately in that order. `--pretrained` must hash to the published official DeiT-S SHA256 prefix `cd65a155`; it cannot silently substitute author WeakTr weights. Full hashes, dataset-list hashes, Git revision, resolved command/configuration, versions, GPU, session times and logs are preserved under output directories. Pin the accepted environment after checks pass.

Phase 1 uses 60 epochs, batch 64, reduction 8, max pooling, supplied LR 0.0006 (internally 0.000075), weight decay 0.03, and official per-epoch classification/CAM selection. CAMs use the train-augmented split and scales 1.0/1.2; CRF pseudo-label generation uses threshold 43, converted to 0.43 by the upstream evaluator. Phase 2 uses 100 epochs, batch 4, SGD/poly, LR 0.0001, encoder multiplier 0.1, GC 1.2/120, resize 520, crop/window 480, stride 320, no AMP, and evaluation every epoch (the CLI's official behavior overriding YAML frequency 8). The seed is now explicit in both stages; the upstream Phase-2 CLI had no seed option. No architecture, loss or schedule substitution is intended.

`evaluate` uses the upstream final latest checkpoint and multiscale inference, preserving its default `--backbone multi --weight 0.8`: the two decoder-layer predictions are combined as `0.8 * first + second`. This evaluation behavior differs from the single-output decoder used during training/epoch validation and must be preserved. Raw per-class scores are written to `phase2/scores_ms.yml`. `crf` logs post-CRF per-class scores and saves PNGs separately. Preserve both outputs for the report; 74.0% remains an unmeasured target.

## Recovery acceptance and persistence

Full recovery is supported for a single training GPU, at completed epoch boundaries. Legacy/author weights-only checkpoints are usable for evaluation and explicit ImageNet initialization; training resume rejects them. Legacy per-epoch checkpoint caching is disabled because it bypassed RNG/scheduler recovery. Latest, previous and best snapshots include model/optimizer/scheduler/scaler, completed epoch, best metrics, config and Python/NumPy/Torch/CUDA RNG, plus Phase 1's dedicated loader generator. Latest is saved after validation/selection. Atomic writes protect existing snapshots on serialization failure. A crash during an epoch replays it; log lines may repeat during that replay, so use checkpoint epochs as authority.

For each stage on the accepted GPU/environment, use a separate **pilot output** and compare:

1. Run through two completed epochs with `--stop-after-epoch 2`, preserving the original total schedule.
2. In a second output, run through epoch one with `--stop-after-epoch 1`; restart the Python process and resume that snapshot through epoch two.
3. Run `python -m reproduction.compare BASELINE/checkpoint.pth RESUMED/checkpoint.pth`. Default model/state tolerances are rtol=1e-5, atol=1e-7; RNG must match exactly. Record results and investigate any divergence rather than widening tolerances silently.

For Phase 1 pass the full path to `--resume`. Phase 2 expects the restored directory's `phase2/checkpoint.pth`; it also needs the identical generated `pseudo/` contents. Its `variant.yml` must stay beside model weights for evaluation. Single-GPU loader workers are recreated each epoch, with Torch's worker seeding and global RNG recovered; the CPU regression tests use zero workers, so GPU worker recovery still needs acceptance. cuDNN benchmarking in Phase 2 retains upstream behavior; numerical equivalence must be measured.

Before ending a runtime:

```bash
python -m reproduction.preflight --transfer-dir /kaggle/working/weaktr-artifacts
# After restoring the complete directory in a fresh runtime:
python -m reproduction.preflight --transfer-dir /kaggle/working/weaktr-artifacts --verify-transfer
```

Save the complete output tree, config/variant files, environment freeze, logs and manifest in Kaggle notebook-version outputs, then download a local backup. Verify the saved version and hashes before relying on it. Running runtime files alone are not durable. Optional Drive archiving can run from the local machine. Dataset/CAM/mask collections and checkpoints remain excluded from Git.

## Budget and references

Check actual account/runtime allocation. Budget remains at most the user-reported 60 Kaggle hours, with 12 reserved. Measure representative epochs including scheduled evaluation, CAM/CRF throughput and final evaluation costs before committing the remaining schedule. Forecast all remaining work; if it exceeds available hours after reserve, report the shortfall. No shortened schedules or author pseudo-label substitution. Colab spending remains unauthorized.

Author reference evaluation is still pending. For Phase 1, use the checkpoint linked in upstream README, separate outputs, `main.py --gen_attention_maps --resume AUTHOR_CHECKPOINT`, `--data-set VOC12MS --img-ms-list voc12/train_id.txt --scales 1.0 --reduction 8 --pool-type max`; evaluate with `evaluation.py --list voc12/train_id.txt --type npy --curve True --start 40 --end 60`, the matching `--data-path` and `--predict_dir`. Do not pass these reference masks to Phase 2.

For Phase 2, download its matching variant file beside the author checkpoint. From `OnlineRetraining`, with `DATASET` set to the absolute data directory:

```bash
python -m segm.eval.miou --backbone multi --weight 0.8 --multiscale --predict-dir /kaggle/working/author-reference/seg_prob_ms /kaggle/working/author-reference/checkpoint.pth pascal_voc
```

Record checkpoint URLs, hashes, exact commands, measured scores and elapsed costs. Both GPU forward/backward checks, full-size batch acceptance, GPU recovery comparisons, and author-checkpoint evaluation must pass before long production runs. Final reports need raw and post-CRF scores, per-class IoU, selected figures, time consumed and deviations; audit a final gap over one percentage point before considering retraining.
