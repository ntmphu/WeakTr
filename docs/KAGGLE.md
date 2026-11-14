# Kaggle execution and recovery

The common workflow is **import notebook or clone code -> attach private input dataset -> run setup -> optional pilot -> train/evaluate -> save a resumable run**. Code belongs in Git; datasets, ImageNet weights and saved training outputs belong in private Kaggle datasets or notebook outputs. Local work prepares code/notebooks/input bundles and runs CPU checks. GPU/runtime compatibility is claimed only after actually measured in that runtime.

VS Code's remote Jupyter connection is the same execution workflow, not a second implementation. Open the local notebook and choose **Select Kernel -> Select Another Kernel -> Existing Jupyter Server**, then enter the running Kaggle server's URL. The notebook stays on your PC, while code cells and subprocesses execute in Kaggle and use Kaggle's filesystem. Windows files are not implicitly uploaded. The first cell checks that it is running on Kaggle and clones/updates the reproduction code there; attached inputs and saved runs are discovered remotely. Separate local Python edits must be committed/pushed before updating the remote checkout. Do not treat a browser-only VS Code tunnel as a Jupyter server URL. Reference: https://code.visualstudio.com/docs/datascience/jupyter-kernel-management#_existing-jupyter-server

Open `WeakTr_Kaggle.ipynb`. After the reported 12-hour timeout, the upload notebook defaults to `START_FROM = 'phase2'`, `RESUME_FROM = 'auto'`, `RUN_SMOKE = False`, `PHASE2_PILOT = False`, and `PHASE2_EPOCHS_THIS_SESSION = 30`. Attach original inputs and the cancelled run's complete active `weaktr-artifacts` folder (including Phase-1 snapshots, `phase2/checkpoint.pth`, `phase2/variant.yml`, `pseudo/`, logs and records), or a new resume ZIP containing those latest files. The older ZIP logged before Phase 2 started does not contain its updates. Setup automatically selects a unique active artifact folder, computes/verifies transfer hashes if its export manifest is absent, and reuses an existing importable Python 3.10 environment. Metadata checks require completed Phase 1 and an actual Phase-2 snapshot for this resume route. Missing files fail clearly instead of restarting from zero.

Phase-2 training uses at most 30 additional epochs per invocation, retaining the original 100-epoch schedule/scheduler. At the observed 20.6 minutes per epoch, a chunk takes about 10.3 hours plus setup; this cap is not an automatic wall-clock guarantee if hardware/runtime changes. The command exits normally and exports `weaktr-resume.zip`. Final evaluation is skipped until the latest checkpoint confirms all 100 epochs. Download/attach the new ZIP between sessions and repeat. CLI equivalent: `python reproduction/kaggle.py phase2 --epochs-this-session 30`. Use `START_FROM='cams'` to regenerate CAMs from a completed Phase-1 run; a fresh Phase-1 run requires `START_FROM='phase1'`.

For a one-epoch Phase-2 check, set `PHASE2_PILOT = True`; full training and final evaluation are skipped after it. Continue with `PHASE2_PILOT = False` and `START_FROM = 'phase2'`. Use `START_FROM = 'evaluate'` after completing Phase 2. A new runtime requires setup first. `RUN_SMOKE = True` repeats the optional both-model synthetic GPU check. The full schedules, seed, data protocol and compute spending boundaries are unchanged.

## First use

Temporary runtime files now live in `/kaggle/tmp/weaktr-runtime`: the Python environment, copied inputs, ImageNet weights, selection/training CAMs, and final prediction probabilities. Checkpoints, pseudo-labels, configs, metrics, logs, and `weaktr-resume.zip` remain under `/kaggle/working`. Setup/stage launches relocate known temporary paths from the old layout without moving active checkpoints. Scratch files are expendable between sessions; rerun setup from the attached inputs. CAMs must be regenerated if a session ends between CAM generation and pseudo-label creation. The launcher accepts `--scratch` for another temporary location.

On `OSError: [Errno 28] No space left on device`, stop the failed job, run `python -m pip cache purge` in the Kaggle notebook to remove downloaded installer archives, rerun configuration to update code, and rerun preparation. New installs disable pip/uv package caching. Each stage reports saved-output size and filesystem free space; moving files to scratch reduces saved outputs but does not create capacity if scratch shares a full filesystem. Keep completed checkpoints and resume from the latest valid epoch. A run that failed before its first checkpoint must replay that epoch. Do not interpret successful training as proof that notebook conversion/output saving succeeded.

Resume ZIPs are written directly from active artifacts; an additional unpacked export copy is no longer created. Existing old `weaktr-resume/` folders are left intact. Peak export space still includes active checkpoints, the previous ZIP, and the new temporary ZIP; a failed export preserves the previous ZIP and removes its partial file.

1. Upload `data/voc12/weaktr-kaggle-inputs.zip` as a **private Kaggle dataset**, with the ZIP extracted. This prepared bundle contains only required VOC inputs, exact split files, verified ImageNet weights, and a payload-hash manifest. The raw archives and heavy data are never added to Git.
2. Import `WeakTr_Kaggle.ipynb` and attach that private dataset. Enable Internet for code/package installation. The dataset's name does not matter; setup discovers its manifest automatically.
3. Run configuration and setup. Setup copies verified inputs to runtime-local disk and installs an isolated Python 3.10 environment, rebuilding it if an existing interpreter is missing, unusable, or the wrong version. It checks both trainer imports, CRF, focused CPU tests and the real dataset. Run this preparation with the accelerator disabled where possible; the subprocess also hides CUDA during setup.
4. Run the remaining cells with one GPU. Inspect the restored checkpoint summary and multiscale CAM score interactively before the long Phase-2 run. Optionally enable `PHASE2_PILOT` for one real epoch. No full schedule is shortened implicitly.
5. Download `/kaggle/working/weaktr-resume.zip` after training, or preserve it in notebook-version outputs. Attach its extracted contents as a private dataset in the next session, together with the original input dataset.

`RESUME_FROM = 'auto'` prefers checkpoints already on working disk, otherwise discovers one attached saved run, otherwise starts fresh. Set it to `'none'` for a fresh empty run or to a specific saved-run directory when several are attached. Restored payload hashes are checked, and training restores optimizer/scheduler/scaler/RNG/selection state. Existing different working artifacts are preserved with a clear error rather than overwritten.

The candidate environment is Torch 1.13.1+cu117/TorchVision 0.14.1+cu117, binary `mmcv-full==1.7.1` from OpenMMLab's matching wheel index, and `reproduction/requirements-kaggle.txt`. Package installation does not compile CUDA operators. These versions remain candidates until runtime acceptance; preserve the package freeze and resolved pydensecrf source commit. Apex/xformers are loaded only for EVA backbones; DeiT-S computations are unchanged.

The GPU smoke check allows up to 20 Phase-1 AMP calibration attempts: GradScaler can skip early updates and lower its scale after overflow. Each retry clears gradients and recomputes the forward pass on the same synthetic batch. Passing requires finite loss, gradients and parameters plus an unskipped optimizer update; persistent overflow still fails. The record includes attempt count and final scale. This affects only smoke acceptance, not the trainers or their official settings.

The same workflow can be invoked from a cloned repository in Kaggle:

```bash
python reproduction/kaggle.py prepare
python reproduction/kaggle.py smoke         # optional
python reproduction/kaggle.py phase1-pilot  # optional; retains full scheduler
python reproduction/kaggle.py phase1        # starts or resumes full schedule
python reproduction/kaggle.py cams
python reproduction/kaggle.py pseudo
python reproduction/kaggle.py phase2
python reproduction/kaggle.py evaluate
python reproduction/kaggle.py crf
python reproduction/kaggle.py preserve
```

Training commands export a fresh resumable archive on completion and on a catchable interruption when a completed checkpoint exists. After a hard runtime termination, the latest completed checkpoint still needs to be recovered from saved runtime outputs; the export cannot run after the process is killed. The export contains latest/previous/best checkpoints, variant/config files, logs and pseudo-labels, excluding intermediate CAM/probability collections. Phase 2 reuses complete restored pseudo-labels instead of regenerating CAMs. GPU recovery comparisons and author-checkpoint evaluation remain verification tasks; an optional pilot does not silently mark them passed.

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
python -m reproduction.launch preflight --data /kaggle/working/data --output /kaggle/working/weaktr-artifacts --pretrained /kaggle/working/weights/deit_small_patch16_224.pth
python -m reproduction.smoke --pretrained /kaggle/working/weights/deit_small_patch16_224.pth --record /kaggle/working/weaktr-artifacts/gpu-smoke.json
python -m reproduction.launch phase1 --data /kaggle/working/data --output /kaggle/working/weaktr-artifacts --pretrained /kaggle/working/weights/deit_small_patch16_224.pth --stop-after-epoch 1
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

Record checkpoint URLs, hashes, exact commands, measured scores and elapsed costs. The user clarified that pilot checks are optional and normal training should not require manually toggling acceptance/forecast flags. GPU forward/backward checks, full-size batch acceptance, GPU recovery comparisons, and author-checkpoint evaluation remain necessary evidence for declaring the workflow verified. Final reports need raw and post-CRF scores, per-class IoU, selected figures, time consumed and deviations; audit a final gap over one percentage point before considering retraining.
