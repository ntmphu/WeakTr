# Three short WeakTr weakness experiments

Prepared 2026-11-16. Implementation and CPU checks do not establish GPU compatibility or improvement. The completed reproduction remains 71.450% raw / 72.8654278306% CRF VOC val. These experiments are exploratory additions, separate from the original 60/100 schedules and seed-504 artifacts.

## Kaggle delivery

Import `WeakTr_Experiments_Kaggle.ipynb`, enable Internet and one T4, and attach:

1. The original private VOC/ImageNet input bundle (`weaktr-kaggle-inputs.zip`, or its extracted dataset).
2. The completed original saved run (`weaktr-resume (1).zip`, or the extracted folder containing `run-manifest.json`). It must contain the Phase-1 latest/best snapshots, Phase-2 epoch-99 snapshot, `variant.yml`, and 10,582 pseudo masks.
3. Optionally `artifacts/weaktr-experiments-source.zip` as a small private source dataset. The notebook uses it preferentially and checks every source hash. Otherwise it clones/pulls `reproduction/voc-deit-small` from `ntmphu/WeakTr`.
4. For a later session, the prior `weaktr-experiments.zip`. Automatic restore verifies every file hash and refuses conflicting working files.

One configuration cell controls paths, fixed settings, stage switches, and the number of additional pilot epochs per call. `RESUME_FROM` disambiguates original saved runs; `EXPERIMENT_RESUME` disambiguates pilot exports. Setup handles ZIPs and extracted inputs, verifies original transfer hashes, reuses or repairs the isolated Python 3.10 environment, runs CPU tests, and validates the actual snapshots/data. Baseline artifacts are inputs; installer records go to `experiment-setup`, not into the baseline.

The default Run All performs CAM pilots, frozen clipping diagnostics, and three epochs each for the original-clipping control and residual-clipping treatment. Estimated six-epoch training is approximately 2–2.5 hours based on the original late-epoch timing, plus setup and evaluation. This is an estimate, not a measured pilot runtime. Every stage saves/exports; train recovery is at completed epochs and frozen CAM recovery is per image.

The full uncertainty-training extension is prepared but defaults off to conserve compute. Set `RUN_FULL_UNCERTAINTY=True` to generate all 10,582 raw labels and run two more matched three-epoch arms. This adds a full CAM pass, six training epochs, and evaluations. Start with default diagnostics to measure throughput/label coverage before enabling it. No Colab compute is used.

## Hypotheses and controls

### E1: high-loss rejection also discards difficult correct supervision

The original active loss is `min(loss_i, threshold_i)`, with a detached threshold from batch-averaged local/global losses. Its derivative is zero above threshold. The treatment is `min(loss_i, threshold_i) + alpha * max(loss_i - threshold_i, 0)`, with alpha fixed at 0.1. Its above-threshold derivative is alpha. Both arms retain the original activation condition (mean loss <= 1.2), local patch 120, and batch averaging. The original loss operator is unchanged; the experimental alpha-zero version is checked against its actual values/gradients in CPU tests.

Frozen diagnostic: 64 fixed training images, 480x480 resize, batch size 4; score pseudo-label correctness only after loss rejection is computed. Report correct-label rejection, incorrect-label rejection, accuracy of retained/rejected pixels, and counts per true class. This uses the final model and a fixed resize, so it does not measure the early-training noise distribution. A low correct-label rejection rate weakens the proposed explanation at this checkpoint.

Training: `hard` versus `soft`, both initialized strictly from the same completed Phase-2 model weights. Each starts a fresh SGD optimizer with momentum 0.9, zero weight decay, constant decoder LR 1e-5 and encoder LR 1e-6. The original polynomial schedule and optimizer are intentionally not resumed: this is explicit short fine-tuning. Both receive the same saved CRF pseudo masks, full 10,582-image training set, data order, augmentation seed, and official crop/resize pipeline. Both preserve the original all-pixel mean loss, including ignored positions as zero. Only alpha differs. Dedicated loader RNG and full training RNG/optimizer state are checkpointed.

Primary endpoint: last planned third-epoch full validation mIoU, treatment minus matched control. Also report before/after values, per-class IoUs, precision/recall, clipping fractions, boundary-band and interior IoUs. Do not select whichever pilot epoch scores best. One seed and short fine-tuning support a feasibility result, not a claim about full retraining or significance.

### E2: attention propagation crosses semantic boundaries

The original refined CAM is `F_c = P (CA_c * M_c)`. AAF produces one scalar per head per image, shared across positions/classes, so it cannot independently remove every bad patch connection. The treatment gates P using `max(0, cosine(f_i,f_j))` from the final encoder patch features. Each gated row is rescaled to preserve its original row mass; zero-mass rows fall back to the original. This tests connectivity while limiting changes due solely to amplitude. Feature similarity can also reflect context, so the gate can remove useful connections or retain wrong ones.

Both arms freeze the same best-CAM checkpoint, use image-level class tags, native-resolution scales 1.0/1.2 and horizontal flips, and use background threshold 0.43. Both run model/propagation in FP32 for a controlled comparison. The baseline mirrors the official CAM operations, but FP32 differs from the official AMP generation and no CRF is applied; it is not a replacement measurement of 66.011%/68.189%. The saved original CRF masks are scored separately as a reference, not a matched raw arm.

Development/held-out diagnostic sets contain 100/200 disjoint deterministic IDs from the 1,464-image VOC train list. They are held out from pilot development only: Phase 1/2 already trained on them. There is no claim of unseen-image generalization. No parameter sweep or best-score threshold search is performed. Confirm on a new subset if the development set motivates any setting change.

Primary endpoint: held-out diagnostic raw-CAM mIoU, gated minus original. Supporting endpoints: class precision/recall, spatial boundary/interior scores, and fixed-ID contact sheets. Train-raw mask gains are evidence about pseudo-label generation, not proof of final segmentation gains. A larger downstream retraining is outside this short E2 pilot.

### E3: hard labels lose view-disagreement information

CAM min-max normalization is not probability calibration. The treatment computes hard assignments for each geometrically aligned scale/flip view and compares them with the normalized aggregate assignment. Keep the aggregate label if at least 75% of views agree; otherwise assign ignore 255. With four default views, three votes suffice. All views can agree incorrectly; agreement is a reliability proxy, not a correctness guarantee.

The default pilot reuses the exact E2 baseline views, so no extra model forwards are required. Report accuracy versus retained coverage, per-class retained coverage, and conditional IoU on the same dev/held-out diagnostic images. Never compare selective mIoU alone with complete-mask mIoU: discarding difficult pixels can raise it trivially. A useful filter must improve label reliability while retaining useful foreground supervision.

Optional downstream comparison: regenerate all trainaug baseline raw labels plus disagreement-filtered raw labels. `uncertainty-control` trains on the regenerated RAW baseline, while `uncertainty` trains on filtered versions of those same labels. Both use hard clipping (alpha 0), identical fresh optimization, valid-pixel mean reduction, the same initialization and three epochs. This avoids confounding filtering with removing original CRF pseudo-labels or changing loss scale. The existing CRF-trained initialization may still limit gains from short adaptation. The extension tests raw-label filtering, not a CRF-plus-filter variant.

Full label generation does not read segmentation GT. GT is used only for diagnostic scoring and validation, never training targets or uncertainty/gating decisions. Original WeakTr CAM selection used GT as documented in the reproduction.

## Evaluation, reporting, and persistence

Pilot segmentation evaluation covers all 1,449 validation images at the source variant's resize, single-scale sliding windows 480/320, single decoder, FP32, no CRF. Both arms use the same procedure, and their initial scores must match before the report emits a completed comparison. It differs from the final reproduced multiscale/two-decoder/CRF procedure, so a pilot score must not be subtracted from 72.865% to claim improvement. Only matched complete pilot pairs produce treatment-control deltas. Smoke runs are labeled and excluded from completed comparisons.

Boundary-band IoU is confusion-matrix IoU restricted to a two-pixel neighborhood of GT class transitions, excluding ignore transitions. It is a spatial diagnostic, not the standard Boundary IoU metric. Pixel coverage excludes GT 255. Selective precision, recall, and IoU condition on retained predictions; per-class coverage exposes discarded object pixels. Absent-class metrics are null rather than silently zero.

Outputs under `/kaggle/working/weaktr-experiments`:

- `preflight.json`, `identity.json`, `splits/`, and `records/`: settings, code digest, commit or source revision, checkpoint and annotation hashes, dataset-list hashes, packages/GPU, statuses and measured elapsed times.
- `cams/`: hashed per-image baseline/gated/agreement masks and dev/held-out metrics. `figures/`: fixed held-out image/GT/prediction contact sheets.
- `clipping/scores.json`: correctness/rejection counts, rates, and per-batch activation evidence.
- `training/{arm}/`: independent full checkpoints (latest/previous), per-epoch validation/per-class metrics, initialization scores, final validation masks, and explicit completion receipts. Original checkpoints are never overwritten.
- `experiment-summary.json`, `experiment-report.md`: partial/pending/completed scopes and matched deltas.
- Sibling `weaktr-experiments.zip`: all pilot outputs plus `experiment-manifest.json`; checksums verified on restore. Save Version outputs and download the ZIP before ending the session.

Settings/source changes fail identity checks; use a new output directory for a new protocol. Path relocation and per-session epoch caps are allowed. Checkpoints are trusted local/user-supplied full pickle snapshots, as in the original workflow. A runtime killed before the next export may require attaching the active output folder; exported ZIPs reflect only their export time. Recover interrupted stages before comparing final outcomes.

## CLI and local checks

```powershell
python -m reproduction.experiments.build_notebook
python -m reproduction.experiments.bundle --output artifacts/weaktr-experiments-source.zip
python -m pytest tests/test_experiments.py -q
python -m pytest tests -q
```

This Windows workspace uses isolated pytest 7.4.4. The managed Windows sandbox denied pytest access to newly created temporary directories with both pytest versions; the verified 71-test run used normal filesystem permissions with the following environment:

```powershell
$env:PYTHONPATH='artifacts/pytest74'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests -q --tb=short -p no:cacheprovider --basetemp artifacts/pytest-experiments-verified
```

In the prepared Kaggle interpreter, with `DATASET` and `PYTHONPATH` set as in the notebook:

```bash
python -m reproduction.experiments.run preflight --config /kaggle/working/weaktr-experiment-config.json
python -m reproduction.experiments.run cams --config /kaggle/working/weaktr-experiment-config.json
python -m reproduction.experiments.run clipping --config /kaggle/working/weaktr-experiment-config.json
python -m reproduction.experiments.run train --arm hard --config /kaggle/working/weaktr-experiment-config.json
python -m reproduction.experiments.run train --arm soft --config /kaggle/working/weaktr-experiment-config.json
python -m reproduction.experiments.run report --config /kaggle/working/weaktr-experiment-config.json
python -m reproduction.experiments.run preserve --config /kaggle/working/weaktr-experiment-config.json
```

## Closely related research

- [WeakTr](https://arxiv.org/abs/2304.01184): AAF and clipping are the baseline mechanisms; use original DeiT-S results for the reproduction comparison.
- [MCTformer](https://arxiv.org/abs/2203.02891): multiple class tokens and patch-attention refinement provide the closest architectural context.
- [AFA](https://arxiv.org/abs/2203.02664): explicitly learned semantic affinity motivates investigating attention leakage; this pilot does not reproduce AFA.
- [ToCo](https://arxiv.org/abs/2303.01267): token contrast addresses semantic over-smoothing; this pilot does not implement contrastive retraining.
- [SEAM](https://arxiv.org/abs/2004.04581): equivariance motivates using transformed-view reliability. Agreement filtering is distinct from SEAM's training loss.
- [WeCLIP](https://arxiv.org/abs/2406.11189): stronger frozen vision-language representations and dynamic refinement are useful context; different pretraining prevents attributing its results to one change relative to DeiT-S.

## Acceptance state

- Implementation/local verification: all 71 CPU tests pass; the 19-cell notebook schema/code compiles. Tests cover original-versus-experimental clipping gradients, actual short-trainer recovery with a tiny CPU model, CAM alignment/caching, GT-free full label generation, transfer hashes, and ZIP/extracted source routing. See the latest handoff entry in PROJECT_CONTEXT.md.
- Kaggle experiment imports, operator/model compatibility, runtime, recovery equivalence, and improvement metrics: pending actual remote execution.
- Positive results, negative results, and insufficient retained coverage must all be reported without changing the completed baseline claim.
