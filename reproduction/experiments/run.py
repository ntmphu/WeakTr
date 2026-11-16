"""Run explicit, resumable pilots: python -m reproduction.experiments.run --help."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from reproduction.data import read_ids, sha256, annotation_fingerprint, validate_voc
from reproduction.experiments.core import (ROOT, CLASSES, DEFAULTS, config_identity, guarded_run,
                                          load_config, pilot_splits, preserve, write_json)
from reproduction.experiments.bootstrap import REQUIRED_RUN_FILES


def provenance(cfg):
    import torch
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = ROOT / 'source-revision.json'
        commit = json.loads(revision.read_text()).get('git_commit') if revision.exists() else None
    return dict(config=cfg, semantics=config_identity(cfg), git_commit=commit,
                python=sys.version, platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
                split_sha256={p.name: sha256(p) for p in (ROOT / 'voc12').glob('*_id.txt')})


def preflight(cfg):
    from reproduction.checkpoint import load_checkpoint
    baseline, output = Path(cfg['baseline']), Path(cfg['output'])
    voc = Path(cfg['data']) / 'voc12/VOCdevkit/VOC2012'
    missing = [name for name in REQUIRED_RUN_FILES if not (baseline / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete saved run at {baseline}: missing {missing}. '
                         'Attach/restore the completed 100-epoch reproduction output or final ZIP.')
    checkpoint_hashes = {}
    for name, stage, epoch in [('phase1/checkpoint.pth', 'phase1', 59),
                               ('phase1/checkpoint_best_mIoU.pth', 'phase1', None),
                               ('phase2/checkpoint.pth', 'phase2', 99)]:
        path = baseline / name
        state = load_checkpoint(path)
        if state.get('format_version') != 1 or state.get('stage') != stage:
            raise ValueError(f'Expected full {stage} snapshot: {path}')
        if epoch is not None and state.get('epoch') != epoch:
            raise ValueError(f'Expected completed original schedule in {path}, got epoch {state.get("epoch")}')
        if stage == 'phase2':
            fingerprint = annotation_fingerprint(baseline / 'pseudo', read_ids(ROOT / 'voc12/train_aug_id.txt'))
            if state['config'].get('annotation_sha256') != fingerprint:
                raise ValueError('Saved pseudo-label collection differs from Phase-2 training targets')
        checkpoint_hashes[name] = sha256(path)
        del state
    validation = validate_voc(voc, ROOT / 'voc12/train_aug_id.txt', ROOT / 'voc12/val_id.txt',
                              baseline / 'pseudo', progress=True)
    for local, remote in [('train_aug_id.txt', 'trainaug.txt'), ('val_id.txt', 'val.txt')]:
        if read_ids(ROOT / 'voc12' / local) != read_ids(voc / 'ImageSets/Segmentation' / remote):
            raise ValueError(f'Runtime split differs: {remote}')
    if cfg['clip_patch_size'] != 120:
        raise ValueError('This controlled pilot fixes clipping patch size at official 120 for 480 crops')
    identity = dict(config=config_identity(cfg), checkpoint_sha256=checkpoint_hashes,
                    pseudo_sha256=fingerprint)
    guarded_run(output, identity)
    splits = pilot_splits(cfg)
    for name, ids in splits.items():
        path = output / 'splits' / f'{name}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(ids)+'\n')
    write_json(output / 'preflight.json', dict(status='completed', validation=validation,
               identity=identity, diagnostic_splits=splits,
               note='All CAM diagnostic images were training images; held out only from pilot development, not model training.'))


def summarize(cfg):
    output = Path(cfg['output'])
    result = dict(status='exploratory pilots; pending stages shown explicitly', experiments={})
    lines = ['# WeakTr exploratory experiment report', '',
             'Compare matched arms within each protocol. These pilots do not replace the completed 72.865% reproduction.', '',
             '## Frozen CAM pilots (training images, raw CAMs, no CRF)', '',
             '| Split | Arm | mIoU | Pixel accuracy | Coverage | Boundary-band mIoU |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for split in ('dev', 'holdout'):
        path = output / 'cams' / f'{split}-scores.json'
        if not path.exists():
            result['experiments'][f'cams-{split}'] = 'pending'
            continue
        scores = json.loads(path.read_text())
        result['experiments'][f'cams-{split}'] = scores
        for arm, s in scores['arms'].items():
            vals = [s['miou'], s['pixel_accuracy'], s['coverage_percent'], s['boundary_band']['miou']]
            lines.append(f'| {split} | {arm} | '+ ' | '.join(f'{v:.3f}' if v is not None else 'NA' for v in vals)+' |')
    path = output / 'clipping/scores.json'
    result['experiments']['clipping-diagnostic'] = json.loads(path.read_text()) if path.exists() else 'pending'
    lines += ['', 'Agreement scores are conditional on retained pixels: inspect coverage and per-class coverage. '
              'Saved CRF masks are a separate reference, not a matched raw-CAM arm.', '',
              '## Phase-2 pilot validation', '',
              'Full 1,449-image VOC val, single scale, single decoder, no CRF. Compare final planned epochs, not best epochs.', '',
              '| Arm | Completed epochs | Initial mIoU | Final mIoU | Status |',
              '| --- | ---: | ---: | ---: | --- |']
    from reproduction.experiments.finetune import ARMS
    for arm in ARMS:
        directory = output / 'training' / arm
        initial = json.loads((directory / 'initial.json').read_text()) if (directory / 'initial.json').exists() else None
        epochs = sorted(directory.glob('epoch-*.json'), key=lambda p: int(p.stem.split('-')[-1]))
        latest = json.loads(epochs[-1].read_text()) if epochs else None
        completed = json.loads((directory / 'completed.json').read_text()) if (directory / 'completed.json').exists() else None
        result['experiments'][arm] = dict(initial=initial, latest=latest, completion=completed)
        first = f'{initial["miou"]:.3f}' if initial else 'pending'
        last = f'{latest["scores"]["miou"]:.3f}' if latest else 'pending'
        status = completed['status'] if completed else ('partial' if latest else 'pending')
        lines.append(f'| {arm} | {latest["epoch"] if latest else 0} | {first} | {last} | {status} |')
    lines += ['', '## Matched comparisons', '']
    for control, treatment in [('hard', 'soft'), ('uncertainty-control', 'uncertainty')]:
        left, right = (result['experiments'][a] for a in (control, treatment))
        ready = all(x['completion'] and x['completion']['status'] == 'completed' for x in (left, right))
        if ready:
            if abs(left['initial']['miou'] - right['initial']['miou']) > 1e-6:
                raise ValueError(f'Matched initial model scores differ for {control}/{treatment}')
            delta = right['latest']['scores']['miou'] - left['latest']['scores']['miou']
            result['experiments'][f'{treatment}-minus-{control}'] = delta
            lines.append(f'- {treatment} minus {control}: {delta:+.4f} percentage points.')
        else:
            lines.append(f'- {control}/{treatment}: matched completed comparison pending.')
    lines += ['', 'One seed and short fine-tuning are exploratory. A null result does not establish that a full retraining would fail. '
              'GT supports diagnostics/evaluation only. Do not compare selective CAM mIoU to complete-mask validation mIoU.']
    write_json(output / 'experiment-summary.json', result)
    (output / 'experiment-report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    from reproduction.experiments.cams import make_examples
    make_examples(cfg)
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'cams', 'clipping', 'train', 'uncertainty-labels', 'report', 'preserve'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', choices=['hard', 'soft', 'uncertainty-control', 'uncertainty'])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == 'train' and not args.arm:
        parser.error('train requires --arm')
    output = Path(cfg['output'])
    if args.stage not in ('preflight', 'preserve') and not (output / 'preflight.json').exists():
        raise ValueError('Run preflight first')
    if args.stage != 'preflight' and (output / 'identity.json').exists():
        if json.loads((output / 'identity.json').read_text())['config'] != config_identity(cfg):
            raise ValueError('Experiment settings/code changed; use a new output directory')
    if args.stage == 'preserve':
        preserve(output)
        return
    if args.stage not in ('preflight', 'report'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This stage needs one Kaggle GPU; CPU preparation cannot validate it')
    output.mkdir(parents=True, exist_ok=True)
    record = provenance(cfg)
    record.update(stage=args.stage, arm=args.arm, status='running', started_unix=time.time())
    path = output / 'records' / f'{args.stage}-{args.arm or "all"}-{time.time_ns()}.json'
    write_json(path, record)
    started = time.monotonic()
    try:
        if args.stage == 'preflight':
            preflight(cfg)
        elif args.stage in ('cams', 'uncertainty-labels'):
            from reproduction.experiments.cams import run_cams
            run_cams(cfg, full=args.stage == 'uncertainty-labels')
        elif args.stage == 'clipping':
            from reproduction.experiments.finetune import clipping_diagnostic
            clipping_diagnostic(cfg)
        elif args.stage == 'train':
            from reproduction.experiments.finetune import train_arm
            train_arm(cfg, args.arm)
        elif args.stage == 'report':
            summarize(cfg)
        record['status'] = 'command_completed'
    except BaseException as exc:
        record.update(status='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        write_json(path, record)


if __name__ == '__main__':
    main()
