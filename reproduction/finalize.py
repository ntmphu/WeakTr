"""Collect final metrics only after complete training and both evaluation stages."""
import argparse
import json
import math
from pathlib import Path

import yaml

from reproduction.checkpoint import load_checkpoint
from reproduction.data import read_ids, sha256

ROOT = Path(__file__).resolve().parents[1]
VOC_NAMES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
             'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def build_summary(output, validation_ids=None):
    output = Path(output)
    ids = validation_ids if validation_ids is not None else read_ids(ROOT / 'voc12/val_id.txt', 1449)
    checkpoint_path = output / 'phase2/checkpoint.pth'
    snapshot = load_checkpoint(checkpoint_path)
    if snapshot.get('stage') != 'phase2' or snapshot.get('format_version') != 1 or snapshot.get('epoch') != 99:
        raise ValueError('Final reporting requires a full 100-epoch Phase-2 checkpoint.')
    del snapshot
    checkpoint_hash = sha256(checkpoint_path)
    records = {}
    for stage in ('evaluate', 'crf'):
        records[stage] = json.loads((output / f'{stage}-record.json').read_text())
        if records[stage].get('status') != 'command_completed':
            raise ValueError(f'Final reporting requires completed {stage}.')
        if records[stage].get('checkpoints', {}).get(str(checkpoint_path.resolve())) != checkpoint_hash:
            raise ValueError(f'{stage} scores were not recorded for the current final checkpoint.')
    raw_path = output / 'phase2/scores_ms.yml'
    crf_path = output / 'phase2/scores_ms_crf.json'
    raw = yaml.safe_load(raw_path.read_text())
    crf = json.loads(crf_path.read_text())
    if raw.get('inference') != 'multi_scale' or len(raw.get('cat_iou', [])) != 21:
        raise ValueError('Expected final multiscale scores for all 21 VOC classes.')
    if crf.get('count') != len(ids) or set(crf.get('per_class_iou', {})) != set(VOC_NAMES):
        raise ValueError('CRF evaluation does not cover the complete validation set/classes.')
    missing = [name for name in ids if not (output / 'seg_pred_ms' / (name + '.png')).is_file()]
    if missing:
        raise ValueError(f'Missing final CRF prediction masks: {missing[:3]}')
    raw_miou, crf_miou = float(raw['mean_iou']), float(crf['mIoU'])
    if not all(math.isfinite(value) and 0 <= value <= 100 for value in (raw_miou, crf_miou)):
        raise ValueError('Invalid final mIoU values.')
    summary = dict(
        status='training_and_final_evaluation_completed', phase2_completed_epochs=100,
        validation_images=len(ids), raw_multiscale_miou=raw_miou, post_crf_miou=crf_miou,
        paper_target_miou=74.0, paper_target_shortfall_pp=74.0 - crf_miou,
        raw_per_class_iou=dict(zip(VOC_NAMES, raw['cat_iou'])), crf_per_class_iou=crf['per_class_iou'],
        checkpoint_sha256=checkpoint_hash,
        score_file_sha256={str(p.relative_to(output)): sha256(p) for p in (raw_path, crf_path)},
        stage_seconds={stage: record.get('elapsed_seconds') for stage, record in records.items()},
        inference=dict(backbone='multi', decoder_weight=0.8, postprocessing='CRF'),
        note='Scores are measured outputs; completion does not imply the paper target is matched.',
    )
    (output / 'final-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = (f'# WeakTr final VOC 2012 results\n\n'
              f'Phase 2: 100/100 epochs; validation images: {len(ids)}.\n\n'
              f'Raw multiscale mIoU: {raw_miou:.3f}%.\n\n'
              f'Post-CRF mIoU: {crf_miou:.3f}%.\n\n'
              f'Paper target: 74.0%; shortfall: {74.0 - crf_miou:.3f} percentage points.\n\n'
              f'Checkpoint SHA256: {summary["checkpoint_sha256"]}\n\n'
              'Per-class IoUs and score hashes are in final-summary.json. '
              'Prediction masks are in seg_pred_ms/; stage commands/environment are in run records.\n')
    (output / 'final-report.txt').write_text(report)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build_summary(args.output), indent=2))


if __name__ == '__main__':
    main()
