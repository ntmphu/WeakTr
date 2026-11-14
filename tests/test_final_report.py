import json

import pytest
import torch
import yaml

from reproduction.data import sha256
from reproduction.finalize import VOC_NAMES, build_summary


@pytest.fixture
def final_run(tmp_path):
    output = tmp_path / 'run'
    (output / 'phase2').mkdir(parents=True)
    checkpoint = output / 'phase2/checkpoint.pth'
    torch.save({'stage': 'phase2', 'format_version': 1, 'epoch': 99}, checkpoint)
    for stage in ('evaluate', 'crf'):
        (output / f'{stage}-record.json').write_text(json.dumps({
            'status': 'command_completed', 'elapsed_seconds': 20,
            'checkpoints': {str(checkpoint.resolve()): sha256(checkpoint)},
        }))
    (output / 'phase2/scores_ms.yml').write_text(yaml.safe_dump({
        'inference': 'multi_scale', 'mean_iou': 72.0, 'cat_iou': [72.0] * 21}))
    (output / 'phase2/scores_ms_crf.json').write_text(json.dumps({
        'mIoU': 73.5, 'count': 2, 'per_class_iou': dict.fromkeys(VOC_NAMES, 73.5)}))
    (output / 'seg_pred_ms').mkdir()
    for name in ('a', 'b'):
        (output / f'seg_pred_ms/{name}.png').write_bytes(b'prediction')
    return output


def test_final_report_contains_measured_scores_and_checkpoint_receipt(final_run):
    summary = build_summary(final_run, ['a', 'b'])
    assert summary['phase2_completed_epochs'] == 100
    assert summary['raw_multiscale_miou'] == 72.0
    assert summary['post_crf_miou'] == 73.5
    assert summary['paper_target_shortfall_pp'] == .5
    assert len(summary['crf_per_class_iou']) == 21
    assert json.loads((final_run / 'final-summary.json').read_text()) == summary
    assert '73.500%' in (final_run / 'final-report.txt').read_text()


@pytest.mark.parametrize('fault', ['incomplete_training', 'failed_evaluation', 'stale_scores',
                                   'missing_mask', 'partial_validation'])
def test_final_report_rejects_incomplete_or_stale_results(final_run, fault):
    if fault == 'incomplete_training':
        torch.save({'stage': 'phase2', 'format_version': 1, 'epoch': 88}, final_run / 'phase2/checkpoint.pth')
    elif fault in ('failed_evaluation', 'stale_scores'):
        path = final_run / 'evaluate-record.json'
        data = json.loads(path.read_text())
        if fault == 'failed_evaluation':
            data['status'] = 'failed_or_interrupted'
        else:
            data['checkpoints'] = {}
        path.write_text(json.dumps(data))
    elif fault == 'missing_mask':
        (final_run / 'seg_pred_ms/b.png').unlink()
    else:
        path = final_run / 'phase2/scores_ms_crf.json'
        data = json.loads(path.read_text())
        data['count'] = 1
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        build_summary(final_run, ['a', 'b'])
    assert not (final_run / 'final-summary.json').exists()
