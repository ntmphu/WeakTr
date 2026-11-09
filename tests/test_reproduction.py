import copy
import random
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from reproduction.checkpoint import atomic_save, load_checkpoint, make_snapshot, restore_training
from reproduction.data import annotation_fingerprint, read_ids, validate_voc
from reproduction.records import transfer_manifest


class ScalerState:
    """Exercise nonempty scaler state without requiring a CUDA context."""
    def __init__(self):
        self.steps = 0

    def state_dict(self):
        return {'steps': self.steps}

    def load_state_dict(self, state):
        self.steps = state['steps']


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.25), torch.nn.Linear(5, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.8)
    return model, optimizer, scheduler, ScalerState(), torch.Generator().manual_seed(seed)


def epoch(model, optimizer, scheduler, scaler, generator):
    data = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 36
    loader = torch.utils.data.DataLoader(data, batch_size=3, shuffle=True, generator=generator)
    model.train()
    for x in loader:
        optimizer.zero_grad()
        target = x.sum(1, keepdim=True) + random.random() + np.random.random()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        scaler.steps += 1
    scheduler.step()
    # Validation consumes RNG too, as the real loaders do.
    return float(torch.rand(1))


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('stage', ['phase1', 'phase2'])
def test_interrupted_matches_uninterrupted(tmp_path, stage):
    config = {'epochs': 4, 'seed': 504, 'batch_size': 3}
    first = setup(504)
    generator = first[4] if stage == 'phase1' else None
    best = {'score': -1, 'epoch': -1}
    for index in range(4):
        score = epoch(*first[:4], generator)
        if score > best['score']:
            best = {'score': score, 'epoch': index}
        if index == 1:
            atomic_save(make_snapshot(stage, *first[:4], index, best, config, generator), tmp_path / 'checkpoint.pth')
    expected_model = copy.deepcopy(first[0].state_dict())
    expected_opt = copy.deepcopy(first[1].state_dict())
    expected_best = best.copy()
    resumed = setup(999)  # model creation and other startup RNG is overwritten by restore
    restored_generator = resumed[4] if stage == 'phase1' else None
    next_epoch, best = restore_training(load_checkpoint(tmp_path / 'checkpoint.pth'),
                                         stage, *resumed[:4], config, restored_generator)
    assert next_epoch == 2
    for index in range(next_epoch, 4):
        score = epoch(*resumed[:4], restored_generator)
        if score > best['score']:
            best = {'score': score, 'epoch': index}
    assert_state_equal(expected_model, resumed[0].state_dict())
    assert_state_equal(expected_opt, resumed[1].state_dict())
    assert_state_equal(first[2].state_dict(), resumed[2].state_dict())
    assert first[3].steps == resumed[3].steps
    assert best == expected_best


def test_resume_rejects_weights_and_changed_config():
    state = setup(1)
    with pytest.raises(ValueError, match='Full recovery'):
        restore_training({'model': state[0].state_dict()}, 'phase1', *state[:4], {})
    snapshot = make_snapshot('phase1', *state[:4], 0, {}, {'epochs': 60}, state[4])
    with pytest.raises(ValueError, match='Incompatible'):
        restore_training(snapshot, 'phase1', *state[:4], {'epochs': 30}, state[4])
    with pytest.raises(ValueError, match='Incompatible'):
        restore_training(snapshot, 'phase2', *state[:4], {'epochs': 60})


def test_atomic_failure_and_previous(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.pth'
    atomic_save({'epoch': 0}, path)
    atomic_save({'epoch': 1}, path, retain_previous=True)
    assert load_checkpoint(tmp_path / 'checkpoint_previous.pth')['epoch'] == 0
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        atomic_save({'epoch': 2}, path, retain_previous=True)
    assert load_checkpoint(path)['epoch'] == 1
    assert not list(tmp_path.glob('*.tmp'))


def voc_fixture(tmp_path):
    for folder in ['JPEGImages', 'SegmentationClassAug', 'pseudo']:
        (tmp_path / folder).mkdir()
    for name in ['train', 'val']:
        Image.new('RGB', (3, 2)).save(tmp_path / 'JPEGImages' / (name + '.jpg'))
        mask = Image.fromarray(np.array([[0, 1, 20], [255, 0, 2]], dtype=np.uint8))
        mask.save(tmp_path / 'SegmentationClassAug' / (name + '.png'))
        mask.save(tmp_path / 'pseudo' / (name + '.png'))
    train_list, val_list = tmp_path / 'train.txt', tmp_path / 'val.txt'
    train_list.write_text('train\n')
    val_list.write_text('val\n')
    return train_list, val_list


def test_voc_validation(tmp_path):
    lists = voc_fixture(tmp_path)
    result = validate_voc(tmp_path, *lists, tmp_path / 'pseudo', expected=(1, 1))
    assert result['train_count'] == result['val_count'] == 1
    with pytest.raises(ValueError, match='ground-truth'):
        validate_voc(tmp_path, *lists, tmp_path / 'SegmentationClassAug', expected=(1, 1))
    Image.new('L', (3, 2), 21).save(tmp_path / 'pseudo/train.png')
    with pytest.raises(ValueError, match='class IDs'):
        validate_voc(tmp_path, *lists, tmp_path / 'pseudo', expected=(1, 1))
    Image.new('L', (1, 1)).save(tmp_path / 'pseudo/train.png')
    with pytest.raises(ValueError, match='dimensions'):
        validate_voc(tmp_path, *lists, tmp_path / 'pseudo', expected=(1, 1))
    (tmp_path / 'pseudo/train.png').unlink()
    with pytest.raises(FileNotFoundError):
        validate_voc(tmp_path, *lists, tmp_path / 'pseudo', expected=(1, 1))


def test_split_validation(tmp_path):
    lists = voc_fixture(tmp_path)
    lists[0].write_text('train\ntrain\n')
    with pytest.raises(ValueError, match='Duplicate'):
        read_ids(lists[0])
    lists[0].write_text('train\n')
    with pytest.raises(ValueError, match='Expected'):
        read_ids(lists[0], 10582)
    lists[1].write_text('train\n')
    with pytest.raises(ValueError, match='overlap'):
        validate_voc(tmp_path, *lists, expected=(1, 1))


def test_official_cam_iou_ignore_and_absent_classes(tmp_path):
    from evaluation import do_python_eval
    gt, pred = tmp_path / 'gt', tmp_path / 'pred'
    gt.mkdir()
    pred.mkdir()
    Image.fromarray(np.array([[0, 0, 1], [1, 255, 255]], dtype=np.uint8)).save(gt / 'a.png')
    Image.fromarray(np.array([[0, 1, 1], [1, 20, 0]], dtype=np.uint8)).save(pred / 'a.png')
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = do_python_eval(pred, gt, ['a'], n_jobs=1)
    assert scores['background'] == pytest.approx(50)
    assert scores['aeroplane'] == pytest.approx(200 / 3)
    assert np.isnan(scores['tvmonitor'])  # prediction on ignored GT must not count
    assert scores['mIoU'] == pytest.approx((50 + 200 / 3) / 2)


def test_transfer_detects_corruption(tmp_path):
    atomic_save({'epoch': 0}, tmp_path / 'checkpoint.pth')
    transfer_manifest(tmp_path)
    transfer_manifest(tmp_path, verify=True)
    (tmp_path / 'checkpoint.pth').write_bytes(b'incomplete transfer')
    with pytest.raises(ValueError, match='mismatch'):
        transfer_manifest(tmp_path, verify=True)


def test_annotation_identity_survives_relocation_but_not_changes(tmp_path):
    voc_fixture(tmp_path)
    original = annotation_fingerprint(tmp_path / 'pseudo', ['train'])
    assert original == annotation_fingerprint(tmp_path / 'SegmentationClassAug', ['train'])
    Image.new('L', (3, 2), 1).save(tmp_path / 'pseudo/train.png')
    assert original != annotation_fingerprint(tmp_path / 'pseudo', ['train'])


def test_official_launcher_settings(tmp_path):
    from reproduction.launch import commands
    phase1 = commands('phase1', tmp_path / 'data', tmp_path / 'output', 'imagenet.pth')[0][0]
    phase2 = commands('phase2', tmp_path / 'data', tmp_path / 'output', 'imagenet.pth')[0][0]
    def value(command, option):
        return command[command.index(option) + 1]
    assert value(phase1, '--epochs') == '60'
    assert value(phase1, '--batch-size') == '64'
    assert float(value(phase1, '--lr')) * 64 / 512 == pytest.approx(7.5e-5)
    assert value(phase2, '--epochs') == '100'
    assert value(phase2, '--batch-size') == '4'
    assert value(phase2, '--eval-freq') == '1'
    assert Path(value(phase2, '--ann-dir')) == tmp_path / 'output/pseudo'
    assert value(phase2, '--pretrained') == 'imagenet.pth'
    assert value(phase2, '--start-value') == '1.2'
    assert value(phase2, '--patch-size') == '120'
    evaluation = commands('evaluate', tmp_path / 'data', tmp_path / 'output', 'imagenet.pth')[0][0]
    assert value(evaluation, '--backbone') == 'multi'
    assert value(evaluation, '--weight') == '0.8'
    assert str(tmp_path / 'output/phase2/checkpoint.pth') in evaluation


def test_recovery_comparison_detects_state_divergence():
    from reproduction.compare import compare
    compare({'model': torch.tensor([1.0]), 'best': .5}, {'model': torch.tensor([1.0]), 'best': .5})
    with pytest.raises(AssertionError):
        compare({'optimizer': torch.tensor([.1])}, {'optimizer': torch.tensor([.2])})
    with pytest.raises(AssertionError):
        compare({'epoch': 2}, {'epoch': 1})
