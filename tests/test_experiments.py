"""Mathematics, pilot recovery, transfer integrity, and notebook routing (CPU only)."""
import ast
import copy
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from reproduction.checkpoint import load_checkpoint
from reproduction.experiments import core, finetune
from reproduction.experiments.bootstrap import unpack_manifest_archive
from reproduction.experiments.methods import (ResidualClipping, gate_affinity, agreement_mask,
                                             cam_labels, normalize_cams)


def official_clipping():
    path = core.ROOT / 'OnlineRetraining/segm/model/decoder.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GradientClipping')
    namespace = dict(nn=torch.nn, torch=torch, F=F)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['GradientClipping'](1.2, 2)


@pytest.mark.parametrize('active', [True, False])
def test_hard_arm_matches_upstream_values_and_gradients(active):
    torch.manual_seed(6)
    pred = (torch.randn(2, 3, 4, 4)*.2).requires_grad_()
    target = torch.randint(0, 3, (2, 4, 4))
    target[0, 0, 0] = 255
    if not active:
        pred = (pred.detach()*15).requires_grad_()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    official = official_clipping()(pred, target, criterion)[1]
    ours = ResidualClipping(0, 1.2, 2)(pred, target, criterion)[1]
    torch.testing.assert_close(ours, official, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(ours.sum(), pred)[0],
                               torch.autograd.grad(official.sum(), pred)[0], rtol=0, atol=0)


def test_residual_gradient_is_alpha_on_rejected_pixels():
    # Custom per-pixel loss makes the expected derivatives unambiguous.
    losses = torch.tensor([[[.1, .2], [.3, 1.]]], requires_grad=True)
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    _, result = ResidualClipping(.1, 1.2, 2)(losses, target, lambda p, t: p)
    result.sum().backward()
    torch.testing.assert_close(losses.grad, torch.tensor([[[1., 1.], [1., .1]]]))


def test_gate_preserves_row_mass_and_falls_back_for_zero_features():
    attention = torch.tensor([[[.1, .3], [.2, .4]]])
    features = torch.tensor([[[1., 0.], [-1., 0.]]])
    gated = gate_affinity(attention, features)
    torch.testing.assert_close(gated.sum(-1), attention.sum(-1))
    assert gated[0, 0, 1] == 0 and gated[0, 1, 0] == 0
    torch.testing.assert_close(gate_affinity(attention, torch.zeros_like(features)), attention)


def test_agreement_ignores_disagreement_and_preserves_class_indices():
    a = torch.tensor([[[0., 1.]], [[1., 0.]]])
    b = a.flip(-1)
    masked, agreement = agreement_mask([a, a, a, b], 3*a+b, .43, .75)
    assert masked.tolist() == [[2, 1]]
    assert agreement.tolist() == [[.75, .75]]
    masked, _ = agreement_mask([a, a, b, b], 3*a+b, .43, .75)
    assert masked.tolist() == [[255, 255]]
    assert cam_labels(normalize_cams(torch.zeros(2, 1, 2)), .43).tolist() == [[0, 0]]


def test_cam_view_alignment_and_feature_hook_on_cpu():
    from reproduction.experiments.cams import predict_views
    class ToyCam(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv2d(3, 20, 1, bias=False)
            torch.nn.init.constant_(self.head.weight, .1)
        def forward(self, x, **kwargs):
            features = F.avg_pool2d(x, 16)
            seeds = self.head(features).relu()
            n = seeds.shape[-1]*seeds.shape[-2]
            return torch.zeros(1, 20), seeds, torch.eye(n)[None, None]
    model = ToyCam().eval()
    array = np.zeros((32, 32, 3), dtype='uint8')
    array[:, :16] = 255
    present = np.zeros(20)
    present[0] = 1
    original, gated = predict_views(model, Image.fromarray(array), present, [1.])
    assert len(original) == 2
    torch.testing.assert_close(original[0], original[1])
    torch.testing.assert_close(original[0], gated[0])
    assert original[0][1:].count_nonzero() == 0
    assert not model.head._forward_pre_hooks  # Temporary hook always removed.
    original, gated = predict_views(model, Image.fromarray(array), present, [1.], compute_gate=False)
    assert len(original) == 2 and not gated


def test_frozen_cam_pipeline_cache_and_gt_free_full_generation(tmp_path, monkeypatch):
    from reproduction.experiments import cams
    root = tmp_path / 'repository'
    (root / 'voc12').mkdir(parents=True)
    names = ['a', 'b', 'c', 'd']
    np.save(root / 'voc12/cls_labels.npy', {name: np.array([1]+[0]*19) for name in names})
    (root / 'voc12/train_aug_id.txt').write_text('\n'.join(names)+'\n')
    monkeypatch.setattr(cams, 'ROOT', root)
    monkeypatch.setattr(cams, 'pilot_splits', lambda cfg: dict(dev=names[:2], holdout=names[2:]))
    baseline = tmp_path / 'baseline'
    (baseline / 'phase1').mkdir(parents=True)
    (baseline / 'phase1/checkpoint_best_mIoU.pth').write_bytes(b'fixed checkpoint')
    voc = tmp_path / 'data/voc12/VOCdevkit/VOC2012'
    for folder in (voc / 'JPEGImages', voc / 'SegmentationClassAug', baseline / 'pseudo'):
        folder.mkdir(parents=True)
    for name in names:
        array = np.zeros((32, 32, 3), dtype='uint8')
        array[:, :16] = 255
        Image.fromarray(array).save(voc / 'JPEGImages' / f'{name}.jpg')
        target = np.zeros((32, 32), dtype='uint8')
        target[:, :16] = 1
        Image.fromarray(target).save(voc / 'SegmentationClassAug' / f'{name}.png')
        Image.fromarray(target).save(baseline / 'pseudo' / f'{name}.png')
    created = []
    class ToyCam(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv2d(3, 20, 1, bias=False)
            torch.nn.init.constant_(self.head.weight, .1)
        def forward(self, x, **kwargs):
            seeds = self.head(F.avg_pool2d(x, 16)).relu()
            n = seeds.shape[-1]*seeds.shape[-2]
            return torch.zeros(1, 20), seeds, torch.eye(n)[None, None]
    def create(path):
        created.append(path)
        return ToyCam().eval()
    monkeypatch.setattr(cams, 'create_cam_model', create)
    cfg = dict(core.DEFAULTS, data=str(tmp_path / 'data'), baseline=str(baseline),
               output=str(tmp_path / 'output'), scales=[1.])
    cams.run_cams(cfg)
    cams.run_cams(cfg)
    assert len(created) == 1  # Second invocation scores hash-checked cached masks.
    scores = json.loads((tmp_path / 'output/cams/holdout-scores.json').read_text())
    assert scores['arms']['baseline']['images'] == 2
    assert scores['arms']['saved_crf_reference']['miou'] == 100.
    for path in (voc / 'SegmentationClassAug').glob('*.png'):
        path.unlink()
    cams.run_cams(cfg, full=True)  # Label generation must not try to read GT.
    assert len(list((tmp_path / 'output/uncertainty-full/trainaug/agreement').glob('*.png'))) == 4


def test_metrics_report_coverage_and_ignore_gt_255():
    stats = core.MaskStats()
    stats.update(np.array([[0, 255], [1, 2]]), np.array([[0, 1], [1, 255]]))
    result = stats.result()
    assert result['coverage_percent'] == pytest.approx(200/3)
    assert result['pixel_accuracy'] == 100.
    assert result['class_coverage']['aeroplane'] == 50.
    assert result['recall']['aeroplane'] == 100.  # Conditional on retained pixels.


def test_splits_are_disjoint_stable_and_order_independent():
    ids = [f'{i:03}' for i in range(40)]
    a = core.deterministic_splits(ids, 10, 20, 504)
    assert a == core.deterministic_splits(ids[::-1], 10, 20, 504)
    assert not set(a['dev']) & set(a['holdout'])
    with pytest.raises(ValueError):
        core.deterministic_splits(ids, 30, 20, 504)


def test_output_cannot_overlap_baseline(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(baseline=str(tmp_path / 'original'), output=str(tmp_path / 'original/pilot'))))
    with pytest.raises(ValueError, match='separate'):
        core.load_config(path)


def test_export_restore_hashes_and_refuses_overwrite(tmp_path):
    source = tmp_path / 'experiment'
    core.write_json(source / 'scores.json', dict(miou=12.3))
    archive = core.preserve(source)
    target = tmp_path / 'restored'
    core.restore(archive, target)
    assert (target / 'scores.json').read_bytes() == (source / 'scores.json').read_bytes()
    (target / 'scores.json').write_text('different')
    with pytest.raises(ValueError, match='overwrite'):
        core.restore(archive, target)


def test_bootstrap_rejects_unsafe_and_corrupt_archives(tmp_path):
    path = tmp_path / 'source.zip'
    for name, digest in [('../outside.txt', '0'*64), ('payload.txt', '0'*64)]:
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(name, 'payload')
            z.writestr('run-manifest.json', json.dumps(dict(schema=1, files={name: digest})))
        with pytest.raises(ValueError):
            unpack_manifest_archive(path, tmp_path / 'destination', 'run-manifest.json')
    assert not (tmp_path / 'outside.txt').exists()


def test_actual_pilot_loop_resume_equals_uninterrupted_cpu(tmp_path, monkeypatch):
    # Exercise the real experiment training/checkpoint loop with a tiny model/loader,
    # rather than only testing the underlying snapshot helper a second time.
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a, **k: self)
    for name in ('segm', 'segm.utils', 'segm.utils.torch'):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['segm.utils.torch'].set_gpu_dist_mode = lambda mode: None
    root = tmp_path / 'repository'
    (root / 'voc12').mkdir(parents=True)
    (root / 'voc12/train_aug_id.txt').write_text('a\nb\n')
    (root / 'voc12/val_id.txt').write_text('c\n')
    monkeypatch.setattr(finetune, 'ROOT', root)
    baseline = tmp_path / 'baseline'
    (baseline / 'phase2').mkdir(parents=True)
    (baseline / 'phase2/checkpoint.pth').write_bytes(b'fixed trusted source')
    for name in ('a', 'b'):
        (baseline / 'pseudo').mkdir(exist_ok=True)
        Image.fromarray(np.zeros((4, 4), dtype='uint8')).save(baseline / 'pseudo' / f'{name}.png')
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(1, 3, 1)
            self.decoder = torch.nn.Conv2d(3, 21, 1)
        def forward(self, x):
            return self.decoder(torch.nn.functional.dropout(self.encoder(x), .2, self.training))
    def model(cfg):
        return Toy(), {}
    def loaders(cfg, variant, annotations):
        gen = torch.Generator().manual_seed(cfg['seed'])
        data = torch.utils.data.TensorDataset(torch.arange(64.).reshape(4, 1, 4, 4)/64,
                                              torch.zeros(4, 4, 4, dtype=torch.long))
        class Batches:
            def __iter__(self):
                for x, y in self.loader:
                    yield dict(im=x, segmentation=y)
            def __len__(self):
                return len(self.loader)
        batch = Batches()
        batch.loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True, generator=gen)
        return batch, None, gen
    def evaluate(model, loader, save_dir=None):
        model.eval()
        return dict(miou=float(model(torch.ones(1, 1, 4, 4)).mean().detach())+float(torch.rand(()))+10)
    monkeypatch.setattr(finetune, 'model_and_variant', model)
    monkeypatch.setattr(finetune, 'data_loaders', loaders)
    monkeypatch.setattr(finetune, 'evaluate', evaluate)
    cfg = dict(core.DEFAULTS, baseline=str(baseline), epochs=2, clip_patch_size=2, learning_rate=.01,
               output=str(tmp_path / 'continuous'))
    finetune.train_arm(cfg, 'soft')
    cfg['output'] = str(tmp_path / 'resumed')
    cfg['train_epochs_this_session'] = 1
    finetune.train_arm(cfg, 'soft')
    finetune.train_arm(cfg, 'soft')
    left = load_checkpoint(tmp_path / 'continuous/training/soft/checkpoint.pth')
    right = load_checkpoint(tmp_path / 'resumed/training/soft/checkpoint.pth')
    for name in left['model']:
        torch.testing.assert_close(left['model'][name], right['model'][name], rtol=0, atol=0)
    assert left['experiment_metrics']['scores'] == right['experiment_metrics']['scores']


def test_experiment_notebook_is_clean_and_routes_stages():
    import nbformat
    notebook = nbformat.read(core.ROOT / 'WeakTr_Experiments_Kaggle.ipynb', as_version=4)
    nbformat.validate(notebook)
    config = ast.parse(notebook.cells[1].source)
    switch = next(n.value for n in config.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'RUN_FULL_UNCERTAINTY')
    assert ast.literal_eval(switch) is True
    commands = []
    namespace = dict(run=lambda stage, arm=None: commands.append((stage, arm)),
                     RUN_CLIPPING_TRAINING=True, RUN_FULL_UNCERTAINTY=False)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, '<experiment notebook>', 'exec')
            if cell.metadata.get('experiment_stage'):
                exec(cell.source, namespace)
    assert ('cams', None) in commands and ('clipping', None) in commands
    assert ('train', 'hard') in commands and ('train', 'soft') in commands
    assert ('uncertainty-labels', None) not in commands
    commands.clear()
    namespace['RUN_FULL_UNCERTAINTY'] = True
    for cell in notebook.cells:
        if cell.metadata.get('experiment_stage'):
            exec(cell.source, namespace)
    assert ('uncertainty-labels', None) in commands
    assert ('train', 'uncertainty-control') in commands and ('train', 'uncertainty') in commands


@pytest.mark.parametrize('extracted', [False, True])
def test_notebook_bootstrap_accepts_zip_and_extracted_source(tmp_path, extracted):
    import hashlib
    import os
    import nbformat
    notebook = nbformat.read(core.ROOT / 'WeakTr_Experiments_Kaggle.ipynb', as_version=4)
    code = notebook.cells[3].source
    inputs, work = tmp_path / 'inputs', tmp_path / 'work'
    inputs.mkdir()
    work.mkdir()
    content = b'# experiment runner fixture\n'
    name = 'reproduction/experiments/run.py'
    manifest = json.dumps(dict(schema=1, files={name: hashlib.sha256(content).hexdigest()}))
    if extracted:
        path = inputs / name
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        (inputs / 'source-manifest.json').write_text(manifest)
    else:
        with zipfile.ZipFile(inputs / 'weaktr-experiments-source.zip', 'w') as z:
            z.writestr(name, content)
            z.writestr('source-manifest.json', manifest)
    commands = []
    namespace = dict(INPUTS=inputs, WORK=work, SCRATCH=tmp_path / 'scratch',
                     REPO=work / 'code', RESUME_FROM='auto', SETTINGS={'seed':504},
                     Path=Path, hashlib=hashlib, json=json, os=os, sys=sys, zipfile=zipfile,
                     subprocess=types.SimpleNamespace(run=lambda argv, **kw: commands.append(argv)))
    exec(code, namespace)
    assert (work / 'code' / name).read_bytes() == content
    assert len(commands) == 1 and 'reproduction.experiments.bootstrap' in commands[0]
