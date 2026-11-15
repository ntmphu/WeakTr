"""CPU-friendly experiment configuration, metrics, and durable outputs."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np

from reproduction.data import read_ids, sha256

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ('background aeroplane bicycle bird boat bottle bus car cat chair cow '
           'diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor').split()
DEFAULTS = dict(schema=1, seed=504, data='/kaggle/tmp/weaktr-runtime/data',
                baseline='/kaggle/working/weaktr-artifacts',
                output='/kaggle/working/weaktr-experiments',
                cam_dev_images=100, cam_holdout_images=200, diagnostic_images=64,
                scales=[1.0, 1.2], background_threshold=0.43, agreement=0.75,
                soft_alpha=0.1, epochs=3, learning_rate=1e-5, encoder_lr_scale=0.1,
                batch_size=4, num_workers=2, start_value=1.2, clip_patch_size=120,
                train_epochs_this_session=3, smoke_steps=0, example_images=8)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def load_config(path):
    given = json.loads(Path(path).read_text(encoding='utf8'))
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown experiment settings: {sorted(unknown)}')
    cfg = dict(DEFAULTS, **given)
    if cfg['schema'] != 1:
        raise ValueError('Unsupported experiment schema')
    for name in ('epochs', 'batch_size', 'train_epochs_this_session', 'diagnostic_images',
                 'cam_dev_images', 'cam_holdout_images', 'clip_patch_size'):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not 0 < cfg['soft_alpha'] < 1 or not 0 < cfg['agreement'] <= 1:
        raise ValueError('soft_alpha must be in (0,1); agreement in (0,1]')
    if not 0 < cfg['background_threshold'] < 1 or cfg['learning_rate'] <= 0:
        raise ValueError('Invalid threshold or learning rate')
    if not cfg['scales'] or any(s <= 0 for s in cfg['scales']):
        raise ValueError('scales must be positive')
    if cfg['diagnostic_images'] > cfg['cam_dev_images'] + cfg['cam_holdout_images']:
        raise ValueError('diagnostic_images exceeds the available fixed diagnostic IDs')
    if not 0 < cfg['encoder_lr_scale'] <= 1 or cfg['start_value'] <= 0:
        raise ValueError('Invalid encoder LR multiplier or clipping activation threshold')
    if cfg['num_workers'] < 0 or cfg['smoke_steps'] < 0 or cfg['example_images'] < 0:
        raise ValueError('Worker, smoke-step and example counts must be nonnegative')
    for name in ('data', 'baseline', 'output'):
        cfg[name] = str(Path(cfg[name]).resolve())
    source, output = Path(cfg['baseline']), Path(cfg['output'])
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Experiment output must be separate from baseline artifacts')
    return cfg


def config_identity(cfg):
    # Relocation/session caps can change on restore; experimental semantics cannot.
    excluded = {'data', 'baseline', 'output', 'train_epochs_this_session'}
    identity = {k: v for k, v in cfg.items() if k not in excluded}
    # Protect restored results against silently changed operator/trainer code.
    paths = list((ROOT / 'reproduction/experiments').glob('*.py'))
    paths += [ROOT / name for name in ('models.py', 'AAF.py', 'vision_transformer.py')]
    paths += list((ROOT / 'OnlineRetraining/segm').rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update((path.relative_to(ROOT).as_posix()+':'+sha256(path)+'\n').encode())
    identity['implementation_sha256'] = digest.hexdigest()
    return identity


def deterministic_splits(ids, dev_count, holdout_count, seed):
    if len(set(ids)) != len(ids) or dev_count + holdout_count > len(ids):
        raise ValueError('Duplicate IDs or insufficient images for disjoint diagnostic splits')
    ordered = sorted(ids, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    return dict(dev=ordered[:dev_count], holdout=ordered[dev_count:dev_count + holdout_count])


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def confusion(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape:
        raise ValueError('Prediction and GT dimensions differ')
    valid = (truth >= 0) & (truth < 21) & (pred >= 0) & (pred < 21)
    return np.bincount(21 * truth[valid].astype('int64') + pred[valid], minlength=441).reshape(21, 21)


def metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    gt, pred, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    def ratio(a, b):
        return [float(100*x/y) if y else None for x, y in zip(a, b)]
    ious = ratio(tp, gt + pred - tp)
    present = [x for x in ious if x is not None]
    return dict(miou=float(np.mean(present)) if present else None,
                pixel_accuracy=float(100*tp.sum()/matrix.sum()) if matrix.sum() else None,
                class_iou=dict(zip(CLASSES, ious)),
                precision=dict(zip(CLASSES, ratio(tp, pred))),
                recall=dict(zip(CLASSES, ratio(tp, gt))), confusion_matrix=matrix.astype('int64').tolist())


def boundary_band(truth, radius=2):
    """GT boundary band, excluding transitions into VOC ignore regions."""
    truth = np.asarray(truth)
    valid = truth != 255
    edge = np.zeros(truth.shape, dtype=bool)
    horizontal = (truth[:, 1:] != truth[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (truth[1:] != truth[:-1]) & valid[1:] & valid[:-1]
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:] |= vertical
    edge[:-1] |= vertical
    padded = np.pad(edge, radius)
    band = np.zeros_like(edge)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            band |= padded[dy:dy+edge.shape[0], dx:dx+edge.shape[1]]
    return band & valid


class MaskStats:
    def __init__(self):
        self.matrix = np.zeros((21, 21), dtype=np.int64)
        self.boundary = self.matrix.copy()
        self.interior = self.matrix.copy()
        self.total_gt = np.zeros(21, dtype=np.int64)
        self.images = 0

    def update(self, pred, truth):
        self.matrix += confusion(pred, truth)
        band = boundary_band(truth)
        self.boundary += confusion(pred, np.where(band, truth, 255))
        self.interior += confusion(pred, np.where(~band, truth, 255))
        valid = (truth >= 0) & (truth < 21)
        self.total_gt += np.bincount(truth[valid], minlength=21)
        self.images += 1

    def result(self):
        result = metrics(self.matrix)
        kept = self.matrix.sum(1)
        result.update(images=self.images, coverage_percent=100*int(kept.sum())/max(1, int(self.total_gt.sum())),
                      class_coverage={c: float(100*k/t) if t else None
                                      for c, k, t in zip(CLASSES, kept, self.total_gt)},
                      boundary_band=metrics(self.boundary), interior=metrics(self.interior))
        return result


def guarded_run(directory, identity):
    """Reruns reuse only results produced with exactly the same semantics and sources."""
    directory = Path(directory)
    path = directory / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f'Existing experiment has different settings/sources: {directory}. Use a new output directory.')
    directory.mkdir(parents=True, exist_ok=True)
    write_json(path, identity)


def preserve(output):
    output = Path(output).resolve()
    files = {p.relative_to(output).as_posix(): sha256(p)
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'experiment-manifest.json'
             and not p.name.endswith(('.tmp', '.extracting'))}
    if not files:
        raise ValueError('No experiment artifacts to preserve')
    manifest = dict(schema=1, created_unix=time.time(), files=files)
    write_json(output / 'experiment-manifest.json', manifest)
    target = output.with_suffix('.zip')
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in files:
            archive.write(output / name, name)
        archive.write(output / 'experiment-manifest.json', 'experiment-manifest.json')
    os.replace(temporary, target)
    print(f'Experiment export: {target} ({target.stat().st_size / 1024**2:.1f} MiB)', flush=True)
    return target


def restore(source, output):
    """Restore experiment-only archives/folders, verifying payload hashes."""
    from reproduction.prepare import safe_name, copy_atomic
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        return
    output.mkdir(parents=True, exist_ok=True)
    pending = output / '.experiment-restore-in-progress.json'
    write_json(pending, dict(source=str(source)))
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        manifest = json.loads(archive.read('experiment-manifest.json') if archive else
                              (source / 'experiment-manifest.json').read_text())
        if manifest.get('schema') != 1 or not manifest.get('files'):
            raise ValueError('Invalid experiment export manifest')
        for name, expected in manifest['files'].items():
            relative = safe_name(name)
            target = output.joinpath(*relative.parts).resolve()
            if output not in target.parents:
                raise ValueError('Restore path escapes output')
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError(f'Refusing to overwrite different experiment artifact: {target}')
                continue
            if archive:
                with archive.open(name) as stream:
                    copy_atomic(stream, target)
            else:
                origin = source.joinpath(*relative.parts).resolve()
                if source not in origin.parents:
                    raise ValueError('Source path escapes restore folder')
                with origin.open('rb') as stream:
                    copy_atomic(stream, target)
            if sha256(target) != expected:
                target.unlink()
                raise ValueError(f'Experiment restore checksum mismatch: {name}')
        pending.unlink()
    finally:
        if archive:
            archive.close()


def pilot_splits(cfg):
    return deterministic_splits(read_ids(ROOT / 'voc12/train_id.txt'), cfg['cam_dev_images'],
                                cfg['cam_holdout_images'], cfg['seed'])
