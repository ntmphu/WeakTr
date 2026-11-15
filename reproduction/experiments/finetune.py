"""Matched short Phase-2 experiments with fresh SGD and a constant nonzero LR.

These are explicit post-reproduction fine-tuning experiments, not continuations
of the official 100-epoch polynomial schedule. Baseline artifacts are read-only.
"""
import copy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from reproduction.checkpoint import atomic_save, load_checkpoint, make_snapshot, restore_training
from reproduction.data import annotation_fingerprint, read_ids, sha256
from reproduction.experiments.core import (ROOT, CLASSES, MaskStats, config_identity, guarded_run,
                                          pilot_splits, seed_all, write_json)
from reproduction.experiments.methods import ResidualClipping, clip_threshold

ARMS = ('hard', 'soft', 'uncertainty-control', 'uncertainty')


def model_and_variant(cfg):
    from segm.model.factory import create_segmenter
    baseline = Path(cfg['baseline'])
    variant = yaml.safe_load((baseline / 'phase2/variant.yml').read_text())
    net = copy.deepcopy(variant['net_kwargs'])
    net['initialize'] = False
    model = create_segmenter(net)
    source = load_checkpoint(baseline / 'phase2/checkpoint.pth')
    model.load_state_dict(source['model'], strict=True)
    return model.cuda(), variant


def data_loaders(cfg, variant, annotations):
    from segm.data.factory import create_dataset
    kwargs = copy.deepcopy(variant['dataset_kwargs'])
    kwargs.update(ann_dir=str(annotations), batch_size=cfg['batch_size'],
                  num_workers=cfg['num_workers'], split='train')
    train = create_dataset(kwargs)
    valid = dict(kwargs, split='val', batch_size=1, crop=False)
    val = create_dataset(valid)
    # RandomSampler and DataLoader workers share a dedicated recoverable RNG.
    generator = torch.Generator().manual_seed(cfg['seed'])
    train.generator = generator
    train.sampler.generator = generator
    return train, val, generator


@torch.no_grad()
def evaluate(model, loader, save_dir=None):
    from segm.model.utils import inference
    model.eval()
    stats = MaskStats()
    gt = loader.dataset.get_gt_seg_maps()
    from reproduction.experiments.cams import save_mask
    started = time.monotonic()
    for index, batch in enumerate(loader):
        images = [image.cuda() for image in batch['im']]
        metadata = batch['im_metas']
        shape = metadata[0]['ori_shape']
        shape = (int(shape[0].item()), int(shape[1].item()))
        filename = metadata[0]['ori_filename'][0]
        logits = inference(model, images, metadata, shape, 480, 320, batch_size=1)
        prediction = logits.argmax(0).cpu().numpy()
        stats.update(prediction, gt[filename])
        if save_dir is not None:
            save_mask(Path(save_dir) / (Path(filename).stem + '.png'), prediction)
        if (index+1) % 200 == 0:
            print(f'Validation {index+1}/{len(loader)}', flush=True)
    result = stats.result()
    result.update(elapsed_seconds=time.monotonic()-started,
                  protocol='full VOC val; single-scale sliding window 480/320, single decoder, FP32, no CRF')
    return result


def train_arm(cfg, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    import segm.utils.torch as ptu
    ptu.set_gpu_dist_mode(True)
    seed_all(cfg['seed'])
    output = Path(cfg['output'])
    annotations = Path(cfg['baseline']) / 'pseudo'
    uncertainty = arm.startswith('uncertainty')
    if uncertainty:
        # Both uncertainty arms use the exact same regenerated RAW pseudo targets.
        # This avoids confounding agreement filtering with removing CRF targets.
        generation = output / 'uncertainty-full'
        receipt = generation / 'completed.json'
        if not receipt.exists():
            raise ValueError('Run uncertainty-labels first to generate all 10,582 labels')
        complete = json.loads(receipt.read_text())
        if complete['identity']['config'] != config_identity(cfg):
            raise ValueError('Full pseudo-label generation settings differ from training settings')
        annotations = generation / 'trainaug' / ('baseline' if arm.endswith('control') else 'agreement')
    names = read_ids(ROOT / 'voc12/train_aug_id.txt')
    fingerprint = annotation_fingerprint(annotations, names)
    checkpoint = Path(cfg['baseline']) / 'phase2/checkpoint.pth'
    directory = output / 'training' / arm
    identity = dict(config=config_identity(cfg), arm=arm, source_sha256=sha256(checkpoint),
                    pseudo_sha256=fingerprint, optimizer='fresh SGD momentum=.9; constant LR',
                    loss_reduction='valid-pixel mean' if uncertainty else 'official all-pixel mean',
                    training_list_sha256=sha256(ROOT / 'voc12/train_aug_id.txt'),
                    validation_list_sha256=sha256(ROOT / 'voc12/val_id.txt'))
    variant_path = Path(cfg['baseline']) / 'phase2/variant.yml'
    if variant_path.exists():
        identity['source_variant_sha256'] = sha256(variant_path)
    guarded_run(directory, identity)
    done = directory / 'completed.json'
    if done.exists():
        print(f'{arm}: already completed; reusing saved result', flush=True)
        return
    model, variant = model_and_variant(cfg)
    train, val, generator = data_loaders(cfg, variant, annotations)
    optimizer = torch.optim.SGD([
        dict(params=model.encoder.parameters(), lr=cfg['learning_rate']*cfg['encoder_lr_scale']),
        dict(params=model.decoder.parameters(), lr=cfg['learning_rate'])], momentum=.9, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.)
    clipping = ResidualClipping(cfg['soft_alpha'] if arm == 'soft' else 0.,
                                cfg['start_value'], cfg['clip_patch_size'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    latest = directory / 'checkpoint.pth'
    next_epoch, best = 0, dict(max_miou=-1., max_epoch=-1)
    if latest.exists():
        snapshot = load_checkpoint(latest)
        next_epoch, best = restore_training(snapshot, 'experiment-phase2', model, optimizer,
                                            scheduler, None, identity, generator)
        # The snapshot carries its last score, so an interrupted JSON write is recoverable.
        write_json(directory / f'epoch-{next_epoch}.json', snapshot['experiment_metrics'])
    else:
        initial = evaluate(model, val)
        write_json(directory / 'initial.json', initial)
    end = min(cfg['epochs'], next_epoch + cfg['train_epochs_this_session'])
    for epoch in range(next_epoch, end):
        started = time.monotonic()
        model.train()
        clipping.valid_pixels = clipping.clipped_pixels = 0
        loss_sum = 0.
        for step, batch in enumerate(train):
            images, targets = batch['im'].cuda(), batch['segmentation'].long().cuda()
            optimizer.zero_grad(set_to_none=True)
            pred = model(images)
            original, losses = clipping(pred, targets, criterion)
            # Ignored targets contribute zero; normalize by valid pixels in BOTH uncertainty arms.
            valid = targets != 255
            loss = losses[valid].sum()/valid.sum().clamp_min(1) if uncertainty else losses.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {arm}, epoch {epoch}, step {step}')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (step+1) % 100 == 0:
                print(f'{arm} epoch {epoch+1}/{cfg["epochs"]} batch {step+1}/{len(train)} '
                      f'loss={loss.item():.4f}', flush=True)
            if cfg['smoke_steps'] and step+1 >= cfg['smoke_steps']:
                break
        scheduler.step()
        train_seconds = time.monotonic()-started
        save_dir = directory / 'predictions' if epoch+1 == cfg['epochs'] else None
        scores = evaluate(model, val, save_dir)
        if scores['miou'] > best['max_miou']:
            best = dict(max_miou=scores['miou'], max_epoch=epoch)
        result = dict(epoch=epoch+1, scores=scores, train_loss=loss_sum/(step+1),
                      train_seconds=train_seconds, batches=step+1,
                      clipped_fraction=clipping.clipped_pixels/max(1, clipping.valid_pixels),
                      alpha=clipping.alpha, smoke_only=bool(cfg['smoke_steps']))
        snapshot = make_snapshot('experiment-phase2', model, optimizer, scheduler, None,
                                 epoch, best, identity, generator)
        snapshot['experiment_metrics'] = result
        atomic_save(snapshot, latest, retain_previous=True)
        write_json(directory / f'epoch-{epoch+1}.json', result)
        print(f'{arm}: epoch {epoch+1}, val mIoU {scores["miou"]:.4f}', flush=True)
    if end == cfg['epochs']:
        write_json(done, dict(status='smoke_only' if cfg['smoke_steps'] else 'completed',
                   completed_epochs=end, checkpoint_sha256=sha256(latest),
                   selection='last planned pilot epoch; no best-epoch selection', identity=identity))


@torch.no_grad()
def clipping_diagnostic(cfg):
    seed_all(cfg['seed'])
    model, variant = model_and_variant(cfg)
    model.eval()
    from segm.data.utils import STATS
    normalization = STATS[variant['net_kwargs']['normalization']]
    mean = torch.tensor(normalization['mean'], device='cuda')[None, :, None, None]
    std = torch.tensor(normalization['std'], device='cuda')[None, :, None, None]
    voc = Path(cfg['data']) / 'voc12/VOCdevkit/VOC2012'
    pseudo = Path(cfg['baseline']) / 'pseudo'
    splits = pilot_splits(cfg)
    names = (splits['dev']+splits['holdout'])[:cfg['diagnostic_images']]
    directory = Path(cfg['output']) / 'clipping'
    identity = dict(config=config_identity(cfg), checkpoint_sha256=sha256(Path(cfg['baseline']) / 'phase2/checkpoint.pth'),
                    names=names, protocol='fixed 480x480 resize, batches as configured; training GT diagnostic only')
    guarded_run(directory, identity)
    counts = np.zeros((21, 4), dtype=np.int64)  # correct kept/rejected, incorrect kept/rejected
    batches = []
    for offset in range(0, len(names), cfg['batch_size']):
        images, targets, truths = [], [], []
        for name in names[offset:offset+cfg['batch_size']]:
            with Image.open(voc / 'JPEGImages' / f'{name}.jpg') as raw:
                im = np.array(raw.convert('RGB').resize((480, 480), Image.Resampling.BICUBIC))
            images.append(torch.from_numpy(im).permute(2, 0, 1).float()/255)
            for folder, collection in ((pseudo, targets), (voc / 'SegmentationClassAug', truths)):
                with Image.open(folder / f'{name}.png') as raw:
                    collection.append(torch.from_numpy(np.array(raw.resize((480, 480), Image.Resampling.NEAREST))).long())
        images = (torch.stack(images).cuda()-mean)/std
        target, truth = torch.stack(targets).cuda(), torch.stack(truths).cuda()
        loss = F.cross_entropy(model(images), target, ignore_index=255, reduction='none')
        threshold = clip_threshold(loss, cfg['start_value'], cfg['clip_patch_size'])
        rejected = loss > threshold if threshold is not None else torch.zeros_like(loss, dtype=torch.bool)
        valid = (target != 255) & (truth != 255)
        correct = target == truth
        for c in range(21):
            selected = valid & (truth == c)
            counts[c] += np.array([int((selected & correct & ~rejected).sum()),
                                   int((selected & correct & rejected).sum()),
                                   int((selected & ~correct & ~rejected).sum()),
                                   int((selected & ~correct & rejected).sum())])
        batches.append(dict(names=names[offset:offset+cfg['batch_size']], mean_loss=float(loss.mean()),
                            clipping_active=threshold is not None))
    total = counts.sum(0)
    def ratio(x, y):
        return float(100*x/y) if y else None
    write_json(directory / 'scores.json', dict(status='completed', identity=identity,
        columns=['correct_kept', 'correct_rejected', 'incorrect_kept', 'incorrect_rejected'],
        per_gt_class=dict(zip(CLASSES, counts.tolist())), totals=total.tolist(), batches=batches,
        correct_label_rejection_percent=ratio(total[1], total[0]+total[1]),
        incorrect_label_rejection_percent=ratio(total[3], total[2]+total[3]),
        accuracy_of_rejected_percent=ratio(total[1], total[1]+total[3]),
        accuracy_of_kept_percent=ratio(total[0], total[0]+total[2]),
        note='Frozen final model at fixed resize; not the actual early-training rejection distribution. '
             'GT is read only after predictions/loss selection; never used as a training target.'))
