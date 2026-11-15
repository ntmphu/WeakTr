"""Frozen Phase-1 propagation and view-agreement pilots (GT used only for scoring)."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from reproduction.checkpoint import load_checkpoint
from reproduction.data import read_ids, sha256
from reproduction.experiments.core import (ROOT, MaskStats, config_identity, guarded_run,
                                          pilot_splits, seed_all, write_json)
from reproduction.experiments.methods import (gate_affinity, normalize_cams, cam_labels,
                                             agreement_mask)


def save_mask(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    Image.fromarray(np.asarray(labels, dtype=np.uint8)).save(temp, format='PNG')
    temp.replace(path)


def create_cam_model(checkpoint):
    from timm.models import create_model
    import models  # Registers the original DeiT-S WeakTr.
    model = create_model('deit_small_WeakTr_patch16_224', pretrained=False,
                         num_classes=20, reduction=8, pool='max', feat_reduction=8)
    state = load_checkpoint(checkpoint)
    model.load_state_dict(state['model'], strict=True)
    return model.cuda().eval()


@torch.no_grad()
def predict_views(model, image, present, scales, compute_gate=True):
    """Same native-resolution scales/flips/normalization as official CAM generation.

    Model and propagation run in FP32 in both arms for a controlled pilot.
    The final encoder patch features are captured before the CAM convolution.
    """
    original_w, original_h = image.size
    device = next(model.parameters()).device
    captured = {}
    def capture(module, inputs):
        captured['features'] = inputs[0].flatten(2).transpose(1, 2).detach()
    hook = model.head.register_forward_pre_hook(capture)
    views, gated_views = [], []
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    present = torch.as_tensor(present, device=device).reshape(20, 1, 1)
    try:
        for scale in scales:
            size = (round(original_w * scale), round(original_h * scale))
            if min(size) < 16:
                raise ValueError(f'Image too small for patch16: {size}')
            resized = image.resize(size, Image.Resampling.BICUBIC)
            tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1)[None].to(device).float()/255
            tensor = (tensor - mean) / std
            for flip in (False, True):
                inputs = tensor.flip(-1) if flip else tensor
                _, seeds, pa = model(inputs, return_att=True, attention_type='fused')
                attention = pa.sum(0)
                operators = [(attention, views)]
                if compute_gate:
                    operators.append((gate_affinity(attention, captured['features']), gated_views))
                flat = seeds.flatten(2)
                for operator, collection in operators:
                    maps = torch.einsum('bij,bcj->bci', operator, flat).reshape_as(seeds)
                    maps = F.interpolate(maps, size=(original_h, original_w),
                                         mode='bilinear', align_corners=False)[0] * present
                    collection.append(maps.flip(-1) if flip else maps)
    finally:
        hook.remove()
    return views, gated_views


def image_label(labels, name):
    key = name if name in labels else name + '.jpg'
    return labels[key]


def run_cams(cfg, full=False):
    seed_all(cfg['seed'])
    baseline, output = Path(cfg['baseline']), Path(cfg['output'])
    checkpoint = baseline / 'phase1/checkpoint_best_mIoU.pth'
    directory = output / ('uncertainty-full' if full else 'cams')
    splits = {'trainaug': read_ids(ROOT / 'voc12/train_aug_id.txt')} if full else pilot_splits(cfg)
    identity = dict(config=config_identity(cfg), checkpoint_sha256=sha256(checkpoint), splits=splits,
                    implementation='frozen-fp32-native-resolution-v1')
    guarded_run(directory, identity)
    voc = Path(cfg['data']) / 'voc12/VOCdevkit/VOC2012'
    labels = np.load(ROOT / 'voc12/cls_labels.npy', allow_pickle=True).item()
    model = None
    started = time.monotonic()
    for split, names in splits.items():
        arms = ('baseline', 'gated', 'agreement') if not full else ('baseline', 'agreement')
        stats = {arm: MaskStats() for arm in arms}
        saved_reference = MaskStats() if not full else None
        for index, name in enumerate(names):
            targets = {arm: directory / split / arm / f'{name}.png' for arm in arms}
            receipt = directory / split / 'receipts' / f'{name}.json'
            cached = False
            if receipt.exists() and all(p.exists() for p in targets.values()):
                hashes = json.loads(receipt.read_text())
                cached = all(sha256(p) == hashes.get(arm) for arm, p in targets.items())
            if not cached:
                if model is None:
                    model = create_cam_model(checkpoint)
                with Image.open(voc / 'JPEGImages' / f'{name}.jpg') as source:
                    image = source.convert('RGB')
                views, gated = predict_views(model, image, image_label(labels, name), cfg['scales'], compute_gate=not full)
                aggregate = torch.stack(views).sum(0)
                pred = cam_labels(normalize_cams(aggregate), cfg['background_threshold'])
                selected, agreement = agreement_mask(views, aggregate, cfg['background_threshold'], cfg['agreement'])
                predictions = dict(baseline=pred, agreement=selected)
                if not full:
                    predictions['gated'] = cam_labels(normalize_cams(torch.stack(gated).sum(0)),
                                                     cfg['background_threshold'])
                for arm, path in targets.items():
                    save_mask(path, predictions[arm].cpu().numpy())
                write_json(receipt, {arm: sha256(path) for arm, path in targets.items()})
                del views, gated, aggregate, predictions, selected, agreement, pred
            # Full-generation stage deliberately never loads GT; training cannot depend on it.
            if not full:
                with Image.open(voc / 'SegmentationClassAug' / f'{name}.png') as image:
                    truth = np.array(image)
                for arm, path in targets.items():
                    with Image.open(path) as image:
                        stats[arm].update(np.array(image), truth)
                with Image.open(baseline / 'pseudo' / f'{name}.png') as image:
                    saved_reference.update(np.array(image), truth)
            if (index + 1) % 10 == 0 or index + 1 == len(names):
                elapsed = time.monotonic() - started
                print(f'{split}: {index+1}/{len(names)}, {elapsed:.1f}s elapsed', flush=True)
        results = {arm: value.result() for arm, value in stats.items()} if not full else {}
        if saved_reference:
            results['saved_crf_reference'] = saved_reference.result()
        write_json(directory / f'{split}-scores.json', dict(
            status='completed', scope='training-pseudo-label diagnostics; not VOC validation' if not full else 'full label generation',
            images=len(names), arms=results,
            note='Agreement metrics condition on retained pixels; compare coverage, not mIoU alone. '
                 'Boundary-band IoU is restricted to a two-pixel GT band, not the standard boundary-IoU metric.'))
    write_json(directory / 'completed.json', dict(status='completed', identity=identity,
               elapsed_seconds=time.monotonic()-started,
               label_directory=str(directory / 'trainaug/agreement') if full else None))


def make_examples(cfg):
    """Contact sheets use fixed IDs, so examples are not chosen for positive results."""
    from PIL import ImageDraw
    output = Path(cfg['output'])
    voc = Path(cfg['data']) / 'voc12/VOCdevkit/VOC2012'
    palette = []
    for index in range(256):
        r = g = b = 0
        value = index
        for bit in range(8):
            r |= ((value >> 0) & 1) << (7-bit)
            g |= ((value >> 1) & 1) << (7-bit)
            b |= ((value >> 2) & 1) << (7-bit)
            value >>= 3
        palette.extend([r, g, b])
    palette[765:768] = [180, 180, 180]
    for name in pilot_splits(cfg)['holdout'][:cfg['example_images']]:
        paths = [voc / 'JPEGImages' / f'{name}.jpg', voc / 'SegmentationClassAug' / f'{name}.png']
        paths += [output / 'cams/holdout' / arm / f'{name}.png' for arm in ('baseline', 'gated', 'agreement')]
        if not all(p.exists() for p in paths):
            continue
        sheet = Image.new('RGB', (5*240, 265), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (path, title) in enumerate(zip(paths, ('Image', 'GT (diagnostic)', 'Baseline CAM', 'Gated CAM', 'Agreement; gray=ignore'))):
            with Image.open(path) as raw:
                if i:
                    raw = Image.fromarray(np.asarray(raw).astype('uint8'), mode='P')
                    raw.putpalette(palette)
                tile = raw.convert('RGB')
                tile.thumbnail((240, 240), Image.Resampling.NEAREST)
                sheet.paste(tile, (i*240, 25))
            draw.text((i*240+3, 5), title, fill='black')
        target = output / 'figures' / f'{name}.png'
        target.parent.mkdir(exist_ok=True, parents=True)
        sheet.save(target)
