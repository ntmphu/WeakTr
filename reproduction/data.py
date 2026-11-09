"""Strict VOC checks, without importing either training stack."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path, expected=None):
    ids = Path(path).read_text().splitlines()
    if not ids or any(not name or name.strip() != name or Path(name).name != name or '/' in name or '\\' in name for name in ids):
        raise ValueError(f'Invalid image IDs: {path}')
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate image IDs: {path}')
    if expected is not None and len(ids) != expected:
        raise ValueError(f'Expected {expected} IDs in {path}, got {len(ids)}')
    return ids


def annotation_fingerprint(directory, ids):
    digest = hashlib.sha256()
    for name in ids:
        digest.update((name + ':' + sha256(Path(directory) / (name + '.png')) + '\n').encode())
    return digest.hexdigest()


def validate_masks(ids, images, masks):
    for name in ids:
        image_path, mask_path = Path(images) / (name + '.jpg'), Path(masks) / (name + '.png')
        with Image.open(image_path) as image:
            image.load()
            size = image.size
        with Image.open(mask_path) as mask:
            mask.load()
            labels = np.asarray(mask)
            if mask.size != size or labels.ndim != 2:
                raise ValueError(f'Mask dimensions mismatch: {mask_path}')
            if not np.isin(labels, list(range(21)) + [255]).all():
                raise ValueError(f'Invalid class IDs: {mask_path}')


def validate_voc(root, train_list, val_list, pseudo_dir=None, expected=(10582, 1449)):
    root = Path(root).resolve()
    train, val = read_ids(train_list, expected[0]), read_ids(val_list, expected[1])
    if set(train) & set(val):
        raise ValueError('Training and validation IDs overlap.')
    images = root / 'JPEGImages'
    validate_masks(train, images, root / 'SegmentationClassAug')
    validate_masks(val, images, root / 'SegmentationClassAug')
    # Official CAM selection uses train_id and augmented GT; test those IDs too.
    if pseudo_dir is not None:
        pseudo_dir = Path(pseudo_dir).resolve()
        if pseudo_dir in (root / 'SegmentationClass', root / 'SegmentationClassAug'):
            raise ValueError('Phase 2 requires generated pseudo-labels, not ground-truth annotations.')
        validate_masks(train, images, pseudo_dir)
    return dict(train_count=len(train), val_count=len(val),
                train_list_sha256=sha256(train_list), val_list_sha256=sha256(val_list),
                pseudo_dir=str(pseudo_dir) if pseudo_dir else None)
