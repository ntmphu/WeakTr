"""Build one private Kaggle input ZIP from locally validated VOC inputs."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile

from reproduction.data import read_ids, sha256, validate_voc
from reproduction.prepare import DEIT_SHA256, ROOT


def build_bundle(data, weights, output):
    data, weights, output = Path(data), Path(weights), Path(output)
    voc = data / 'voc12/VOCdevkit/VOC2012'
    validation = validate_voc(voc, ROOT / 'voc12/train_aug_id.txt', ROOT / 'voc12/val_id.txt', progress=True)
    pretrained = weights / 'deit_small_patch16_224.pth'
    if sha256(pretrained) != DEIT_SHA256:
        raise ValueError('Bundle requires the verified official ImageNet DeiT-S checkpoint.')
    ids = sorted(read_ids(ROOT / 'voc12/train_aug_id.txt') + read_ids(ROOT / 'voc12/val_id.txt'))
    entries = {}
    for name in ids:
        for folder, extension in [('JPEGImages', '.jpg'), ('SegmentationClassAug', '.png')]:
            path = voc / folder / (name + extension)
            entries['data/' + path.relative_to(data).as_posix()] = path
        original = voc / 'SegmentationClass' / (name + '.png')
        if original.exists():
            entries['data/' + original.relative_to(data).as_posix()] = original
    for name in ('trainaug.txt', 'val.txt', 'train.txt'):
        path = voc / 'ImageSets/Segmentation' / name
        entries['data/' + path.relative_to(data).as_posix()] = path
    entries['weights/deit_small_patch16_224.pth'] = pretrained
    manifest = dict(schema=1, validation=validation, files={},
                    pretrained_sha256=DEIT_SHA256,
                    preparation=json.loads((data / 'preparation.json').read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output.name + '.', suffix='.part', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for index, (name, path) in enumerate(sorted(entries.items()), 1):
            manifest['files'][name] = sha256(path)
            archive.write(path, name)
            if index % 2000 == 0:
                print(f'Bundled {index}/{len(entries)} files', flush=True)
        archive.writestr('weaktr-inputs-manifest.json', json.dumps(manifest, indent=2) + '\n')
    temporary.replace(output)
    summary = dict(path=str(output.resolve()), sha256=sha256(output), bytes=output.stat().st_size,
                   files=len(entries), validation=validation)
    output.with_name(output.stem + '-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--weights', default='data/weights')
    parser.add_argument('--output', default='data/voc12/weaktr-kaggle-inputs.zip')
    args = parser.parse_args()
    build_bundle(args.data, args.weights, args.output)


if __name__ == '__main__':
    main()
