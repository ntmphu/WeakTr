"""Download VOC train/val, augmented masks and ImageNet weights using CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VOC_URLS = [
    'https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
    'https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',
]
VOC_MD5 = '6cd6e144f989b92b3379bac3b3de84fd'  # torchvision.datasets.voc
AUG_URLS = [
    'https://dl.dropboxusercontent.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip',
    'https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1',
]
DEIT_URL = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
DEIT_SHA256 = 'cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f'


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def check_file(path, checksum=None, algorithm='sha256', kind=None):
    if not Path(path).is_file() or not Path(path).stat().st_size:
        raise ValueError(f'Empty/missing download: {path}')
    if checksum and digest(path, algorithm) != checksum:
        raise ValueError(f'{algorithm} checksum mismatch: {path}')
    if kind == 'zip' and not zipfile.is_zipfile(path):
        raise ValueError(f'Expected a ZIP archive, possibly received an HTML download page: {path}')
    if kind == 'tar' and not tarfile.is_tarfile(path):
        raise ValueError(f'Expected a TAR archive: {path}')


def download(urls, target, checksum=None, algorithm='sha256', kind=None, retries=3):
    """Resume partial responses; restart if the server ignores Range. Keep cached files."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        check_file(target, checksum, algorithm, kind)
        print(f'Reusing verified cache: {target}', flush=True)
        return target
    partial = target.with_name(target.name + '.part')
    errors = []
    for url in urls:
        for attempt in range(retries):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {'User-Agent': 'WeakTr-reproduction/1.0', 'Accept-Encoding': 'identity'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                print(f'Downloading {target.name} (attempt {attempt + 1}): {url}', flush=True)
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    resumed = response.status == 206
                    if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Server returned an unexpected Range response.')
                    length = response.headers.get('Content-Length')
                    expected = int(length) if length is not None else None
                    received = 0
                    next_report = time.monotonic() + 5
                    with partial.open('ab' if resumed else 'wb') as stream:
                        for chunk in iter(lambda: response.read(1024 * 1024), b''):
                            stream.write(chunk)
                            received += len(chunk)
                            if time.monotonic() >= next_report:
                                total = (offset if resumed else 0) + received
                                print(f'  {target.name}: {total / 1024**2:.0f} MiB', flush=True)
                                next_report = time.monotonic() + 5
                    if expected is not None and received != expected:
                        raise OSError(f'Incomplete HTTP body: {received}/{expected} bytes')
                    resolved_url = response.geturl()
                try:
                    check_file(partial, checksum, algorithm, kind)
                except ValueError:
                    partial.replace(partial.with_name(partial.name + f'.invalid-{time.time_ns()}'))
                    raise
                os.replace(partial, target)
                target.with_name(target.name + '.source.json').write_text(json.dumps(dict(
                    url=url, resolved_url=resolved_url, sha256=digest(target),
                    checksum=checksum, checksum_algorithm=algorithm,
                    bytes=target.stat().st_size), indent=2) + '\n')
                return target
            except (OSError, ValueError) as error:
                errors.append(f'{url}: {error}')
                print(f'  Download failed: {error}', flush=True)
                if attempt + 1 < retries:
                    time.sleep(attempt + 1)
    raise RuntimeError('All download attempts failed. Rerun to resume, or provide a local archive.\n' + '\n'.join(errors))


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or any(':' in part for part in path.parts):
        raise ValueError(f'Unsafe archive path: {name}')
    return path


def copy_atomic(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.extracting')
    with partial.open('wb') as stream:
        shutil.copyfileobj(source, stream)
    os.replace(partial, target)


def extract_voc(archive, destination):
    """Keep segmentation inputs only; skip test images and detection annotations."""
    destination = Path(destination)
    count = 0
    with tarfile.open(archive) as collection:
        for member in collection:
            path = safe_name(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f'Archive links are not supported: {member.name}')
            if not member.isfile():
                continue
            parts = path.parts
            if parts[:2] != ('VOCdevkit', 'VOC2012'):
                continue
            selected = (len(parts) == 4 and parts[2] in ('JPEGImages', 'SegmentationClass')) or (
                len(parts) == 5 and parts[2:4] == ('ImageSets', 'Segmentation'))
            if selected:
                with collection.extractfile(member) as source:
                    copy_atomic(source, destination.joinpath(*parts))
                count += 1
    if not count:
        raise ValueError('VOC archive contains no expected segmentation inputs.')
    return count


def extract_aug(archive, destination):
    destination = Path(destination)
    names = set()
    with zipfile.ZipFile(archive) as collection:
        for member in collection.infolist():
            path = safe_name(member.filename)
            if member.is_dir() or path.suffix.lower() != '.png' or '__MACOSX' in path.parts:
                continue
            if path.name.startswith('._'):
                continue
            if path.name in names:
                raise ValueError(f'Duplicate augmented mask: {path.name}')
            names.add(path.name)
            with collection.open(member) as source:
                copy_atomic(source, destination / path.name)
    if not names:
        raise ValueError('Augmented annotation archive contains no PNG masks.')
    return len(names)


def install_splits(voc, repository=ROOT):
    split_dir = Path(voc) / 'ImageSets/Segmentation'
    split_dir.mkdir(parents=True, exist_ok=True)
    for source, target in [('train_aug_id.txt', 'trainaug.txt'), ('val_id.txt', 'val.txt'), ('train_id.txt', 'train.txt')]:
        ids = (Path(repository) / 'voc12' / source).read_text().splitlines()
        (split_dir / target).write_text('\n'.join(ids) + '\n')


def prepare(data, cache, weights, voc_archive=None, aug_archive=None, skip_weights=False, force_extract=False,
            pretrained_source=None):
    data, cache, weights = Path(data).resolve(), Path(cache).resolve(), Path(weights).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    voc_archive = Path(voc_archive).resolve() if voc_archive else download(
        VOC_URLS, cache / 'VOCtrainval_11-May-2012.tar', VOC_MD5, 'md5', 'tar')
    aug_archive = Path(aug_archive).resolve() if aug_archive else download(
        AUG_URLS, cache / 'SegmentationClassAug.zip', kind='zip')
    check_file(voc_archive, VOC_MD5, 'md5', 'tar')
    check_file(aug_archive, kind='zip')
    voc = data / 'voc12/VOCdevkit/VOC2012'
    voc.mkdir(parents=True, exist_ok=True)
    for archive, marker_name, extract in [
        (voc_archive, '.voc-extracted.json', lambda: extract_voc(voc_archive, data / 'voc12')),
        (aug_archive, '.aug-extracted.json', lambda: extract_aug(aug_archive, voc / 'SegmentationClassAug')),
    ]:
        marker = voc / marker_name
        fingerprint = digest(archive)
        if force_extract or not marker.exists() or json.loads(marker.read_text())['sha256'] != fingerprint:
            print(f'Extracting {archive.name}...', flush=True)
            count = extract()
            marker.write_text(json.dumps(dict(sha256=fingerprint, files=count)) + '\n')
        else:
            print(f'Reusing extracted {archive.name}', flush=True)
    install_splits(voc)
    # Some augmented distributions omit validation masks; use original VOC GT for those.
    copied = 0
    for name in (ROOT / 'voc12/val_id.txt').read_text().splitlines():
        target = voc / 'SegmentationClassAug' / (name + '.png')
        if not target.exists():
            with (voc / 'SegmentationClass' / (name + '.png')).open('rb') as source:
                copy_atomic(source, target)
            copied += 1
    weight_path = weights / 'deit_small_patch16_224.pth'
    if not skip_weights:
        if pretrained_source:
            source_path = Path(pretrained_source).resolve()
            check_file(source_path, DEIT_SHA256)
            if source_path != weight_path:
                with source_path.open('rb') as source:
                    copy_atomic(source, weight_path)
        else:
            download([DEIT_URL], weight_path, DEIT_SHA256)
    result = dict(voc_root=str(voc), data=str(data), pretrained=str(weight_path),
                  validation_masks_copied_from_original=copied,
                  archives={str(p): digest(p) for p in (voc_archive, aug_archive)},
                  sources=dict(voc=VOC_URLS, augmented=AUG_URLS, imagenet=DEIT_URL))
    (data / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Inputs prepared: {voc}\nImageNet initialization: {weight_path}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'data'))
    parser.add_argument('--cache', default=None)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--voc-archive')
    parser.add_argument('--aug-archive')
    parser.add_argument('--skip-weights', action='store_true')
    parser.add_argument('--pretrained-source', help='Reuse an existing verified ImageNet checkpoint')
    parser.add_argument('--force-extract', action='store_true')
    args = parser.parse_args()
    for argument, filename in [('voc_archive', 'VOCtrainval_11-May-2012.tar'),
                               ('aug_archive', 'SegmentationClassAug.zip')]:
        candidate = Path(args.data) / 'voc12' / filename
        if getattr(args, argument) is None and candidate.is_file():
            setattr(args, argument, str(candidate))
    prepare(args.data, args.cache or Path(args.data) / '.downloads',
            args.weights or Path(args.data) / 'weights', args.voc_archive,
            args.aug_archive, args.skip_weights, args.force_extract, args.pretrained_source)


if __name__ == '__main__':
    main()
