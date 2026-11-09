import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from reproduction import prepare
from reproduction import kaggle
from reproduction.kaggle import export_run, restore_inputs, restore_run


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {'Content-Length': str(len(body))}

    def geturl(self):
        return 'https://example.test/archive'


@pytest.mark.parametrize('resume', [True, False])
def test_download_resumes_or_restarts_when_range_is_ignored(tmp_path, monkeypatch, resume):
    target = tmp_path / 'weights.pth'
    target.with_name(target.name + '.part').write_bytes(b'first')
    whole = b'firstsecond'
    def response(request, **kwargs):
        assert request.headers['Range'] == 'bytes=5-'
        if resume:
            return Response(b'second', 206, {'Content-Length': '6', 'Content-Range': 'bytes 5-10/11'})
        return Response(whole)
    monkeypatch.setattr(prepare, 'urlopen', response)
    prepare.download(['https://example.test'], target, hashlib.sha256(whole).hexdigest())
    assert target.read_bytes() == whole
    assert json.loads(target.with_name(target.name + '.source.json').read_text())['sha256'] == hashlib.sha256(whole).hexdigest()
    # Cached files are checked and reused without a second request.
    monkeypatch.setattr(prepare, 'urlopen', lambda *a, **k: pytest.fail('Unexpected network request'))
    prepare.download(['https://example.test'], target, hashlib.sha256(whole).hexdigest())


def test_download_rejects_html_and_preserves_incomplete_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, 'urlopen', lambda *a, **k: Response(b'<html>not a zip</html>'))
    with pytest.raises(RuntimeError, match='Expected a ZIP'):
        prepare.download(['https://example.test'], tmp_path / 'archive.zip', kind='zip', retries=1)
    assert not (tmp_path / 'archive.zip').exists()
    monkeypatch.setattr(prepare, 'urlopen', lambda *a, **k: Response(b'partial', headers={'Content-Length': '100'}))
    with pytest.raises(RuntimeError, match='Incomplete HTTP body'):
        prepare.download(['https://example.test'], tmp_path / 'incomplete', retries=1)
    assert (tmp_path / 'incomplete.part').read_bytes() == b'partial'


def make_tar(path, files):
    with tarfile.open(path, 'w') as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_archive_extraction_selects_inputs_and_rejects_traversal(tmp_path):
    archive = tmp_path / 'voc.tar'
    make_tar(archive, {'VOCdevkit/VOC2012/JPEGImages/a.jpg': b'image',
                       'VOCdevkit/VOC2012/Annotations/a.xml': b'detection'})
    assert prepare.extract_voc(archive, tmp_path / 'data') == 1
    assert (tmp_path / 'data/VOCdevkit/VOC2012/JPEGImages/a.jpg').read_bytes() == b'image'
    assert not (tmp_path / 'data/VOCdevkit/VOC2012/Annotations').exists()
    make_tar(archive, {'../escaped': b'bad'})
    with pytest.raises(ValueError, match='Unsafe'):
        prepare.extract_voc(archive, tmp_path / 'data')
    zipped = tmp_path / 'aug.zip'
    with zipfile.ZipFile(zipped, 'w') as collection:
        collection.writestr('SegmentationClassAug/a.png', b'mask')
        collection.writestr('__MACOSX/SegmentationClassAug/._a.png', b'metadata')
    assert prepare.extract_aug(zipped, tmp_path / 'masks') == 1
    assert (tmp_path / 'masks/a.png').read_bytes() == b'mask'


def test_restore_uploaded_inputs_verifies_hashes_and_reuses_valid_local_copy(tmp_path):
    uploaded, work = tmp_path / 'inputs/dataset', tmp_path / 'work'
    (uploaded / 'weights').mkdir(parents=True)
    (uploaded / 'weights/imagenet.pth').write_bytes(b'weights')
    manifest = dict(schema=1, files={'weights/imagenet.pth': hashlib.sha256(b'weights').hexdigest()})
    (uploaded / 'weaktr-inputs-manifest.json').write_text(json.dumps(manifest))
    restore_inputs(tmp_path / 'inputs', work)
    assert (work / 'weights/imagenet.pth').read_bytes() == b'weights'
    (work / 'weights/imagenet.pth').write_bytes(b'broken local copy')
    restore_inputs(tmp_path / 'inputs', work)
    assert (work / 'weights/imagenet.pth').read_bytes() == b'weights'
    (work / 'weights/imagenet.pth').unlink()
    (uploaded / 'weights/imagenet.pth').write_bytes(b'broken upload')
    with pytest.raises(ValueError, match='Uploaded input checksum'):
        restore_inputs(tmp_path / 'inputs', work)


def test_split_files_are_generated_from_repository_lists(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'voc12').mkdir(parents=True)
    for name in ('train_aug_id.txt', 'val_id.txt', 'train_id.txt'):
        (repo / 'voc12' / name).write_text('2007_000032\n')
    prepare.install_splits(tmp_path / 'VOC2012', repo)
    for name in ('trainaug.txt', 'val.txt', 'train.txt'):
        assert (tmp_path / 'VOC2012/ImageSets/Segmentation' / name).read_text() == '2007_000032\n'


def test_saved_run_roundtrip_keeps_resume_artifacts_and_excludes_cams(tmp_path):
    output = tmp_path / 'active'
    for name, body in {'phase1/checkpoint.pth': b'full snapshot', 'phase1/variant.yml': b'config',
                       'pseudo/a.png': b'pseudo', 'cams/a.npy': b'heavy CAM',
                       'phase1/selection-cams/a.npy': b'selection CAM'}.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    archive = export_run(output, tmp_path / 'export')
    with zipfile.ZipFile(archive) as collection:
        assert 'pseudo/a.png' in collection.namelist()
        assert not any('.npy' in name for name in collection.namelist())
        collection.extractall(tmp_path / 'attached/run')
    restored = tmp_path / 'restored'
    restore_run(tmp_path / 'attached', restored)
    assert (restored / 'phase1/checkpoint.pth').read_bytes() == b'full snapshot'
    assert (restored / 'phase1/variant.yml').read_bytes() == b'config'
    assert (restored / 'pseudo/a.png').read_bytes() == b'pseudo'
    # Local checkpoints win in automatic mode, avoiding rollback to an older attached run.
    (restored / 'phase1/checkpoint.pth').write_bytes(b'newer checkpoint')
    restore_run(tmp_path / 'attached', restored)
    assert (restored / 'phase1/checkpoint.pth').read_bytes() == b'newer checkpoint'
    with pytest.raises(ValueError, match='Fresh-run'):
        restore_run(tmp_path / 'attached', restored, 'none')
    (tmp_path / 'attached/run/phase1/checkpoint.pth').write_bytes(b'corrupt transfer')
    with pytest.raises(ValueError, match='transfer checksum'):
        restore_run(tmp_path / 'attached', tmp_path / 'another-run')


def test_interrupted_restore_finishes_instead_of_treating_partial_copy_as_current_run(tmp_path, monkeypatch):
    source = tmp_path / 'attached/run'
    source.mkdir(parents=True)
    files = {'phase1/checkpoint.pth': b'checkpoint', 'phase1/variant.yml': b'variant'}
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
    (source / 'run-manifest.json').write_text(json.dumps(dict(schema=1, files={
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()})))
    original_copy = kaggle.copy_atomic
    def interrupt(stream, target):
        if target.name == 'variant.yml':
            raise OSError('interrupted transfer')
        original_copy(stream, target)
    monkeypatch.setattr(kaggle, 'copy_atomic', interrupt)
    work = tmp_path / 'restored'
    with pytest.raises(OSError, match='interrupted'):
        restore_run(tmp_path / 'attached', work)
    assert (work / 'phase1/checkpoint.pth').exists()
    monkeypatch.setattr(kaggle, 'copy_atomic', original_copy)
    restore_run(tmp_path / 'attached', work)
    assert (work / 'phase1/variant.yml').read_bytes() == b'variant'
    assert not (work / '.restore-in-progress.json').exists()


def test_prepare_restores_before_installation_writes_new_session_metadata(tmp_path, monkeypatch):
    work, inputs = tmp_path / 'work', tmp_path / 'inputs'
    output = work / 'weaktr-artifacts'
    inputs.mkdir()
    def fake_restore_inputs(*args):
        pass
    def fake_restore_run(*args):
        output.mkdir(parents=True)
        (output / 'environment-freeze.txt').write_text('old environment')
    def fake_install(*args):
        assert (output / 'environment-freeze.txt').read_text() == 'old environment'
        (output / 'environment-freeze.txt').write_text('current environment')
        return work / 'weaktr-env/bin/python'
    monkeypatch.setattr(kaggle, 'restore_inputs', fake_restore_inputs)
    monkeypatch.setattr(kaggle, 'restore_run', fake_restore_run)
    monkeypatch.setattr(kaggle, 'install_environment', fake_install)
    monkeypatch.setattr(kaggle, 'run', lambda *args: None)
    monkeypatch.setattr('sys.argv', ['kaggle.py', 'prepare', '--inputs', str(inputs), '--work', str(work)])
    kaggle.main()
    assert (output / 'environment-freeze.txt').read_text() == 'current environment'
