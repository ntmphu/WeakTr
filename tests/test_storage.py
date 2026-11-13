from pathlib import Path
import zipfile

import pytest

from reproduction.kaggle import export_run
from reproduction.launch import commands
from reproduction.storage import migrate_temporary_files


def test_migration_keeps_checkpoints_and_labels_in_saved_outputs(tmp_path):
    work, scratch = tmp_path / 'working', tmp_path / 'scratch'
    temporary = ['weaktr-env/bin/python', 'data/voc12/image.jpg', 'weights/imagenet.pth',
                 'weaktr-artifacts/phase1/selection-cams/a.npy', 'weaktr-artifacts/cams/a.npy',
                 'weaktr-artifacts/seg_prob_ms/a.npy']
    durable = ['weaktr-artifacts/phase1/checkpoint.pth', 'weaktr-artifacts/pseudo/a.png',
               'weaktr-resume.zip']
    for name in temporary + durable:
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    migrate_temporary_files(work, scratch)
    migrate_temporary_files(work, scratch)  # retry is harmless
    for name in durable:
        assert (work / name).read_bytes() == name.encode()
    for name in temporary:
        assert not (work / name).exists()
        assert (scratch / name.replace('weaktr-artifacts/', 'artifacts/')).read_bytes() == name.encode()


def test_scratch_cannot_be_inside_saved_outputs(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        migrate_temporary_files(tmp_path, tmp_path / 'scratch')


def test_cam_and_probability_producers_match_consumers_in_scratch(tmp_path):
    data, output, scratch = (tmp_path / name for name in ('data', 'output', 'scratch'))
    def argv(stage):
        return commands(stage, data, output, 'imagenet.pth', scratch=scratch)[0][0]
    def option(stage, name):
        command = argv(stage)
        return Path(command[command.index(name) + 1])
    assert option('phase1', '--cam-npy-dir') == scratch / 'phase1/selection-cams'
    assert option('cams', '--cam-npy-dir') == option('pseudo', '--predict_dir') == scratch / 'cams'
    assert option('pseudo', '--out-dir') == option('phase2', '--ann-dir') == output / 'pseudo'
    assert option('evaluate', '--predict-dir') == option('crf', '--predict-dir') == scratch / 'seg_prob_ms'
    assert option('phase1', '--output_dir') == output / 'phase1'
    assert option('phase2', '--log-dir') == output / 'phase2'


def test_failed_archive_keeps_previous_export_and_cleans_partial_copy(tmp_path, monkeypatch):
    output = tmp_path / 'active'
    (output / 'phase1').mkdir(parents=True)
    checkpoint = output / 'phase1/checkpoint.pth'
    checkpoint.write_bytes(b'checkpoint')
    destination = tmp_path / 'export'
    archive = export_run(output, destination)
    previous = archive.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('No space left on device')
    monkeypatch.setattr(zipfile.ZipFile, 'write', fail)
    with pytest.raises(OSError, match='No space'):
        export_run(output, destination)
    assert archive.read_bytes() == previous
    assert checkpoint.read_bytes() == b'checkpoint'
    assert not archive.with_name(archive.name + '.part').exists()
