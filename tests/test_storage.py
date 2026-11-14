from pathlib import Path
import sys
import zipfile

import pytest

from reproduction.kaggle import export_run
from reproduction import kaggle
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
    assert option('cam-eval', '--predict_dir') == scratch / 'cams'
    assert option('cam-eval', '--logfile') == output / 'cam-eval.txt'
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


@pytest.mark.parametrize('completed,expected_stop', [(29, 53), (77, 100), (100, 100)])
def test_phase2_session_cap_preserves_resume_and_exports(tmp_path, monkeypatch, completed, expected_stop):
    work, scratch = tmp_path / 'working', tmp_path / 'scratch'
    checkpoint = work / 'weaktr-artifacts/phase2/checkpoint.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    python = scratch / 'weaktr-env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_bytes(b'python')
    monkeypatch.setattr(kaggle, 'completed_phase2_epochs', lambda *args: completed)
    monkeypatch.setattr(kaggle, 'report_storage', lambda *args: None)
    calls, exports = [], []
    monkeypatch.setattr(kaggle, 'run', lambda command, env: calls.append([str(x) for x in command]))
    monkeypatch.setattr(kaggle, 'export_run', lambda *args: exports.append(args))
    monkeypatch.setattr('sys.argv', ['kaggle.py', 'phase2', '--work', str(work), '--scratch', str(scratch),
                                    '--epochs-this-session', '24'])
    kaggle.main()
    assert calls[0][calls[0].index('--stop-after-epoch') + 1] == str(expected_stop)
    assert calls[0][calls[0].index('--resume') + 1] == str(checkpoint)
    assert len(exports) == 1


def test_completed_phase2_epochs_reads_trusted_snapshot_on_cpu(tmp_path):
    import torch
    checkpoint = tmp_path / 'checkpoint.pth'
    assert kaggle.completed_phase2_epochs(sys.executable, checkpoint) == 0
    torch.save({'stage': 'phase2', 'format_version': 1, 'epoch': 28}, checkpoint)
    assert kaggle.completed_phase2_epochs(sys.executable, checkpoint) == 29
