"""Execute notebook stage cells with a fake launcher to check resume routing."""
import ast
import os
from pathlib import Path
import subprocess

import nbformat
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'WeakTr_Kaggle.ipynb'


def notebook_namespace():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {}
    configuration = ast.parse(notebook.cells[1].source)
    for node in configuration.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('STAGES', 'START_FROM', 'RUN_SMOKE', 'PHASE2_PILOT', 'PHASE2_EPOCHS_THIS_SESSION'):
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name == 'should_run':
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<configuration>', 'exec'), namespace)
    return notebook, namespace


def test_upload_notebook_is_clean_and_valid():
    notebook, namespace = notebook_namespace()
    nbformat.validate(notebook)
    assert namespace['START_FROM'] == 'phase2'
    assert namespace['RUN_SMOKE'] is False
    assert namespace['PHASE2_PILOT'] is False
    assert namespace['PHASE2_EPOCHS_THIS_SESSION'] == 24
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.outputs == []
            assert cell.execution_count is None
            compile(cell.source, '<notebook>', 'exec')


@pytest.mark.parametrize('start,pilot,expected', [
    ('cams', False, ['cams', 'cam-eval', 'pseudo', 'preserve', 'phase2', 'evaluate', 'crf', 'preserve']),
    ('phase2', False, ['phase2', 'evaluate', 'crf', 'preserve']),
    ('evaluate', False, ['evaluate', 'crf', 'preserve']),
    ('cams', True, ['cams', 'cam-eval', 'pseudo', 'preserve', 'phase2-pilot']),
])
def test_resume_stage_sequence(start, pilot, expected):
    notebook, namespace = notebook_namespace()
    namespace.update(START_FROM=start, PHASE2_PILOT=pilot)
    calls = []
    namespace['workflow'] = calls.append
    namespace['phase2_complete'] = lambda: True
    for cell in notebook.cells:
        if 'weaktr-stage' in cell.metadata.get('tags', []):
            exec(compile(cell.source, '<stage-cell>', 'exec'), namespace)
    assert calls == expected
    assert 'phase1' not in calls


def test_incomplete_session_exports_without_running_final_evaluation():
    notebook, namespace = notebook_namespace()
    namespace.update(START_FROM='phase2', PHASE2_PILOT=False, phase2_complete=lambda: False)
    calls = []
    namespace['workflow'] = calls.append
    for cell in notebook.cells:
        if 'weaktr-stage' in cell.metadata.get('tags', []):
            exec(compile(cell.source, '<stage-cell>', 'exec'), namespace)
    assert calls == ['phase2']  # the phase2 launcher exports before returning


@pytest.mark.parametrize('probe_result,skip_install', [(0, True), (1, False), ('denied', False)])
def test_setup_reuses_only_an_importable_environment(tmp_path, monkeypatch, probe_result, skip_install):
    notebook, namespace = notebook_namespace()
    python = tmp_path / 'weaktr-env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('mock interpreter')
    def probe(command, **kwargs):
        if probe_result == 'denied':
            raise PermissionError('unusable interpreter')
        return subprocess.CompletedProcess(command, probe_result, stdout='')
    monkeypatch.setattr(subprocess, 'run', probe)
    calls = []
    namespace.update(START_FROM='phase1', SCRATCH=tmp_path, REPO=tmp_path, os=os, subprocess=subprocess,
                     workflow=lambda stage, **kwargs: calls.append((stage, kwargs)))
    exec(compile(notebook.cells[3].source, '<setup-cell>', 'exec'), namespace)
    assert calls == [('prepare', {'skip_install': skip_install})]


@pytest.mark.parametrize('active_available', [False, True])
def test_timeout_recovery_requires_actual_phase2_artifacts(tmp_path, active_available):
    notebook, namespace = notebook_namespace()
    inputs, work, scratch = (tmp_path / name for name in ('inputs', 'work', 'scratch'))
    inputs.mkdir()
    if active_available:
        source = inputs / 'cancelled-run/weaktr-artifacts'
        for name in ('phase1/checkpoint_best_mIoU.pth', 'phase2/checkpoint.pth', 'pseudo/a.png'):
            p = source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'artifact')
    calls = []
    namespace.update(START_FROM='phase2', RESUME_FROM='auto', INPUTS=inputs, WORK=work,
                     SCRATCH=scratch, REPO=tmp_path, os=os, subprocess=subprocess,
                     workflow=lambda *args, **kwargs: calls.append(args))
    if active_available:
        exec(compile(notebook.cells[3].source, '<setup>', 'exec'), namespace)
        assert namespace['RESUME_FROM'] == str(source)
        assert calls == [('prepare',)]
    else:
        with pytest.raises(RuntimeError, match='older Phase-1 ZIP'):
            exec(compile(notebook.cells[3].source, '<setup>', 'exec'), namespace)
        assert calls == []
