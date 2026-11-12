"""Kaggle entrypoint: prepare uploaded inputs and environment, then run explicit stages."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.prepare import copy_atomic, digest, safe_name

MMCV_WHEELS = 'https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html'


def run(command, env=None):
    print(subprocess.list2cmdline([str(item) for item in command]), flush=True)
    subprocess.run([str(item) for item in command], cwd=ROOT, env=env, check=True)


def python_version(python, env):
    try:
        result = subprocess.run(
            [str(python), '--version'], cwd=ROOT, env=env, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def restore_inputs(inputs, work):
    """Copy the uploaded bundle's contents to local disk, checking every payload hash."""
    inputs, work = Path(inputs).resolve(), Path(work).resolve()
    manifests = sorted(inputs.rglob('weaktr-inputs-manifest.json'))
    if len(manifests) != 1:
        raise ValueError(f'Expected one extracted weaktr-kaggle-inputs.zip under {inputs}; found {len(manifests)} manifests. Attach the prepared dataset, or set --inputs to its directory.')
    manifest_path = manifests[0]
    source_root = manifest_path.parent.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 1 or not manifest.get('files'):
        raise ValueError('Invalid input manifest.')
    for index, (name, expected) in enumerate(manifest['files'].items(), 1):
        relative = safe_name(name)
        source = source_root.joinpath(*relative.parts).resolve()
        target = work.joinpath(*relative.parts).resolve()
        if source_root not in source.parents or work not in target.parents:
            raise ValueError(f'Input path escapes its directory: {name}')
        if target.is_file() and digest(target) == expected:
            continue
        if digest(source) != expected:
            raise ValueError(f'Uploaded input checksum mismatch: {name}')
        with source.open('rb') as stream:
            copy_atomic(stream, target)
        if digest(target) != expected:
            raise ValueError(f'Local copy checksum mismatch: {name}')
        if index % 2000 == 0:
            print(f'Copied and verified {index}/{len(manifest["files"])} files', flush=True)
    (work / 'weaktr-inputs-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def install_environment(work, env):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError('The automatic environment installer targets Kaggle Linux x86_64.')
    work = Path(work)
    python = work / 'weaktr-env/bin/python'
    tools = work / 'weaktr-bootstrap'
    if not (tools / 'uv').is_dir():
        run([sys.executable, '-m', 'pip', 'install', '--target', tools, 'uv==0.5.11'], env)
    uv_env = env.copy()
    uv_env['PYTHONPATH'] = str(tools) + os.pathsep + uv_env.get('PYTHONPATH', '')
    if not (python_version(python, env) or '').startswith('Python 3.10.'):
        run([sys.executable, '-m', 'uv', 'venv', '--clear', '--python', '3.10',
             '--seed', python.parent.parent], uv_env)
    if not (python_version(python, env) or '').startswith('Python 3.10.'):
        raise RuntimeError(f'Could not create a runnable Python 3.10 environment at {python}.')
    run([python, '-m', 'pip', 'install', 'pip<26', 'setuptools<70', 'wheel',
         'Cython==0.29.36', 'numpy==1.23.5'], env)
    run([python, '-m', 'pip', 'install', 'torch==1.13.1+cu117', 'torchvision==0.14.1+cu117',
         '--extra-index-url', 'https://download.pytorch.org/whl/cu117'], env)
    run([python, '-m', 'pip', 'install', '-r', ROOT / 'reproduction/requirements-kaggle.txt',
         '--only-binary=mmcv-full', '--find-links', MMCV_WHEELS], env)
    # pip freeze preserves the exact Git commit resolved by this build.
    run([python, '-m', 'pip', 'install', '--no-build-isolation',
         'git+https://github.com/lucasb-eyer/pydensecrf.git'], env)
    (work / 'weaktr-artifacts').mkdir(parents=True, exist_ok=True)
    with (work / 'weaktr-artifacts/environment-freeze.txt').open('w') as stream:
        subprocess.run([str(python), '-m', 'pip', 'freeze'], stdout=stream, env=env, check=True)
    return python


def restore_run(inputs, output, resume_from='auto'):
    """Prefer this session's checkpoints, otherwise restore a verified saved run."""
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    pending = output / '.restore-in-progress.json'
    if resume_from in ('', 'none'):
        if list(output.glob('phase*/checkpoint.pth')):
            raise ValueError('Fresh-run mode requires an empty run directory; existing checkpoints were preserved.')
        print('Starting a fresh run.', flush=True)
        return None
    if resume_from == 'auto' and not pending.exists() and list(output.glob('phase*/checkpoint.pth')):
        print(f'Resuming checkpoints already on working disk: {output}', flush=True)
        return output
    if resume_from == 'auto':
        candidates = sorted(inputs.rglob('run-manifest.json'))
        if not candidates:
            print('No saved run attached; starting a fresh run.', flush=True)
            return None
        if len(candidates) != 1:
            raise ValueError('Multiple saved runs found. Set RESUME_FROM to the selected run directory.')
        source_root = candidates[0].parent
    else:
        source_root = Path(resume_from).resolve()
    manifest = json.loads((source_root / 'run-manifest.json').read_text())
    if manifest.get('schema') != 1 or not manifest.get('files'):
        raise ValueError('Invalid saved-run manifest.')
    output.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(dict(source=str(source_root))) + '\n')
    for name, expected in manifest['files'].items():
        relative = safe_name(name)
        source = source_root.joinpath(*relative.parts).resolve()
        target = output.joinpath(*relative.parts).resolve()
        if source_root not in source.parents or output not in target.parents:
            raise ValueError(f'Unsafe saved-run path: {name}')
        if digest(source) != expected:
            raise ValueError(f'Saved-run transfer checksum mismatch: {name}')
        if target.exists() and digest(target) == expected:
            continue
        if target.exists():
            raise ValueError(f'Refusing to overwrite a different working artifact: {target}')
        with source.open('rb') as stream:
            copy_atomic(stream, target)
        if digest(target) != expected:
            raise ValueError(f'Restored checkpoint/artifact checksum mismatch: {target}')
    pending.unlink()
    print(f'Restored and verified saved run: {source_root}', flush=True)
    return source_root


def export_run(output, destination):
    """Export checkpoints, configs and pseudo-labels without intermediate CAM collections."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    if not list(output.glob('phase*/checkpoint.pth')):
        raise ValueError('No training checkpoints to preserve yet.')
    if destination == output or output in destination.parents:
        raise ValueError('Resume export must be outside the active run directory.')
    files = {}
    for source in sorted(output.rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(output)
        if source.suffix not in ('.pth', '.yml', '.yaml', '.json', '.txt', '.log') and not (
                relative.parts[0] == 'pseudo' and source.suffix == '.png'):
            continue
        if 'selection-cams' in relative.parts or relative.parts[0] in ('cams', 'seg_prob_ms', 'seg_pred_ms'):
            continue
        target = destination / relative
        with source.open('rb') as stream:
            copy_atomic(stream, target)
        files[relative.as_posix()] = digest(target)
    manifest = dict(schema=1, files=files)
    (destination / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive_path = destination.with_suffix('.zip')
    temporary = archive_path.with_name(archive_path.name + '.part')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        # Archive only the current manifest's files; ignore stale files from a prior export.
        for name in list(files) + ['run-manifest.json']:
            archive.write(destination / name, name)
    temporary.replace(archive_path)
    print(f'Saved run for download/private dataset: {archive_path}', flush=True)
    return archive_path


def pseudo_labels_ready(output):
    output = Path(output)
    record = output / 'pseudo-record.json'
    if not record.is_file() or json.loads(record.read_text()).get('status') != 'command_completed':
        return False
    ids = (ROOT / 'voc12/train_aug_id.txt').read_text().splitlines()
    return all((output / 'pseudo' / (name + '.png')).is_file() for name in ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'phase1-pilot', 'phase1', 'cams',
                                        'pseudo', 'phase2-pilot', 'phase2', 'evaluate', 'crf', 'preserve'])
    parser.add_argument('--inputs', default='/kaggle/input')
    parser.add_argument('--work', default='/kaggle/working')
    parser.add_argument('--skip-install', action='store_true', help='Reuse an existing accepted environment')
    parser.add_argument('--resume-from', default='auto', help='auto, none, or an attached exported-run directory')
    args = parser.parse_args()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    python = work / 'weaktr-env/bin/python'
    data, output, pretrained = work / 'data', work / 'weaktr-artifacts', work / 'weights/deit_small_patch16_224.pth'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'OnlineRetraining'), env.get('PYTHONPATH', '')])
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONHASHSEED'] = '504'
    env['DATASET'] = str(data)
    if args.stage == 'prepare':
        env['CUDA_VISIBLE_DEVICES'] = ''  # No GPU work during preparation, even if enabled accidentally.
        restore_inputs(args.inputs, work)
        # Restore before setup/preflight write this session's records into the run directory.
        restore_run(args.inputs, output, args.resume_from)
        if not args.skip_install:
            python = install_environment(work, env)
        run([python, '-c', 'import main; import segm.train; import mmcv.ops; import pydensecrf.densecrf; print("Both trainer imports and CRF imports passed")'], env)
        run([python, '-m', 'pytest', 'tests', '-q'], env)
        run([python, '-m', 'reproduction.launch', 'preflight', '--data', data,
             '--output', output, '--pretrained', pretrained], env)
        print('PREPARATION PASSED. Optional next step: smoke/pilot. Training uses the original full schedule.', flush=True)
        return
    if not python.is_file():
        raise FileNotFoundError('Run prepare first to create the isolated environment.')
    if args.stage == 'preserve':
        export_run(output, work / 'weaktr-resume')
        return
    env['CUDA_VISIBLE_DEVICES'] = '0'
    if args.stage == 'smoke':
        run([python, '-m', 'reproduction.smoke', '--pretrained', pretrained,
             '--record', output / 'gpu-smoke.json'], env)
        return
    stage = args.stage.replace('-pilot', '')
    if stage in ('cams', 'pseudo') and pseudo_labels_ready(output):
        print('Reusing the saved complete pseudo-label collection; CAM/CRF regeneration is unnecessary.', flush=True)
        return
    command = [python, '-m', 'reproduction.launch', stage, '--data', data,
               '--output', output, '--pretrained', pretrained]
    if stage in ('phase1', 'phase2'):
        checkpoint = output / stage / 'checkpoint.pth'
        if checkpoint.exists():
            command += ['--resume', checkpoint]
        if args.stage.endswith('-pilot'):
            command += ['--stop-after-epoch', '1']
    try:
        run(command, env)
    finally:
        if stage in ('phase1', 'phase2') and list(output.glob('phase*/checkpoint.pth')):
            export_run(output, work / 'weaktr-resume')


if __name__ == '__main__':
    main()
