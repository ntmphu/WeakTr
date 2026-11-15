"""Standard-library bootstrap for attached original inputs and final saved runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from reproduction.kaggle import install_environment, restore_inputs, restore_run
from reproduction.prepare import copy_atomic, digest, safe_name

ROOT = Path(__file__).resolve().parents[2]


def unpack_manifest_archive(archive, destination, manifest_name):
    """Extract only manifested payloads, validating names and SHA256s atomically."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as collection:
        if len(collection.namelist()) != len(set(collection.namelist())):
            raise ValueError('Duplicate archive member names')
        manifest = json.loads(collection.read(manifest_name))
        if manifest.get('schema') != 1 or not manifest.get('files'):
            raise ValueError('Invalid archive manifest')
        for name, expected in manifest['files'].items():
            relative = safe_name(name)
            target = destination.joinpath(*relative.parts).resolve()
            if destination not in target.parents:
                raise ValueError('Archive path escapes destination')
            if target.exists():
                if digest(target) != expected:
                    raise ValueError(f'Existing payload differs: {target}')
                continue
            with collection.open(name) as stream:
                copy_atomic(stream, target)
            if digest(target) != expected:
                target.unlink()
                raise ValueError(f'Archive checksum mismatch: {name}')
        destination.mkdir(parents=True, exist_ok=True)
        (destination / manifest_name).write_text(json.dumps(manifest, indent=2)+'\n')


def find_archive(inputs, manifest_name):
    candidates = []
    for path in sorted(Path(inputs).rglob('*.zip')):
        try:
            with zipfile.ZipFile(path) as collection:
                if manifest_name in collection.namelist():
                    candidates.append(path)
        except zipfile.BadZipFile:
            continue
    if len(candidates) != 1:
        raise ValueError(f'Expected one archive with {manifest_name}; found {candidates}. '
                         'Attach one input bundle and one completed saved run, or set explicit paths.')
    return candidates[0]


def prepare(inputs, work, scratch, resume_from='auto', skip_install=False):
    inputs, work, scratch = map(lambda p: Path(p).resolve(), (inputs, work, scratch))
    work.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', WEAKTR_RUNTIME=str(scratch),
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / 'OnlineRetraining')]),
               PYTHONUNBUFFERED='1')
    manifests = list(inputs.rglob('weaktr-inputs-manifest.json'))
    if len(manifests) == 1:
        restore_inputs(manifests[0].parent, scratch)
    elif not manifests:
        archive = find_archive(inputs, 'weaktr-inputs-manifest.json')
        unpack_manifest_archive(archive, scratch, 'weaktr-inputs-manifest.json')
    else:
        raise ValueError('Multiple input manifests; point INPUTS at one selected dataset tree')
    baseline = work / 'weaktr-artifacts'
    if not (baseline / 'run-manifest.json').exists() or (baseline / '.restore-in-progress.json').exists():
        if resume_from != 'auto':
            source = Path(resume_from)
            if source.is_file():
                unpack_manifest_archive(source, baseline, 'run-manifest.json')
            else:
                restore_run(inputs, baseline, str(source))
        else:
            manifests = list(inputs.rglob('run-manifest.json'))
            if len(manifests) == 1:
                restore_run(inputs, baseline, str(manifests[0].parent))
            elif not manifests:
                unpack_manifest_archive(find_archive(inputs, 'run-manifest.json'), baseline, 'run-manifest.json')
            else:
                raise ValueError('Multiple saved runs; set RESUME_FROM to the completed run folder/ZIP')
    # Verify an existing working baseline as well as a newly transferred one.
    manifest_path = baseline / 'run-manifest.json'
    if not manifest_path.exists():
        raise ValueError('Completed saved run needs run-manifest.json to verify transfer hashes')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        relative = safe_name(name)
        path = baseline.joinpath(*relative.parts).resolve()
        if baseline.resolve() not in path.parents or digest(path) != expected:
            raise ValueError(f'Baseline checksum mismatch: {name}')
    python = scratch / 'weaktr-env/bin/python'
    usable = False
    if python.exists():
        try:
            probe = subprocess.run([str(python), '-c',
                'import sys; assert sys.version_info[:2]==(3,10); import main; import segm.train; import mmcv.ops; import pydensecrf.densecrf'],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            usable = probe.returncode == 0
        except OSError:
            pass
    if not usable:
        if skip_install:
            raise RuntimeError('Existing environment failed import checks; rerun without --skip-install')
        python = install_environment(work / 'experiment-setup', env)
    subprocess.run([str(python), '-m', 'pytest', 'tests', '-q'], cwd=ROOT, env=env, check=True)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', default='/kaggle/input')
    parser.add_argument('--work', default='/kaggle/working')
    parser.add_argument('--scratch', default='/kaggle/tmp/weaktr-runtime')
    parser.add_argument('--resume-from', default='auto')
    parser.add_argument('--skip-install', action='store_true')
    args = parser.parse_args()
    print('Prepared interpreter:', prepare(**vars(args)))


if __name__ == '__main__':
    main()
