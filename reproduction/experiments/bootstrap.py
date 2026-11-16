"""Standard-library bootstrap for attached original inputs and final saved runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from reproduction.kaggle import install_environment, restore_inputs, restore_run
from reproduction.prepare import copy_atomic, digest, safe_name

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_RUN_FILES = ('phase1/checkpoint.pth', 'phase1/checkpoint_best_mIoU.pth',
                      'phase2/checkpoint.pth', 'phase2/variant.yml')


def inspect_saved_run(path):
    """Check required artifacts without importing torch or copying data."""
    path = Path(path).resolve()
    archive = path.is_file()
    manifest = None
    try:
        if archive:
            with zipfile.ZipFile(path) as collection:
                manifest = json.loads(collection.read('run-manifest.json'))
                available = set(collection.namelist())
        else:
            manifest_path = path / 'run-manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
            available = {name for name in REQUIRED_RUN_FILES if (path / name).is_file()}
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        return dict(path=path, archive=archive, error=str(error), missing=list(REQUIRED_RUN_FILES))
    if manifest is not None:
        if not isinstance(manifest, dict) or manifest.get('schema') != 1 or not isinstance(manifest.get('files'), dict) or not manifest['files']:
            return dict(path=path, archive=archive, error='Invalid run manifest', missing=list(REQUIRED_RUN_FILES))
        # An active folder may contain later Phase-2 files omitted by an old
        # export manifest. Those are independently hashed before restoration.
        missing = [name for name in REQUIRED_RUN_FILES if name not in available or (archive and name not in manifest['files'])]
        unlisted = [name for name in REQUIRED_RUN_FILES if name in available and name not in manifest['files']]
        effective = dict(manifest['files'])
        if not archive:
            effective.update({name: digest(path / name) for name in unlisted})
        fingerprint = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    else:
        missing = [name for name in REQUIRED_RUN_FILES if name not in available]
        unlisted, fingerprint = [], None
    return dict(path=path, archive=archive, missing=missing, unlisted=unlisted,
                fingerprint=fingerprint, has_manifest=manifest is not None)


def select_saved_run(inputs, resume_from='auto'):
    """Prefer complete manifested exports; an old Phase-1 export is not usable."""
    inputs = Path(inputs).resolve()
    if resume_from != 'auto':
        candidates = [inspect_saved_run(resume_from)]
    else:
        paths = {p.parent for p in inputs.rglob('run-manifest.json')}
        paths.update(p.parent.parent for p in inputs.rglob('checkpoint.pth') if p.parent.name == 'phase2')
        # A final saved notebook may contain an old extracted export AND a newer ZIP.
        for path in inputs.rglob('*.zip'):
            try:
                with zipfile.ZipFile(path) as collection:
                    if 'run-manifest.json' in collection.namelist():
                        paths.add(path)
            except zipfile.BadZipFile:
                continue
        candidates = [inspect_saved_run(path) for path in sorted(paths)]
    complete = [c for c in candidates if not c['missing']]
    manifested = [c for c in complete if c['has_manifest']]
    if manifested:
        complete = manifested
        # Extracted files and ZIP of the same export are one choice, not ambiguity.
        unique = {}
        for candidate in sorted(complete, key=lambda c: (c['archive'], str(c['path']))):
            unique.setdefault(candidate['fingerprint'], candidate)
        complete = list(unique.values())
    if len(complete) == 1:
        selected = complete[0]
        print(f'Selected saved run with required Phase-2 artifacts: {selected["path"]}', flush=True)
        return selected
    if len(complete) > 1:
        raise ValueError('Multiple saved runs contain Phase-2 artifacts. Set RESUME_FROM to the '
                         'completed 100-epoch ZIP/folder: ' + ', '.join(str(c['path']) for c in complete))
    details = []
    for candidate in candidates:
        line = f'{candidate["path"]}: missing required or manifested files {candidate["missing"]}'
        if candidate.get('unlisted'):
            line += f'; files exist but the manifest omits {candidate["unlisted"]} (stale export manifest)'
        if candidate.get('error'):
            line += f'; {candidate["error"]}'
        details.append(line)
    raise ValueError('No usable completed-run input found. These experiments require '
                     'phase2/checkpoint.pth from the 100-epoch run and phase2/variant.yml, '
                     'plus the Phase-1 snapshots. Attach the final reproduction output/ZIP '
                     '(the reviewed final archive was weaktr-resume (1).zip), not only an older '
                     'Phase-1 export. Selection checked before input copying or installation.\n' +
                     ('\n'.join(details) if details else f'No saved runs found under {inputs}'))


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
    baseline = work / 'weaktr-artifacts'
    local = inspect_saved_run(baseline)
    local_ready = not local['missing'] and not local.get('unlisted') and local['has_manifest'] and not (baseline / '.restore-in-progress.json').exists()
    selected = None if local_ready else select_saved_run(inputs, resume_from)
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
    if selected is not None:
        if selected['archive']:
            unpack_manifest_archive(selected['path'], baseline, 'run-manifest.json')
        else:
            restore_run(inputs, baseline, str(selected['path']), required_files=REQUIRED_RUN_FILES)
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
