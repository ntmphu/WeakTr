"""Small machine-readable records and verified checkpoint transfers."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from reproduction.data import sha256


def write_record(path, config, checkpoints=()):
    import torch
    root = Path(__file__).resolve().parents[1]
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']}
    record = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  config=config, python=sys.version, platform=platform.platform(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  packages=packages, timestamp_unix=time.time(),
                  dataset_lists={str(p.relative_to(root)): sha256(p) for p in (root / 'voc12').glob('*_id.txt')},
                  checkpoints={str(Path(p).resolve()): sha256(p) for p in checkpoints})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(record, indent=2, default=str) + '\n'
    Path(path).write_text(content)
    # Preserve earlier session provenance when resuming into the same directory.
    Path(path).with_name(Path(path).stem + '-' + str(time.time_ns()) + Path(path).suffix).write_text(content)
    return record


def transfer_manifest(directory, verify=False):
    directory = Path(directory)
    manifest = directory / 'checkpoint-sha256.json'
    if verify:
        hashes = json.loads(manifest.read_text())
        if not hashes:
            raise ValueError('Empty checkpoint transfer manifest.')
        for name, digest in hashes.items():
            target = (directory / name).resolve()
            if directory.resolve() not in target.parents or sha256(target) != digest:
                raise ValueError(f'Checkpoint transfer mismatch: {name}')
    else:
        hashes = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*.pth')}
        if not hashes:
            raise ValueError('No checkpoints to preserve.')
        manifest.write_text(json.dumps(hashes, indent=2) + '\n')
    return hashes
