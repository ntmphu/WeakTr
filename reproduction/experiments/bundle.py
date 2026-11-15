"""Build a small source-only Kaggle attachment, excluding data and model artifacts."""
import json
from pathlib import Path
import subprocess
import zipfile

from reproduction.data import sha256
from reproduction.experiments.core import ROOT


def build(output):
    output = Path(output)
    paths = list(ROOT.glob('*.py'))
    for directory in ('reproduction', 'OnlineRetraining/segm', 'tests'):
        paths += [p for p in (ROOT / directory).rglob('*') if p.is_file()
                  and p.suffix in ('.py', '.yml', '.yaml', '.txt') and '__pycache__' not in p.parts]
    paths += [ROOT / p for p in ('LICENSE', 'AGENTS.md', 'WeakTr_Kaggle.ipynb',
                                'WeakTr_Experiments_Kaggle.ipynb', 'docs/EXPERIMENTS.md')]
    paths += list((ROOT / 'voc12').glob('*_id.txt')) + [ROOT / 'voc12/cls_labels.npy']
    paths += list((ROOT / 'reproduction/configs').glob('*.json'))
    files = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths)) if p.is_file()}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    revision = json.dumps(dict(git_commit=commit, source='source bundle; per-file hashes identify working content'))+'\n'
    import hashlib
    files['source-revision.json'] = hashlib.sha256(revision.encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            if name == 'source-revision.json':
                archive.writestr(name, revision)
            else:
                archive.write(ROOT / name, name)
        archive.writestr('source-manifest.json', json.dumps(dict(schema=1, files=files), indent=2)+'\n')
    print(f'Source bundle: {output.resolve()} ({output.stat().st_size:,} bytes); SHA256={sha256(output)}')
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/weaktr-experiments-source.zip')
    build(parser.parse_args().output)
