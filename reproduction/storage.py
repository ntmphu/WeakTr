"""Keep rebuildable Kaggle files separate from durable experiment outputs."""
from pathlib import Path
import shutil


def migrate_temporary_files(work, scratch):
    """Relocate only known temporary paths; never move checkpoints or pseudo-labels."""
    work, scratch = Path(work).resolve(), Path(scratch).resolve()
    if work == scratch or work in scratch.parents or scratch in work.parents:
        raise ValueError('Scratch and saved-output directories must be separate.')
    scratch.mkdir(parents=True, exist_ok=True)
    paths = [(name, name) for name in ('weaktr-env', 'weaktr-bootstrap', 'data', 'weights',
                                      'weaktr-inputs-manifest.json')]
    paths += [('weaktr-artifacts/' + name, 'artifacts/' + name) for name in
              ('phase1/selection-cams', 'cams', 'seg_prob_ms')]
    for old, new in paths:
        source, target = work / old, scratch / new
        if not source.exists() or target.exists():
            continue
        if source.is_symlink() or work not in source.resolve().parents:
            raise ValueError(f'Refusing to relocate a temporary path outside the work directory: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        # rename is cheap on a shared filesystem; move also supports a separate scratch mount.
        shutil.move(str(source), str(target))
        print(f'Relocated temporary files: {source} -> {target}', flush=True)


def report_storage(work, scratch):
    for label, root in [('saved outputs', Path(work)), ('temporary files', Path(scratch))]:
        size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        usage = shutil.disk_usage(root)
        print(f'Storage ({label}): {size / 2**30:.2f} GiB in {root}; '
              f'filesystem free {usage.free / 2**30:.2f} GiB', flush=True)
