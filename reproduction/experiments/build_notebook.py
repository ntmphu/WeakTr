"""Generate the standalone Kaggle experiment notebook from reviewable cells."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[2]


def build():
    cells = []
    def markdown(text):
        cells.append(nb.v4.new_markdown_cell(text))
    def code(text, stage=False):
        cell = nb.v4.new_code_cell(text.strip()+'\n')
        if stage:
            cell.metadata['experiment_stage'] = True
        cells.append(cell)
    markdown('''# WeakTr: three short weakness experiments

Attach your **original private VOC/ImageNet input bundle** and the **completed 100-epoch saved run** (ZIP or extracted folder). Internet on; one T4. Optional: attach `weaktr-experiments-source.zip` to run the exact delivered source without Git access. To resume pilots, attach the prior `weaktr-experiments.zip` too.

For an unattended run, attach inputs, configure this notebook, and choose Save Version / Save & Run All directly; CPU preparation/tests run automatically inside that job. Save & Run All creates a fresh runtime and repeats transfers even if interactive preparation already passed. Alternatively, run all cells in the same interactive session and download the exported ZIP before ending it.

The overnight default runs frozen diagnostics, two matched three-epoch clipping arms, full uncertainty-label generation, and two matched three-epoch uncertainty arms. Set RUN_FULL_UNCERTAINTY=False for the shorter diagnostic/clipping run. No original Phase-1/Phase-2 retraining is launched.

Original result: 72.865% CRF VOC val. Pilot validation uses a separate, single-scale/no-CRF protocol; compare its matched arms. Runtime/improvement of the new experiments is not yet measured. Read `docs/EXPERIMENTS.md`.''')
    code('''# Single configuration cell
from pathlib import Path
import hashlib, json, os, subprocess, sys, zipfile

if not Path('/kaggle').is_dir():
    raise RuntimeError('Use Kaggle or its remote Jupyter kernel.')
INPUTS = Path('/kaggle/input')
WORK = Path('/kaggle/working')
SCRATCH = Path('/kaggle/tmp/weaktr-runtime')
REPO = WORK / 'WeakTr-experiments-code'
RESUME_FROM = 'auto'  # Or the exact completed original saved-run ZIP/folder.
EXPERIMENT_RESUME = 'auto'  # auto | none | exact experiment ZIP/folder
RUN_CLIPPING_TRAINING = True
RUN_FULL_UNCERTAINTY = True  # Overnight: generate all trainaug labels + two matched 3-epoch arms.

SETTINGS = dict(
    schema=1, seed=504, data=str(SCRATCH / 'data'),
    baseline=str(WORK / 'weaktr-artifacts'), output=str(WORK / 'weaktr-experiments'),
    cam_dev_images=100, cam_holdout_images=200, diagnostic_images=64,
    scales=[1.0, 1.2], background_threshold=0.43, agreement=0.75,
    soft_alpha=0.1, epochs=3, learning_rate=1e-5, encoder_lr_scale=0.1,
    batch_size=4, num_workers=2, start_value=1.2, clip_patch_size=120,
    train_epochs_this_session=3, smoke_steps=0, example_images=8)
# To cap each training call, lower train_epochs_this_session, save outputs, then resume.
# Keep settings unchanged on resume. Use a NEW output directory for a new experiment.
''')
    markdown('## Load code and prepare CPU environment/data\nA source bundle is preferred when attached; otherwise clone/update the reproduction branch. Original artifact transfer hashes are checked. Setup runs CPU tests and repairs an unusable environment automatically.')
    code('''bundles = sorted(INPUTS.rglob('weaktr-experiments-source.zip'))
source_manifests = sorted(INPUTS.rglob('source-manifest.json'))
if len(bundles) + len(source_manifests) > 1:
    raise RuntimeError('Attach one experiment source bundle or extracted source dataset.')
if bundles or source_manifests:
    archive = zipfile.ZipFile(bundles[0]) if bundles else None
    source_root = source_manifests[0].parent.resolve() if source_manifests else None
    try:
        manifest = json.loads(archive.read('source-manifest.json') if archive else source_manifests[0].read_text())
        if manifest.get('schema') != 1 or not manifest.get('files'):
            raise ValueError('Invalid source manifest')
        for name, expected in manifest['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or '\\\\' in name or ':' in name:
                raise ValueError('Unsafe source archive path')
            target = (REPO / relative).resolve()
            if REPO.resolve() not in target.parents:
                raise ValueError('Source path escapes checkout')
            if source_root is not None and source_root not in (source_root / relative).resolve().parents:
                raise ValueError('Source symlink escapes dataset')
            content = archive.read(name) if archive else (source_root / relative).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Source checksum mismatch: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    finally:
        if archive is not None:
            archive.close()
else:
    if not (REPO / '.git').exists():
        subprocess.run(['git', 'clone', '--branch', 'reproduction/voc-deit-small',
                        'https://github.com/ntmphu/WeakTr.git', str(REPO)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only', 'origin',
                        'reproduction/voc-deit-small'], check=True)
if not (REPO / 'reproduction/experiments/run.py').exists():
    raise RuntimeError('Checkout lacks experiments; attach the delivered source bundle.')
subprocess.run([sys.executable, '-m', 'reproduction.experiments.bootstrap',
                '--inputs', str(INPUTS), '--work', str(WORK), '--scratch', str(SCRATCH),
                '--resume-from', RESUME_FROM], cwd=REPO, check=True)
PYTHON = SCRATCH / 'weaktr-env/bin/python'
ENV = os.environ.copy()
ENV.update(DATASET=str(SCRATCH / 'data'), CUDA_VISIBLE_DEVICES='0',
           PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / 'OnlineRetraining')]),
           PYTHONHASHSEED=str(SETTINGS['seed']), PYTHONUNBUFFERED='1')
CONFIG = WORK / 'weaktr-experiment-config.json'
CONFIG.write_text(json.dumps(SETTINGS, indent=2)+'\\n')
''')
    markdown('## Restore pilots and verify the completed reproduction\nThe original run must contain 60/60 and 100/100 epochs plus the original pseudo-label fingerprint. A prior pilot export is restored only when the working experiment directory is empty. Training resumes from completed epochs.')
    code('''out = Path(SETTINGS['output'])
if EXPERIMENT_RESUME != 'none' and (not (out / 'identity.json').exists() or (out / '.experiment-restore-in-progress.json').exists()):
    if EXPERIMENT_RESUME == 'auto':
        folders = sorted(INPUTS.rglob('experiment-manifest.json'))
        candidates = [p.parent for p in folders]
        if not candidates:
            for p in sorted(INPUTS.rglob('*.zip')):
                with zipfile.ZipFile(p) as z:
                    if 'experiment-manifest.json' in z.namelist():
                        candidates.append(p)
    else:
        candidates = [Path(EXPERIMENT_RESUME)]
    if len(candidates) > 1:
        raise RuntimeError('Multiple pilot exports found; set EXPERIMENT_RESUME explicitly.')
    if candidates:
        script = 'from reproduction.experiments.core import restore; import sys; restore(sys.argv[1], sys.argv[2])'
        subprocess.run([str(PYTHON), '-c', script, str(candidates[0]), str(out)], cwd=REPO, env=ENV, check=True)

def run(stage, arm=None):
    command = [str(PYTHON), '-m', 'reproduction.experiments.run', stage, '--config', str(CONFIG)]
    if arm is not None:
        command += ['--arm', arm]
    try:
        subprocess.run(command, cwd=REPO, env=ENV, check=True)
    finally:
        if stage != 'preserve' and (out / 'identity.json').exists():
            subprocess.run([str(PYTHON), '-m', 'reproduction.experiments.run', 'preserve',
                            '--config', str(CONFIG)], cwd=REPO, env=ENV, check=True)

run('preflight')
''')
    markdown('## Optional small GPU check\nRun this cell interactively when you want an early forward/backward check. It does not start training pilots. CUDA recovery equivalence remains a separate measurement.')
    code('''# Optional: uncomment to check both original models in this runtime.
# subprocess.run([str(PYTHON), '-m', 'reproduction.smoke', '--pretrained',
#                 str(SCRATCH / 'weights/deit_small_patch16_224.pth'),
#                 '--record', str(out / 'optional-gpu-smoke.json')], cwd=REPO, env=ENV, check=True)
print('Experiment operators passed CPU tests during setup; new GPU pilot evidence is pending.')
''')
    markdown('## Frozen CAM pilots: propagation and uncertainty\n100 fixed development + 200 disjoint diagnostic images from VOC train. Same checkpoint, views, foreground tags and threshold in both propagation arms. Agreement filtering reports coverage and conditional accuracy; it does not establish improved full segmentation. Outputs resume per image with hashes.')
    code("run('cams')", True)
    markdown('## Clipping diagnostic\n64 training images, fixed 480 resize, configured batches. Compare the rejection of correct and incorrect pseudo-labels using GT only for analysis. This frozen-final-model diagnostic does not describe early training.')
    code("run('clipping')", True)
    markdown('## Matched short fine-tuning: hard versus residual clipping\nBoth arms initialize from the completed Phase-2 weights, reset SGD, and use constant LR 1e-5 (encoder multiplier 0.1). Last planned pilot epoch is the reported result. Existing official training/evaluation scores are preserved. Rough training estimate: 2–2.5 hours for six epochs, plus setup/evaluation; actual pilot runtime is unmeasured.')
    code("""if RUN_CLIPPING_TRAINING:
    run('train', 'hard')
    run('train', 'soft')
else:
    print('Clipping training disabled; diagnostics still run.')""", True)
    markdown('## Full uncertainty-label training (enabled for overnight run)\nRUN_FULL_UNCERTAINTY=True regenerates all 10,582 raw pseudo-labels and runs two matched training arms. The control uses regenerated raw labels; the treatment ignores view disagreement. Both normalize by valid pixels. This adds a full CAM pass and six training epochs. Total overnight runtime is unmeasured and may exceed the session limit; save/export outputs and resume if needed.')
    code("""if RUN_FULL_UNCERTAINTY:
    run('uncertainty-labels')
    run('train', 'uncertainty-control')
    run('train', 'uncertainty')
else:
    print('Full uncertainty training skipped; selective-label pilot is included above.')""", True)
    markdown('## Report and export\nDownload experiment-report.md, experiment-summary.json, figures/, and weaktr-experiments.zip. Save the Kaggle notebook version with outputs. The ZIP includes pilot checkpoints, predictions, per-class diagnostics, source identities, and transfer hashes. Attach it next session for automatic restore.')
    code("run('report')\nrun('preserve')", True)
    code("""from IPython.display import display, Markdown, FileLink
display(Markdown((out / 'experiment-report.md').read_text()))
display(FileLink(str(out.with_suffix('.zip'))))
""")
    notebook = nb.v4.new_notebook(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.10'),
        kaggle=dict(accelerator='gpu', isInternetEnabled=True)))
    nb.validate(notebook)
    target = ROOT / 'WeakTr_Experiments_Kaggle.ipynb'
    nb.write(notebook, target)
    print(target)


if __name__ == '__main__':
    build()
