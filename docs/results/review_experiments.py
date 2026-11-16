"""Read-only CPU audit of the delivered Kaggle experiment archive."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.experiments.core import CLASSES

OUT = ROOT / 'docs/results'
OUT.mkdir(exist_ok=True)
archive_path = ROOT / 'weaktr-experiments.zip'
started = time.monotonic()
arms = ('hard', 'soft', 'uncertainty-control', 'uncertainty')
receipt = {}
with zipfile.ZipFile(archive_path) as z:
    def read(name):
        return json.loads(z.read(name))

    names = z.namelist()
    assert len(names) == len(set(names)), 'Duplicate ZIP entries'
    manifest = read('experiment-manifest.json')
    assert manifest['schema'] == 1
    assert set(names) == set(manifest['files']) | {'experiment-manifest.json'}
    print('Verifying', len(manifest['files']), 'payload hashes', flush=True)
    for index, (name, expected) in enumerate(manifest['files'].items(), 1):
        h = hashlib.sha256()
        with z.open(name) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        assert h.hexdigest() == expected, name
        if index % 10000 == 0:
            print('Verified', index, flush=True)
    report = (ROOT / 'experiment-report.txt').read_text(encoding='utf-8-sig')
    assert z.read('experiment-report.md').decode().replace('\r\n', '\n') == report
    identity = read('identity.json')
    stages = [read(n) for n in names if n.startswith('records/')]
    assert all(d['status'] == 'command_completed' for d in stages)
    assert all(d['git_commit'] == 'a62033a7d69a5789ec9e278ae1c1b51393a2c383' for d in stages)
    assert identity['config']['smoke_steps'] == 0
    assert identity['checkpoint_sha256']['phase2/checkpoint.pth'] == '891e0cbcbcabdd695d9a56ef7bce9830a8d1b3e3a77752795cfff4ed0a4b67ba'
    ids = (ROOT / 'voc12/val_id.txt').read_text().split()
    assert len(ids) == 1449
    finals = {}
    per_image = {}
    initial = None
    for arm in arms:
        d = read(f'training/{arm}/identity.json')
        done = read(f'training/{arm}/completed.json')
        assert d['config'] == identity['config']
        assert d['source_sha256'] == identity['checkpoint_sha256']['phase2/checkpoint.pth']
        assert done['status'] == 'completed' and done['completed_epochs'] == 3
        assert done['checkpoint_sha256'] == manifest['files'][f'training/{arm}/checkpoint.pth']
        this_initial = read(f'training/{arm}/initial.json')
        if initial is None:
            initial = this_initial
        assert this_initial['confusion_matrix'] == initial['confusion_matrix']
        epochs = [read(f'training/{arm}/epoch-{i}.json') for i in (1, 2, 3)]
        assert all(e['epoch'] == i and not e['smoke_only'] and e['batches'] == 2646
                   and e['scores']['images'] == 1449 for i, e in enumerate(epochs, 1))
        scores = epochs[-1]['scores']
        prediction_names = {n for n in names if n.startswith(f'training/{arm}/predictions/')}
        assert prediction_names == {f'training/{arm}/predictions/{name}.png' for name in ids}
        print('Checking saved validation predictions:', arm, flush=True)
        matrices = []
        for name in ids:
            with Image.open(ROOT / f'data/voc12/VOCdevkit/VOC2012/SegmentationClassAug/{name}.png') as im:
                truth = np.array(im)
            with Image.open(io.BytesIO(z.read(f'training/{arm}/predictions/{name}.png'))) as im:
                pred = np.array(im)
            assert pred.shape == truth.shape
            valid = (truth < 21) & (pred < 21)
            matrices.append(np.bincount(21 * truth[valid].astype('int64') + pred[valid], minlength=441).reshape(21, 21))
        matrices = np.stack(matrices)
        assert np.array_equal(matrices.sum(0), np.asarray(scores['confusion_matrix']))
        tp = np.diagonal(matrices, axis1=1, axis2=2)
        union = matrices.sum(1) + matrices.sum(2) - tp
        per_image[arm] = (tp, union)
        finals[arm] = dict(initial_miou=this_initial['miou'], final_miou=scores['miou'],
            delta_from_start=scores['miou']-this_initial['miou'],
            epochs=[dict(epoch=e['epoch'], miou=e['scores']['miou'],
                         train_seconds=e['train_seconds'], validation_seconds=e['scores']['elapsed_seconds'],
                         clipped_fraction=e['clipped_fraction']) for e in epochs],
            class_iou=scores['class_iou'], source_identity=d, completion=done)
    comparisons = {}
    rng = np.random.default_rng(504)
    # Resample the SAME image indices for each arm in a matched pair.
    weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=2000)
    for control, treatment in [('hard', 'soft'), ('uncertainty-control', 'uncertainty')]:
        values = []
        for arm in (control, treatment):
            tp, union = per_image[arm]
            sampled_tp, sampled_union = weights @ tp, weights @ union
            values.append(100*np.nanmean(np.divide(sampled_tp, sampled_union,
                out=np.full(sampled_tp.shape, np.nan), where=sampled_union != 0), axis=1))
        delta = finals[treatment]['final_miou'] - finals[control]['final_miou']
        interval = np.percentile(values[1]-values[0], [2.5, 97.5]).tolist()
        class_deltas = {c: finals[treatment]['class_iou'][c]-finals[control]['class_iou'][c] for c in CLASSES}
        comparisons[treatment+'-minus-'+control] = dict(delta_pp=delta,
            paired_image_bootstrap_percentile95_pp=interval, class_delta_pp=class_deltas)
        print(treatment, 'minus', control, delta, 'image bootstrap 95%', interval, flush=True)
    cam_results = {}
    for split in ('dev', 'holdout'):
        scores = read(f'cams/{split}-scores.json')['arms']
        b, a = scores['baseline'], scores['agreement']
        matrix, selected = np.array(b['confusion_matrix']), np.array(a['confusion_matrix'])
        rejected = matrix-selected
        assert (rejected >= 0).all(), 'Filtering must only remove existing labels'
        cam_results[split] = dict(arms={arm: {k: s[k] for k in ('miou','pixel_accuracy','images','coverage_percent','class_coverage','class_iou')}
                                      for arm, s in scores.items()},
            rejected_pixel_accuracy=100*np.trace(rejected)/rejected.sum(),
            rejected_valid_pixels=int(rejected.sum()),
            boundary_valid_pixels=int(np.array(b['boundary_band']['confusion_matrix']).sum()),
            total_valid_pixels=int(matrix.sum()))
    clipping = read('clipping/scores.json')
    class_rejection = {c: dict(correct_rejection_percent=100*v[1]/max(1,v[0]+v[1]),
                               incorrect_rejection_percent=100*v[3]/max(1,v[2]+v[3]), counts=v)
                       for c,v in clipping['per_gt_class'].items()}
    full = read('uncertainty-full/completed.json')
    assert full['status'] == 'completed'
    train_ids = set((ROOT / 'voc12/train_aug_id.txt').read_text().split())
    for arm in ('baseline','agreement'):
        payload_ids = {Path(n).stem for n in names if n.startswith(f'uncertainty-full/trainaug/{arm}/') and n.endswith('.png')}
        assert payload_ids == train_ids and len(payload_ids) == 10582
        fingerprint = hashlib.sha256()
        for image_id in (ROOT / 'voc12/train_aug_id.txt').read_text().splitlines():
            name = f'uncertainty-full/trainaug/{arm}/{image_id}.png'
            fingerprint.update((image_id+':'+manifest['files'][name]+'\n').encode())
        training_arm = 'uncertainty-control' if arm == 'baseline' else 'uncertainty'
        assert fingerprint.hexdigest() == finals[training_arm]['source_identity']['pseudo_sha256']
    split_dev = set(z.read('splits/dev.txt').decode().split())
    split_holdout = set(z.read('splits/holdout.txt').decode().split())
    assert len(split_dev) == 100 and len(split_holdout) == 200 and not split_dev & split_holdout
    assert not (split_dev | split_holdout) & set(ids)
    for n in ['figures/2007_003178.png','figures/2008_006920.png']:
        p = ROOT / 'artifacts/experiment-review' / Path(n).name
        p.write_bytes(z.read(n))
    first = min(s['started_unix'] for s in stages)
    last = max(s['started_unix']+s['elapsed_seconds'] for s in stages)
    receipt = dict(archive=str(archive_path.name), archive_bytes=archive_path.stat().st_size,
        verified_payloads=len(manifest['files']), zip_members=len(names),
        report_matches_archive=True, gpu_prediction_scores_recomputed=True,
        git_commit=stages[0]['git_commit'], seed=identity['config']['seed'],
        gpu=stages[0]['gpu'], torch=stages[0]['torch'], cuda=stages[0]['cuda'], identity=identity,
        source_checkpoint_sha256=identity['checkpoint_sha256']['phase2/checkpoint.pth'],
        stages=[{k:s[k] for k in ('stage','arm','status','elapsed_seconds','started_unix')} for s in stages],
        core_stage_span_seconds=last-first,
        full_label_generation_seconds=full['elapsed_seconds'], full_label_count=10582,
        validation_protocol=initial['protocol'], training=finals, comparisons=comparisons,
        bootstrap=dict(replicates=2000,seed=504,unit='paired VOC validation image',
                       caveat='Fixed trained models only. Does not measure variation across training seeds or account for repeated exploratory comparisons.'),
        cams=cam_results,
        clipping={k:clipping[k] for k in ('totals','columns','correct_label_rejection_percent',
            'incorrect_label_rejection_percent','accuracy_of_rejected_percent','accuracy_of_kept_percent')},
        clipping_per_class=class_rejection,
        caveats=['One seed and three fine-tuning epochs, not full retraining.',
                 'CAM holdout means disjoint diagnostic training images, not unseen model-validation data.',
                 'Selective CAM metrics omit rejected pixels and are not complete-mask mIoU.',
                 'Uncertainty pair uses raw CAM labels and valid-pixel loss reduction; clipping pair uses original CRF labels and all-pixel loss reduction.',
                 'Boundary diagnostic excludes VOC ignore transitions and covers very few pixels.'])

h = hashlib.sha256()
with archive_path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        h.update(chunk)
receipt['archive_sha256'] = h.hexdigest()
(OUT/'EXPERIMENT_REVIEW_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,3,figsize=(14,4.3))
colors = {'hard':'#236795','soft':'#b14a38','uncertainty-control':'#936ab0','uncertainty':'#218766'}
for ax,pair,title in [(axes[0],('hard','soft'),'Clipping: residual gradients did not help'),
                      (axes[1],('uncertainty-control','uncertainty'),'Filtering reduced raw-label degradation')]:
    ax.axhline(initial['miou'],color='#999999',ls='--',lw=1,label='Starting checkpoint')
    for arm in pair:
        y=[initial['miou']]+[e['miou'] for e in finals[arm]['epochs']]
        ax.plot(range(4),y,marker='o',color=colors[arm],label=arm)
        ax.annotate(f'{y[-1]:.3f}',(3,y[-1]),xytext=(4,4),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Fine-tuning epoch',ylabel='VOC val mIoU (%)',xticks=range(4),xlim=(-.1,3.6),title=title)
    ax.legend(fontsize=8,loc='lower left');ax.grid(alpha=.2)
holdout=cam_results['holdout']['arms']
bars=axes[2].bar(['Raw baseline','Feature gate','Agreement\n(selective)'],
    [holdout[a]['miou'] for a in ('baseline','gated','agreement')],color=['#236795','#b14a38','#218766'])
axes[2].bar_label(bars,fmt='%.3f',padding=3)
axes[2].set(ylim=(0,82),ylabel='Training diagnostic mIoU (%)',title='CAM diagnostic: different coverage')
axes[2].text(.5,.08,'Coverage: 100% / 100% / 96.47%',transform=axes[2].transAxes,ha='center',fontsize=9)
fig.suptitle('WeakTr exploratory pilots: one seed, three epochs; matched comparisons only',fontsize=13)
fig.tight_layout()
fig.savefig(OUT/'experiment-results.png',dpi=180)
plt.close(fig)
print('Audit complete:',len(manifest['files']),'verified payloads;',round(time.monotonic()-started,1),'seconds',flush=True)
