"""CPU analysis of saved Phase-1 CAM predictions and original pseudo-labels.

Regenerated FP32 raw predictions are compared descriptively with original
AMP-generated/CRF-processed labels. This is not a freshly matched CRF ablation.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.experiments.core import CLASSES, metrics

OUT = ROOT / 'docs/results'
DATA = ROOT / 'data/voc12/VOCdevkit/VOC2012'
started = time.monotonic()
train_ids = (ROOT / 'voc12/train_id.txt').read_text().splitlines()
all_ids = (ROOT / 'voc12/train_aug_id.txt').read_text().splitlines()
train_set = set(train_ids)


def accumulator():
    return dict(raw=np.zeros((21,21), dtype=np.int64), pseudo=np.zeros((21,21), dtype=np.int64),
                transitions=np.zeros(4,dtype=np.int64), images=0,
                image_class_zero_tp=np.zeros(3,dtype=np.int64),
                absent_predictions=np.zeros(3,dtype=np.int64),
                regions={name:np.zeros((2,3),dtype=np.int64) for name in ('near_label_edge','interior')})


def confusion(pred, truth):
    valid=truth < 21
    return np.bincount(21*truth[valid].astype('int64')+pred[valid],minlength=441).reshape(21,21)


def error_types(matrix):
    return dict(background_to_foreground=int(matrix[0,1:].sum()),
                foreground_to_background=int(matrix[1:,0].sum()),
                foreground_wrong_class=int(matrix[1:,1:].sum()-matrix[1:,1:].trace()))


with zipfile.ZipFile(ROOT/'weaktr-resume (1).zip') as original, zipfile.ZipFile(ROOT/'weaktr-experiments.zip') as experiments:
    old_manifest=json.loads(original.read('run-manifest.json'))['files']
    new_manifest=json.loads(experiments.read('experiment-manifest.json'))['files']
    stage_id=json.loads(experiments.read('uncertainty-full/identity.json'))
    assert stage_id['checkpoint_sha256']==old_manifest['phase1/checkpoint_best_mIoU.pth']
    assert len(all_ids)==10582 and len(train_ids)==1464
    dev=set(experiments.read('splits/dev.txt').decode().splitlines())
    holdout=set(experiments.read('splits/holdout.txt').decode().splitlines())
    groups={n:accumulator() for n in ('train1464','additional9118','trainaug10582','dev100','holdout200')}
    selected_rows=[]

    def mask(archive, manifest, name):
        content=archive.read(name)
        assert hashlib.sha256(content).hexdigest()==manifest[name], name
        with Image.open(io.BytesIO(content)) as im:
            return np.asarray(im).copy()

    for index, name in enumerate(all_ids,1):
        raw=mask(experiments,new_manifest,f'uncertainty-full/trainaug/baseline/{name}.png')
        pseudo=mask(original,old_manifest,f'pseudo/{name}.png')
        with Image.open(DATA/f'SegmentationClassAug/{name}.png') as im:
            truth=np.asarray(im)
        assert raw.shape==pseudo.shape==truth.shape
        assert np.all(raw < 21) and np.all(pseudo < 21)
        matrices=[confusion(p,truth) for p in (raw,pseudo)]
        valid=truth < 21
        a=(raw==truth)&valid
        b=(pseudo==truth)&valid
        transitions=np.array([int((a&b).sum()),int((~a&b&valid).sum()),
                              int((a&~b&valid).sum()),int((~a&~b&valid).sum())])
        support=matrices[0].sum(1)
        present=(support >= 64);present[0]=False
        tp=[np.diag(m) for m in matrices]
        zero=np.array([int(((t==0)&present).sum()) for t in tp]+[int(present.sum())])
        absent=np.array([int(((m.sum(0)==0)&present).sum()) for m in matrices]+[int(present.sum())])
        targets=['trainaug10582','train1464' if name in train_set else 'additional9118']
        if name in dev: targets.append('dev100')
        if name in holdout: targets.append('holdout200')
        for target in targets:
            acc=groups[target];acc['raw']+=matrices[0];acc['pseudo']+=matrices[1]
            acc['transitions']+=transitions;acc['images']+=1
            acc['image_class_zero_tp']+=zero;acc['absent_predictions']+=absent
        if name in dev or name in holdout:
            # Include known-label/void interfaces when locating a margin, but
            # score only valid GT. This is a label-region margin diagnostic,
            # not standard Boundary IoU or an estimate of true void contours.
            edge=np.zeros_like(truth,dtype=bool)
            edge[:,1:]|=truth[:,1:]!=truth[:,:-1]
            edge[:,:-1]|=truth[:,1:]!=truth[:,:-1]
            edge[1:]|=truth[1:]!=truth[:-1]
            edge[:-1]|=truth[1:]!=truth[:-1]
            edge[[0,-1],:]=True;edge[:,[0,-1]]=True
            distance=distance_transform_edt(~edge)
            region={'near_label_edge':valid&(distance<=8),'interior':valid&(distance>8)}
            for region_name, selected in region.items():
                values=np.array([[int(((p!=truth)&selected).sum()),int(selected.sum()),
                                  int(((p!=truth)&selected&(truth>0)).sum())] for p in (raw,pseudo)])
                for target in targets: groups[target]['regions'][region_name]+=values
            if name in holdout:
                selected_rows.append(dict(name=name,corrected=int(transitions[1]),damaged=int(transitions[2]),
                    net_corrected_fraction=float((transitions[1]-transitions[2])/max(1,valid.sum())),
                    residual_foreground_miss=int(((truth>0)&valid&(pseudo==0)).sum()),
                    residual_background_leak=int(((truth==0)&(pseudo>0)).sum())))
        if index%1000==0 or index==len(all_ids): print('Analyzed',index,'/',len(all_ids),flush=True)

    results={}
    for group,acc in groups.items():
        trans=acc['transitions']; r,p=acc['raw'],acc['pseudo']
        total=int(r.sum());wrong=int(total-r.trace())
        row=dict(images=acc['images'],raw=metrics(r),pseudo=metrics(p),
                 raw_errors=error_types(r),pseudo_errors=error_types(p),
                 transition_columns=['both_correct','raw_wrong_pseudo_correct','raw_correct_pseudo_wrong','both_wrong'],
                 transition_counts=trans.tolist(),
                 corrected_fraction_of_raw_errors=100*int(trans[1])/max(1,wrong),
                 damaged_fraction_of_raw_correct=100*int(trans[2])/max(1,int(r.trace())),
                 both_wrong_fraction_of_raw_errors=100*int(trans[3])/max(1,wrong),
                 image_class_zero_tp=acc['image_class_zero_tp'].tolist(),
                 image_class_absent_prediction=acc['absent_predictions'].tolist(),
                 class_delta_pp={c:metrics(p)['class_iou'][c]-metrics(r)['class_iou'][c]
                    for c in CLASSES if metrics(p)['class_iou'][c] is not None and metrics(r)['class_iou'][c] is not None},
                 regions={key:dict(raw_error_percent=100*v[0,0]/max(1,v[0,1]),
                                    pseudo_error_percent=100*v[1,0]/max(1,v[1,1]),
                                    valid_pixels=int(v[0,1]),raw_wrong=int(v[0,0]),pseudo_wrong=int(v[1,0]))
                          for key,v in acc['regions'].items() if v.sum()})
        results[group]=row
        print(group,'raw',row['raw']['miou'],'pseudo',row['pseudo']['miou'],
              'correction%',row['corrected_fraction_of_raw_errors'],'damage%',row['damaged_fraction_of_raw_correct'],flush=True)
    expected=json.loads((OUT/'phase1-crf-train1464.json').read_text())
    print('Prior CRF receipt keys',list(expected),flush=True)
    assert abs(results['train1464']['pseudo']['miou']-68.1890587138)<1e-8
    logs=[json.loads(line) for line in original.read('phase1/log.txt').decode().splitlines()]
    assert [r['epoch'] for r in logs]==list(range(60))
    summary=dict(protocol='Regenerated FP32 raw CAM versus original AMP/CRF pseudo-labels; descriptive, not exact matched CRF isolation.',
        source_checkpoint_sha256=stage_id['checkpoint_sha256'], seed=stage_id['config']['seed'],
        verified_mask_payloads=2*len(all_ids), groups=results,phase1_history=logs,
        missing_evidence=['No saved coarse-only or cross-attention-only maps; no measured component attribution.',
                          'No matched fresh CRF run on regenerated FP32 CAMs; original raw float CAMs were not exported.'],
        region_protocol='Diagnostic 300 images only: <=8 pixel distance to any GT label change (including ignore margins) or image border; score only valid GT. Not Boundary IoU.',
        image_class_count_protocol='One image-class presence with >=64 valid GT pixels, foreground classes only; not object-instance counting.',
        example_selection='Four examples selected by extrema within the fixed 200-image diagnostic split; illustrative categories, not average effect.',
        elapsed_seconds=time.monotonic()-started)
    (OUT/'PHASE1_CAM_ANALYSIS.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')

    chosen=[('Largest net correction',max(selected_rows,key=lambda r:r['net_corrected_fraction'])),
            ('Largest net degradation',min(selected_rows,key=lambda r:r['net_corrected_fraction'])),
            ('Remaining foreground miss',max(selected_rows,key=lambda r:r['residual_foreground_miss'])),
            ('Remaining background leakage',max(selected_rows,key=lambda r:r['residual_background_leak']))]
    palette=[]
    for i in range(256):
        rgb=[0,0,0];value=i
        for bit in range(8):
            for channel in range(3): rgb[channel]|=((value>>channel)&1)<<(7-bit)
            value>>=3
        palette.extend(rgb)
    palette[765:768]=[150,150,150]
    sheet=Image.new('RGB',(5*240,4*280),'white');draw=ImageDraw.Draw(sheet)
    for row,(title,entry) in enumerate(chosen):
        name=entry['name'];y=row*280
        draw.text((5,y+3),f'{title}: {name}',fill='black')
        raw=mask(experiments,new_manifest,f'uncertainty-full/trainaug/baseline/{name}.png')
        pseudo=mask(original,old_manifest,f'pseudo/{name}.png')
        with Image.open(DATA/f'SegmentationClassAug/{name}.png') as im: truth=np.asarray(im)
        valid=truth<21; change=np.full((*truth.shape,3),235,dtype=np.uint8)
        change[(raw!=truth)&(pseudo==truth)&valid]=[30,160,75]
        change[(raw==truth)&(pseudo!=truth)&valid]=[230,130,20]
        change[(raw!=truth)&(pseudo!=truth)&valid]=[195,40,55]
        change[~valid]=[130,130,130]
        with Image.open(DATA/f'JPEGImages/{name}.jpg') as im: original_image=im.convert('RGB')
        images=[original_image]
        for labels in (truth,raw,pseudo):
            im=Image.fromarray(labels.astype('uint8')).convert('P');im.putpalette(palette);images.append(im.convert('RGB'))
        images.append(Image.fromarray(change))
        for col,(im,label) in enumerate(zip(images,['Image','GT','Regenerated raw fine CAM','Saved CRF pseudo-label','Green=corrected orange=damaged'])):
            draw.text((col*240+3,y+19),label,fill='black');im.thumbnail((240,240),Image.Resampling.NEAREST)
            sheet.paste(im,(col*240,y+37))
    sheet.save(OUT/'phase1-cam-examples.png')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,5.4))
r=results['train1464'];x=np.arange(21)
axes[0].barh(x-.18,[r['raw']['class_iou'][c] for c in CLASSES],height=.36,label='Regenerated raw')
axes[0].barh(x+.18,[r['pseudo']['class_iou'][c] for c in CLASSES],height=.36,label='Saved CRF pseudo')
axes[0].set(yticks=x,yticklabels=CLASSES,xlabel='Class IoU (%)',title='Phase-1 training split: 1,464 images');axes[0].invert_yaxis();axes[0].legend(fontsize=8)
epochs=[r['epoch']+1 for r in logs]
axes[1].plot(epochs,[100*r['test_mAP'] for r in logs],label='Class-token mAP')
axes[1].plot(epochs,[r['max_miou'] for r in logs],label='Selection CAM mIoU')
axes[1].axvline(34,color='#888888',ls='--');axes[1].set(xlabel='Original training epoch',ylabel='Different metrics (%)',title='Classification and localization differ');axes[1].legend(fontsize=8)
categories=['Background\nleakage','Foreground\nmiss','Wrong\nforeground class']
raw=np.array(list(r['raw_errors'].values()));pseudo=np.array(list(r['pseudo_errors'].values()))
x=np.arange(3);axes[2].bar(x-.18,raw/1e6,.36,label='Regenerated raw');axes[2].bar(x+.18,pseudo/1e6,.36,label='Saved CRF pseudo')
axes[2].set(xticks=x,xticklabels=categories,ylabel='Wrong valid-GT pixels (millions)',title='Residual error types');axes[2].legend(fontsize=8)
fig.suptitle('Phase-1 evidence: saved-prediction analysis; raw/CRF inference precision differs',fontsize=13)
fig.tight_layout();fig.savefig(OUT/'phase1-cam-analysis.png',dpi=180);plt.close(fig)
print('Phase-1 CPU review completed in',round(time.monotonic()-started,1),'seconds',flush=True)
