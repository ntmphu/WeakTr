"""One-GPU official VOC commands; every stage is an explicit invocation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.data import read_ids, sha256, validate_voc
from reproduction.records import write_record

ROOT = Path(__file__).resolve().parents[1]


def commands(stage, data, output, pretrained, resume='', seed=504, scratch=None):
    """Return argv/cwd pairs, so commands can be reviewed without starting jobs."""
    data, output = Path(data).resolve(), Path(output).resolve()
    phase1, phase2 = output / 'phase1', output / 'phase2'
    temporary = Path(scratch).resolve() if scratch else output
    cams, pseudo = temporary / 'cams', output / 'pseudo'
    common = [sys.executable, 'main.py', '--model', 'deit_small_WeakTr_patch16_224',
              '--data-path', str(data), '--reduction', '8', '--pool-type', 'max',
              '--output_dir', str(phase1), '--seed', str(seed)]
    if stage == 'phase1':
        command = common + ['--data-set', 'VOC12', '--img-ms-list', 'voc12/train_id.txt',
                            '--cam-npy-dir', str(temporary / 'phase1/selection-cams'), '--epochs', '60',
                            '--batch-size', '64', '--lr', '6e-4', '--weight-decay', '0.03',
                            '--finetune', str(pretrained)]
        if resume:
            command += ['--resume', str(resume)]
        return [(command, ROOT)]
    if stage == 'cams':
        return [(common + ['--data-set', 'VOC12MS', '--img-ms-list', 'voc12/train_aug_id.txt',
                           '--scales', '1.0', '1.2', '--gen_attention_maps',
                           '--cam-npy-dir', str(cams), '--resume',
                           str(resume or phase1 / 'checkpoint_best_mIoU.pth')], ROOT)]
    if stage == 'pseudo':
        return [([sys.executable, 'evaluation.py', '--list', 'voc12/train_aug_id.txt',
                  '--data-path', str(data), '--type', 'npy', '--predict_dir', str(cams),
                  '--out-dir', str(pseudo), '--t', '43', '--out-crf'], ROOT)]
    if stage == 'cam-eval':
        return [([sys.executable, 'evaluation.py', '--list', 'voc12/train_id.txt',
                  '--data-path', str(data), '--type', 'npy', '--predict_dir', str(cams),
                  '--curve', 'True', '--start', '40', '--end', '60',
                  '--logfile', str(output / 'cam-eval.txt')], ROOT)]
    if stage == 'phase2':
        command = [sys.executable, '-m', 'segm.train', '--log-dir', str(phase2),
                   '--dataset', 'pascal_voc', '--backbone', 'deit_small_patch16_224',
                   '--decoder', 'mask_transformer', '--ann-dir', str(pseudo),
                   '--start-value', '1.2', '--patch-size', '120', '--epochs', '100',
                   '--batch-size', '4', '--optimizer', 'sgd', '--scheduler', 'polynomial',
                   '--learning-rate', '1e-4', '--enc-lr', '0.1', '--eval-freq', '1',
                   '--im-size', '520', '--crop-size', '480', '--window-size', '480',
                   '--window-stride', '320', '--seed', str(seed), '--pretrained', str(pretrained)]
        if resume:
            if Path(resume).resolve() != phase2 / 'checkpoint.pth':
                raise ValueError('Restore the Phase-2 output directory, then resume its checkpoint.pth.')
            command += ['--resume']
        return [(command, ROOT / 'OnlineRetraining')]
    if stage == 'evaluate':
        return [([sys.executable, '-m', 'segm.eval.miou', '--multiscale',
                  '--backbone', 'multi', '--weight', '0.8', '--predict-dir', str(temporary / 'seg_prob_ms'),
                  str(resume or phase2 / 'checkpoint.pth'), 'pascal_voc'], ROOT / 'OnlineRetraining')]
    if stage == 'crf':
        return [([sys.executable, '-m', 'segm.eval.make_crf', '--list', 'val.txt',
                  '--data-path', str(data), '--predict-dir', str(temporary / 'seg_prob_ms'),
                  '--predict-png-dir', str(output / 'seg_pred_ms'), '--num-cls', '21',
                  '--dataset', 'voc12'], ROOT / 'OnlineRetraining')]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'phase1', 'cams', 'cam-eval', 'pseudo', 'phase2', 'evaluate', 'crf'])
    parser.add_argument('--data', required=True, help='Parent of voc12/VOCdevkit/VOC2012')
    parser.add_argument('--output', required=True)
    parser.add_argument('--scratch', help='Directory for temporary CAM/probability collections')
    parser.add_argument('--pretrained', required=True, help='ImageNet DeiT-S checkpoint, never author WeakTr weights')
    parser.add_argument('--resume', default='')
    parser.add_argument('--seed', type=int, default=504)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=None)
    args = parser.parse_args()
    args.pretrained = str(Path(args.pretrained).resolve())
    if args.resume:
        args.resume = str(Path(args.resume).resolve())
    jobs = [] if args.stage == 'preflight' else commands(args.stage, args.data, args.output, args.pretrained,
                                                       args.resume, args.seed, args.scratch)
    if args.stop_after_epoch is not None:
        if args.stage not in ('phase1', 'phase2') or args.stop_after_epoch < 1:
            parser.error('--stop-after-epoch requires a training stage and a positive epoch count')
        jobs[0][0].extend(['--stop-after-epoch', str(args.stop_after_epoch)])
    for command, cwd in jobs:
        print(f'cwd={cwd}\n{subprocess.list2cmdline(command)}', flush=True)
    if args.dry_run:
        return
    data, output = Path(args.data).resolve(), Path(args.output).resolve()
    voc = data / 'voc12/VOCdevkit/VOC2012'
    pseudo = output / 'pseudo'
    validation = validate_voc(voc, ROOT / 'voc12/train_aug_id.txt', ROOT / 'voc12/val_id.txt',
                              pseudo if args.stage == 'phase2' else None, progress=True)
    # mmseg consumes these files rather than the repository lists. Require exact order.
    for local, runtime in [('train_aug_id.txt', 'trainaug.txt'), ('val_id.txt', 'val.txt')]:
        if read_ids(ROOT / 'voc12' / local) != read_ids(voc / 'ImageSets/Segmentation' / runtime):
            raise ValueError(f'Runtime split differs from repository: {runtime}')
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(args.pretrained)
    # Official ImageNet checkpoint's published SHA256 prefix; full hash is recorded.
    if not sha256(args.pretrained).startswith('cd65a155'):
        raise ValueError('Expected official DeiT-S ImageNet checkpoint (SHA256 prefix cd65a155).')
    if args.stage in ('pseudo', 'cam-eval'):
        cam_list = 'train_aug_id.txt' if args.stage == 'pseudo' else 'train_id.txt'
        for name in read_ids(ROOT / 'voc12' / cam_list):
            if not (Path(args.scratch or output) / 'cams' / (name + '.npy')).is_file():
                raise FileNotFoundError(f'Missing training CAM: {name}')
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['DATASET'] = str(data)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'OnlineRetraining'), env.get('PYTHONPATH', '')])
    # Stable hashing must be set before the child Python process starts.
    env['PYTHONHASHSEED'] = str(args.seed)
    env['PYTHONUNBUFFERED'] = '1'
    record_path = output / f'{args.stage}-record.json'
    record = write_record(record_path, dict(arguments=vars(args), validation=validation,
                                           commands=[command for command, _ in jobs]),
                          [args.pretrained] + ([args.resume] if args.resume else []))
    started = time.monotonic()
    try:
        log_path = output / f'{args.stage}-{time.time_ns()}.log'
        with log_path.open('w', encoding='utf-8') as log:
            for command, cwd in jobs:
                with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, command)
        record['log_path'] = str(log_path)
        if args.stage == 'pseudo':
            validate_voc(voc, ROOT / 'voc12/train_aug_id.txt', ROOT / 'voc12/val_id.txt', pseudo)
        record['status'] = 'command_completed'
    except BaseException:
        record['status'] = 'failed_or_interrupted'
        raise
    finally:
        import json
        record['elapsed_seconds'] = time.monotonic() - started
        record_path.write_text(json.dumps(record, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
