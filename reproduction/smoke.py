"""Explicit GPU acceptance check; uses synthetic inputs and ImageNet weights."""
import argparse
import copy
import gc
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from reproduction.checkpoint import load_checkpoint
from reproduction.data import sha256
from reproduction.records import write_record


def check_gradients(model, loss):
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite smoke loss.')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise ValueError('Missing/nonfinite smoke gradients.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--record', default='artifacts/gpu-smoke.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke checks require a CUDA runtime.')
    if not sha256(args.pretrained).startswith('cd65a155'):
        raise ValueError('Expected official DeiT-S ImageNet weights.')
    torch.manual_seed(504)
    from timm.models import create_model
    import models  # registers original WeakTr
    from timm.utils import NativeScaler
    model = create_model('deit_small_WeakTr_patch16_224', pretrained=False,
                         num_classes=20, drop_rate=0.0, drop_path_rate=0.1,
                         drop_block_rate=None, reduction=8, pool='max', feat_reduction=8).cuda()
    weights = copy.deepcopy(load_checkpoint(args.pretrained)['model'])
    for key in ['head.weight', 'head.bias']:
        weights.pop(key, None)
    weights['cls_token'] = weights['cls_token'].repeat(1, 20, 1)
    weights['pos_embed'] = torch.cat([weights['pos_embed'][:, :1].repeat(1, 20, 1),
                                      weights['pos_embed'][:, 1:]], dim=1)
    incompatible = model.load_state_dict(weights, strict=False)
    if incompatible.unexpected_keys or any(key not in ('head.weight', 'head.bias') and
                                          not key.startswith('adaptive_attention_fusion.')
                                          for key in incompatible.missing_keys):
        raise ValueError(f'Unexpected Phase-1 ImageNet loading: {incompatible}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-5, weight_decay=.03)
    scaler = NativeScaler()
    with torch.cuda.amp.autocast():
        outputs = model(torch.randn(1, 3, 224, 224, device='cuda'))
        labels = torch.randint(0, 2, (1, 20), device='cuda').float()
        loss = sum(F.multilabel_soft_margin_loss(out, labels) for out in outputs)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite Phase-1 smoke loss.')
    scaler(loss, optimizer, parameters=model.parameters())
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise ValueError('Missing/nonfinite Phase-1 AMP gradients.')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite Phase-1 parameters after AMP update.')
    phase1_loss = loss.item()
    del model, optimizer, outputs, loss, weights
    gc.collect()
    torch.cuda.empty_cache()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'OnlineRetraining'))
    from segm import config
    from segm.model.factory import create_segmenter
    from segm.model.decoder import GradientClipping
    cfg = config.load_config()
    net = copy.deepcopy(cfg['model']['deit_small_patch16_224'])
    decoder = copy.deepcopy(cfg['decoder']['mask_transformer'])
    decoder['name'] = 'mask_transformer'
    net.update(backbone='deit_small_patch16_224', image_size=(480, 480), n_cls=21,
               dropout=0.0, drop_path_rate=.1, decoder=decoder, pretrained_path=args.pretrained)
    model = create_segmenter(net).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=.9)
    prediction = model(torch.randn(1, 3, 480, 480, device='cuda'))
    target = torch.randint(0, 21, (1, 480, 480), device='cuda')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    _, loss = GradientClipping(start_value=1.2, patch_size=120).cuda()(prediction, target, criterion)
    loss = loss.mean()
    check_gradients(model, loss)
    optimizer.step()
    write_record(args.record, dict(status='both_forward_backward_checks_passed', seed=504,
                                   synthetic_batch_size=1, phase1_loss=phase1_loss,
                                   phase2_loss=loss.item(),
                                   pending=['GPU interrupted recovery', 'author-checkpoint evaluation',
                                            'official batch-size memory acceptance']), [args.pretrained])
    print('Both synthetic GPU forward/backward checks passed. See record for remaining acceptance checks.')


if __name__ == '__main__':
    main()
