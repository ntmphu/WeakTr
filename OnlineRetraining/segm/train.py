import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
import random
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproduction.checkpoint import atomic_save, load_checkpoint, make_snapshot, restore_training
from reproduction.data import sha256, annotation_fingerprint, read_ids
from reproduction.records import write_record
import time
from torch.nn.parallel import DistributedDataParallel as DDP

import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_segmenter
from segm.optim.factory import create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params

from segm.model.decoder import GradientClipping

from timm.utils import NativeScaler
from contextlib import suppress

from segm.engine import train_one_epoch, evaluate

import mlflow

from segm.utils.logger import printd

from segm.optim.optim_factory import get_parameter_groups, LayerDecayValueAssigner, create_optimizer


@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--eval-freq", default=1, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=False, is_flag=True)
@click.option("--run-id", default=None, type=str)
@click.option("--num-workers", default=2, type=int)
@click.option("--seed", default=504, type=int)
@click.option("--pretrained", default=None, type=click.Path(exists=True))
@click.option("--stop-after-epoch", default=None, type=int)
# data parameters
@click.option("--dataset", type=str)
@click.option("--max-ratio", type=int, default=None)
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--ann-dir", default="data/voc12/SegmentationClassAug", type=str)
# model parameters
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
# optimizer parameters
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=4, type=int)
@click.option("--epochs", default=100, type=int)
@click.option("-lr", "--learning-rate", default=1e-4, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--enc-lr", default=0.1, type=float)
@click.option("--layer-decay", default=1.0, type=float)
@click.option("--iter-warmup", default=0, type=int)
@click.option("--min-lr", default=1e-5, type=float)
# pus parameters
@click.option("--start-value", default=None, type=float)
@click.option("--patch-size", default=None, type=int)
@click.option("--gc/--no-gc", default=True, is_flag=True)
# dist parameters
@click.option("--local_rank", type=int, default=None)
def main(
        log_dir,
        eval_freq,
        amp,
        resume,
        run_id,
        num_workers,
        seed,
        pretrained,
        stop_after_epoch,
        dataset,
        max_ratio,
        im_size,
        crop_size,
        window_size,
        window_stride,
        ann_dir,
        backbone,
        decoder,
        optimizer,
        scheduler,
        weight_decay,
        dropout,
        drop_path,
        batch_size,
        epochs,
        learning_rate,
        normalization,
        enc_lr,
        layer_decay,
        iter_warmup,
        min_lr,
        start_value,
        patch_size,
        gc,
        local_rank,
):
    if local_rank is not None:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    ptu.set_gpu_dist_mode(True)
    if ptu.distributed:
        raise ValueError('Full reproduction recovery currently supports one training GPU.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if ptu.dist_rank == 0:
        # start mlflow
        if resume:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name=log_dir)

    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size // ptu.world_size
    variant = dict(
        seed=seed,
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        layer_decay=layer_decay,
        gradientclipping=gc,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=num_workers,
            ann_dir=ann_dir,
            max_ratio=max_ratio
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=min_lr,
            poly_power=0.9,
            poly_step_size=1,
            enc_lr=enc_lr,
            iter_warmup=iter_warmup
        ),
        clip_kwargs=dict(
            start_value=start_value,
            patch_size=patch_size,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / "checkpoint.pth"
    checkpoint_best_path = log_dir / "checkpoint_best.pth"

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    model = create_segmenter(dict(net_kwargs, pretrained_path=pretrained, initialize=not resume))
    model.to(ptu.device)

    # GradientClipping
    clip_kwargs = variant["clip_kwargs"]
    gradientclipping = None
    if gc:
        gradientclipping = GradientClipping(**clip_kwargs)
        gradientclipping.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v

    model_without_ddp = model

    num_layers = model_without_ddp.encoder.get_num_layers()
    if layer_decay < 1.0 or enc_lr < 1.0:
        # 总共包含num_blocks + 2层，block前(patch_embed, cls_token, pos_embed), block后
        assigner = LayerDecayValueAssigner(
            list(enc_lr * layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))+[1])
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
    skip_weight_decay_list = model.no_weight_decay()

    if assigner is not None:
        # params分组，包括if_weight_decay和lr_scale
        optimizer = create_optimizer(
            opt_args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        print('==========lwd==========')
    else:
        optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
    max_miou = 0
    max_epoch = 0

    # resume
    # Compare training semantics, allowing artifact/data relocation between runtimes.
    import copy
    recovery_config = copy.deepcopy(variant)
    for key in ('resume', 'log_dir'):
        recovery_config.pop(key)
    recovery_config['dataset_kwargs'].pop('ann_dir')
    recovery_config['split_hashes'] = {
        name: sha256(Path(__file__).resolve().parents[2] / 'voc12' / name)
        for name in ('train_aug_id.txt', 'val_id.txt')}
    if dataset == 'pascal_voc':
        recovery_config['annotation_sha256'] = annotation_fingerprint(
            train_loader.unwrapped.dataset.ann_dir,
            read_ids(Path(__file__).resolve().parents[2] / 'voc12/train_aug_id.txt'))
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        printd(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        next_epoch, best = restore_training(checkpoint, 'phase2', model, optimizer,
                                           lr_scheduler, loss_scaler, recovery_config)
        variant['algorithm_kwargs']['start_epoch'] = next_epoch
        max_epoch, max_miou = best['max_epoch'], best['max_miou']

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    printd(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()
    write_record(log_dir / 'run.json', dict(variant=variant, recovery=recovery_config),
                 [checkpoint_path] if resume else ([pretrained] if pretrained else []))
    started = time.monotonic()

    printd(f"Train dataset length: {len(train_loader.dataset)}")
    printd(f"Val dataset length: {len(val_loader.dataset)}")
    printd(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    printd(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")

    end_epoch = min(num_epochs, stop_after_epoch) if stop_after_epoch is not None else num_epochs
    for epoch in range(start_epoch, end_epoch):
        # train for one epoch
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            gradientclipping
        )

        # evaluate
        improved = False
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            eval_logger, miou = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
            )

            # save checkpoint best
            if miou > max_miou and ptu.dist_rank == 0:
                improved = True
                max_miou = float(miou)
                max_epoch = epoch
            eval_logger.update(**{"max_miou": max_miou, "n": 1})
            eval_logger.update(**{"max_epoch": max_epoch, "n": 1})

            printd(f"Stats [{epoch}]:", eval_logger, flush=True)
            printd("")

        # Snapshot only after validation has updated checkpoint-selection state.
        if ptu.dist_rank == 0:
            snapshot = make_snapshot('phase2', model_without_ddp, optimizer,
                                     lr_scheduler, loss_scaler, epoch,
                                     dict(max_epoch=max_epoch, max_miou=max_miou),
                                     recovery_config)
            snapshot.update(n_cls=n_cls, miou=max_miou)
            snapshot['elapsed_seconds_this_session'] = time.monotonic() - started
            if improved:
                atomic_save(snapshot, checkpoint_best_path)
            atomic_save(snapshot, checkpoint_path, retain_previous=True)

        # log stats
        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }
            val_stats = {}
            if eval_epoch:
                val_stats = {
                    k: meter.global_avg for k, meter in eval_logger.meters.items()
                }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
            }

            for key, value in log_stats.items():
                mlflow.log_metric(key, value, log_stats["epoch"])
            with open(log_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            break


if __name__ == "__main__":
    main()
