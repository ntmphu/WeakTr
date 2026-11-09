"""Full training snapshots. Only load checkpoints from trusted sources."""
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def load_checkpoint(path):
    # Full snapshots include Python/NumPy RNG state, not just tensor weights.
    return torch.load(path, map_location="cpu", weights_only=False)


def atomic_save(snapshot, path, retain_previous=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(snapshot, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if retain_previous and path.exists():
            # The previous file stays valid even if replacing latest fails.
            import shutil
            previous_tmp = path.with_name(path.stem + "_previous.tmp")
            shutil.copyfile(path, previous_tmp)
            os.replace(previous_tmp, path.with_name(path.stem + "_previous" + path.suffix))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_snapshot(stage, model, optimizer, scheduler, scaler, epoch, best, config, generator=None):
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        rng["loader"] = generator.get_state()
    return dict(
        format_version=1, stage=stage, model=model.state_dict(),
        optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
        loss_scaler=scaler.state_dict() if scaler is not None else None,
        epoch=epoch, best=dict(best), config=config, rng=rng,
    )


def restore_training(snapshot, stage, model, optimizer, scheduler, scaler, config, generator=None):
    required = {"format_version", "stage", "model", "optimizer", "lr_scheduler", "loss_scaler", "epoch", "best", "config", "rng"}
    if not required.issubset(snapshot) or snapshot["format_version"] != 1:
        raise ValueError("Full recovery requires a version-1 training snapshot; use weights-only checkpoints for evaluation or initialization.")
    if snapshot["stage"] != stage or snapshot["config"] != config:
        raise ValueError("Incompatible resume stage/configuration; restore the original training settings.")
    rng = snapshot["rng"]
    if generator is not None and "loader" not in rng:
        raise ValueError("Checkpoint is missing the dedicated data-loader RNG state.")
    if (scaler is None) != (snapshot["loss_scaler"] is None):
        raise ValueError("Resume AMP configuration does not match checkpoint.")
    model.load_state_dict(snapshot["model"], strict=True)
    optimizer.load_state_dict(snapshot["optimizer"])
    scheduler.load_state_dict(snapshot["lr_scheduler"])
    if scaler is not None:
        scaler.load_state_dict(snapshot["loss_scaler"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if "cuda" in rng:
        if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the checkpoint's CUDA device count.")
        torch.cuda.set_rng_state_all(rng["cuda"])
    if generator is not None:
        generator.set_state(rng["loader"])
    return snapshot["epoch"] + 1, dict(snapshot["best"])
