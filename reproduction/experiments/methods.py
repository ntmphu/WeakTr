"""Small experimental operators, independent of timm/mmcv and model imports."""
import torch
from torch import nn
import torch.nn.functional as F


def clip_threshold(loss, start_value=1.2, patch_size=120):
    """Match upstream batch-averaged thresholds, including its padding convention."""
    detached = loss.detach()
    mean = detached.mean()
    if mean > start_value:
        return None
    b, h, w = detached.shape
    local = F.avg_pool2d(detached.mean(0)[None, None], kernel_size=patch_size,
                         stride=patch_size, padding=h % patch_size, count_include_pad=False)
    local = torch.maximum(local[:, 0], mean)
    threshold = local.repeat_interleave(b, 0).repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)
    if threshold.shape != loss.shape:
        raise ValueError('Use 480x480 crops with clipping patch size 120; threshold shape differs')
    return threshold


class ResidualClipping(nn.Module):
    def __init__(self, alpha=0., start_value=1.2, patch_size=120):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        self.alpha, self.start_value, self.patch_size = alpha, start_value, patch_size
        self.valid_pixels = self.clipped_pixels = 0

    def forward(self, pred, target, criterion):
        original = criterion(pred, target)
        threshold = clip_threshold(original, self.start_value, self.patch_size)
        valid = target != 255
        self.valid_pixels += int(valid.sum().detach())
        if threshold is None:
            return original, original
        self.clipped_pixels += int(((original > threshold) & valid).sum().detach())
        # alpha=0 preserves the original function and gradient away from ties.
        hard = (original - threshold).clamp(max=0) + threshold
        return original, hard + self.alpha * (original - threshold).clamp(min=0)


def gate_affinity(attention, features):
    """Nonnegative cosine gate; preserve row mass and fall back on empty rows."""
    f = F.normalize(features.float(), dim=-1)
    gate = (f @ f.transpose(-1, -2)).clamp(min=0)
    attention = attention.float()
    weighted = attention * gate
    mass = weighted.sum(-1, keepdim=True)
    result = weighted / mass.clamp_min(1e-12) * attention.sum(-1, keepdim=True)
    return torch.where(mass > 1e-12, result, attention)


def normalize_cams(cams):
    low = cams.flatten(-2).amin(-1)[..., None, None]
    high = cams.flatten(-2).amax(-1)[..., None, None]
    return (cams - low) / (high - low + 1e-8)


def cam_labels(cams, threshold):
    """Foreground channels are VOC classes 1..20; background is channel zero."""
    bg = torch.full_like(cams[:1], threshold)
    return torch.cat([bg, cams], dim=0).argmax(0)


def agreement_mask(views, aggregate, threshold=0.43, required=0.75):
    reference = cam_labels(normalize_cams(aggregate), threshold)
    votes = torch.stack([cam_labels(normalize_cams(view), threshold) for view in views])
    agreement = (votes == reference).float().mean(0)
    return reference.masked_fill(agreement < required, 255), agreement
