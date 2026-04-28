"""Loss functions.

We use MONAI's DiceCELoss as the base. A light wrapper (`MultiOutputLoss`) is
provided for models that emit deep-supervision auxiliary logits (list/tuple of
tensors at multiple resolutions) -- the standard practice is to downsample the
ground truth and compute weighted losses summed across scales.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, FocalLoss


def build_base_loss(cfg: dict) -> nn.Module:
    """Build the per-logit-pair loss. Sigmoid + soft-Dice + CE is the default."""
    name = cfg.get("name", "dicece").lower()
    if name == "dicece":
        return DiceCELoss(
            sigmoid=True,
            smooth_nr=cfg.get("smooth_nr", 0.0),
            smooth_dr=cfg.get("smooth_dr", 1e-5),
            squared_pred=cfg.get("squared_pred", False),
            lambda_dice=cfg.get("lambda_dice", 1.0),
            lambda_ce=cfg.get("lambda_ce", 1.0),
        )
    if name == "dicefocal":
        dice = DiceCELoss(sigmoid=True, lambda_ce=0.0, smooth_dr=1e-5)
        focal = FocalLoss(use_softmax=False, include_background=True,
                          gamma=cfg.get("gamma", 2.0))

        class DiceFocal(nn.Module):
            def forward(self, logits, target):
                return dice(logits, target) + focal(logits, target)

        return DiceFocal()
    raise ValueError(f"Unknown loss name: {name!r}")


class MultiOutputLoss(nn.Module):
    """Apply a base loss to each scale of a multi-output network.

    `weights` are applied to each scale; the target is downsampled to match
    each logits tensor's spatial dimensions using trilinear interpolation.
    """
    def __init__(self, base_loss: nn.Module,
                 weights: Sequence[float] = (1.0, 0.5, 0.25, 0.125)):
        super().__init__()
        self.base_loss = base_loss
        self.weights = list(weights)

    def forward(self,
                logits: Union[torch.Tensor, List[torch.Tensor]],
                target: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, torch.Tensor):
            return self.base_loss(logits, target)
        losses = []
        for i, lg in enumerate(logits):
            if i >= len(self.weights):
                break
            w = self.weights[i]
            if lg.shape[2:] != target.shape[2:]:
                tgt = F.interpolate(target, size=lg.shape[2:],
                                    mode="trilinear", align_corners=False)
                tgt = (tgt > 0.5).float()
            else:
                tgt = target
            losses.append(w * self.base_loss(lg, tgt))
        return torch.stack(losses).sum()
