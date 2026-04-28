"""Preprocessing and augmentation transforms for BraTS-PEDs.

Key pieces:
    * `BratsPEDsLabelConverterd` - configurable raw-label -> (TC, WT, ET)
      multi-channel converter. Replaces MONAI's adult-only version.
    * `ModalityDropoutd` - randomly zero whole MRI channels + emit mask.
    * `train_transforms` / `val_transforms` - full pipelines as MONAI Composes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.transforms import (
    Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    MapTransform, NormalizeIntensityd, Orientationd, RandAffined,
    RandBiasFieldd, RandFlipd, RandGaussianNoised, RandGaussianSmoothd,
    RandGibbsNoised, RandScaleIntensityd, RandShiftIntensityd,
    RandSpatialCropd, Spacingd,
)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
@dataclass
class LabelMap:
    """Map a set of raw seg integer labels to the evaluation regions.

    Defaults match BraTS-PEDs 2024 convention. Override via config.
    """
    et: Tuple[int, ...] = (1,)                      # Enhancing Tumor
    tc: Tuple[int, ...] = (1, 2, 3)                 # Tumor Core (ET+NET+CC)
    wt: Tuple[int, ...] = (1, 2, 3, 4)              # Whole Tumor (all)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Sequence[int]]]) -> "LabelMap":
        if d is None:
            return cls()
        return cls(
            et=tuple(d.get("et", (1,))),
            tc=tuple(d.get("tc", (1, 2, 3))),
            wt=tuple(d.get("wt", (1, 2, 3, 4))),
        )


class BratsPEDsLabelConverterd(MapTransform):
    """Raw-integer segmentation -> (3, H, W, D) multi-channel {TC, WT, ET}."""
    def __init__(self, keys: Sequence[str], label_map: LabelMap):
        super().__init__(keys)
        self.lm = label_map

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = d[key]
            if hasattr(arr, "detach"):  # torch tensor
                lab = arr.detach().cpu().numpy()
            else:
                lab = np.asarray(arr)
            # Squeeze a potential singleton channel dim -> (H, W, D)
            while lab.ndim > 3 and lab.shape[0] == 1:
                lab = lab[0]

            def mask_of(vals):
                m = np.zeros_like(lab, dtype=np.float32)
                for v in vals:
                    m[lab == v] = 1.0
                return m

            out = np.stack([mask_of(self.lm.tc),
                            mask_of(self.lm.wt),
                            mask_of(self.lm.et)], axis=0)
            d[key] = torch.as_tensor(out, dtype=torch.float32)
        return d


# ---------------------------------------------------------------------------
# Modality dropout
# ---------------------------------------------------------------------------
class ModalityDropoutd(MapTransform):
    """Zero out 0-3 MRI channels per sample and emit a binary availability mask.

    Guarantees at least `min_present` channels remain. Also writes
    `data["modality_mask"]` as a float32 tensor of shape (C,).
    """
    def __init__(self, keys: Sequence[str], p_drop: float = 0.30,
                 min_present: int = 1, num_modalities: int = 4):
        super().__init__(keys)
        self.p_drop = p_drop
        self.min_present = min_present
        self.num_modalities = num_modalities

    def __call__(self, data):
        d = dict(data)
        # Sample a mask; retry until enough modalities remain.
        while True:
            mask = torch.ones(self.num_modalities, dtype=torch.float32)
            for c in range(self.num_modalities):
                if random.random() < self.p_drop:
                    mask[c] = 0.0
            if mask.sum().item() >= self.min_present:
                break
        for key in self.keys:
            img = d[key]
            # Apply mask in-place along channel axis 0
            for c in range(self.num_modalities):
                if mask[c] == 0:
                    img[c] = img[c] * 0.0
            d[key] = img
        d["modality_mask"] = mask
        return d


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def build_train_transforms(cfg: Dict, label_map: LabelMap) -> Compose:
    """Heavy augmentation pipeline for training."""
    roi = tuple(cfg.get("roi_size", (128, 128, 128)))
    pixdim = tuple(cfg.get("pixdim", (1.0, 1.0, 1.0)))
    p_drop = cfg.get("p_modality_dropout", 0.30)

    tfs: List = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim,
                 mode=("bilinear", "nearest")),
        BratsPEDsLabelConverterd(keys=["label"], label_map=label_map),
        # Foreground crop *before* normalization saves memory & focuses stats.
        CropForegroundd(keys=["image", "label"], source_key="image",
                        allow_smaller=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # Spatial crop of a fixed ROI (random location).
        RandSpatialCropd(keys=["image", "label"], roi_size=roi,
                         random_size=False),
        # Flips (anatomical left-right + sup-inf + ant-post).
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # Small elastic/affine for anatomical variability.
        RandAffined(keys=["image", "label"], prob=0.3,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros"),
        # MRI-specific artefact simulation.
        RandBiasFieldd(keys=["image"], prob=0.2, coeff_range=(0.0, 0.1)),
        RandGibbsNoised(keys=["image"], prob=0.1, alpha=(0.0, 0.6)),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        RandGaussianSmoothd(keys=["image"], prob=0.15,
                            sigma_x=(0.5, 1.2), sigma_y=(0.5, 1.2),
                            sigma_z=(0.5, 1.2)),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
        ModalityDropoutd(keys=["image"], p_drop=p_drop, min_present=1),
    ]
    return Compose(tfs)


def build_val_transforms(cfg: Dict, label_map: LabelMap) -> Compose:
    pixdim = tuple(cfg.get("pixdim", (1.0, 1.0, 1.0)))
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim,
                 mode=("bilinear", "nearest")),
        BratsPEDsLabelConverterd(keys=["label"], label_map=label_map),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def build_inference_transforms(cfg: Dict) -> Compose:
    """Transforms for a subject with no ground-truth label available."""
    pixdim = tuple(cfg.get("pixdim", (1.0, 1.0, 1.0)))
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])
