"""Post-processing of segmentation maps.

Two standard operations:
    * Region-wise connected-component filtering: drop components smaller than
      a configurable voxel threshold. Especially useful for ET, where small
      false-positive blobs dominate the Dice penalty.
    * ET -> NCR conversion: if the total predicted ET volume is below a
      threshold, convert all ET voxels to "non-enhancing" (i.e. part of TC
      but not ET). This is the canonical BraTS post-processing rule.

Both operations are applied per-sample after thresholding sigmoid
probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


try:
    from scipy import ndimage as ndi
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


@dataclass
class PostProcessConfig:
    threshold: float = 0.5
    min_component_voxels: Dict[str, int] = field(
        default_factory=lambda: {"TC": 100, "WT": 100, "ET": 50})
    et_min_volume_voxels: int = 200                 # convert ET -> TC below this
    enable_et_rule: bool = True


DEFAULT_MIN_VOXELS = {"TC": 100, "WT": 100, "ET": 50}


def _connected_component_filter(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if not _HAVE_SCIPY or min_voxels <= 0:
        return mask
    labelled, n = ndi.label(mask > 0)
    if n == 0:
        return mask
    sizes = ndi.sum(mask > 0, labelled, index=np.arange(1, n + 1))
    keep = np.zeros_like(mask, dtype=mask.dtype)
    for i, sz in enumerate(sizes, start=1):
        if sz >= min_voxels:
            keep[labelled == i] = 1
    return keep


def postprocess(probs: np.ndarray, cfg: Optional[PostProcessConfig] = None
                ) -> np.ndarray:
    """probs : (3, H, W, D) sigmoid probabilities in (TC, WT, ET) order.

    Returns binary masks (3, H, W, D) after thresholding, CC-filtering, and
    the ET volume rule.
    """
    if cfg is None:
        cfg = PostProcessConfig(min_component_voxels=DEFAULT_MIN_VOXELS.copy())
    if cfg.min_component_voxels is None:
        cfg.min_component_voxels = DEFAULT_MIN_VOXELS.copy()

    bin_masks = (probs >= cfg.threshold).astype(np.uint8)

    tc, wt, et = bin_masks[0], bin_masks[1], bin_masks[2]

    tc = _connected_component_filter(tc, cfg.min_component_voxels.get("TC", 0))
    wt = _connected_component_filter(wt, cfg.min_component_voxels.get("WT", 0))
    et = _connected_component_filter(et, cfg.min_component_voxels.get("ET", 0))

    # Geometric sanity: ET is a subset of TC which is a subset of WT.
    et = et & tc
    tc = tc & wt

    # Small-ET rule: if total ET volume is too small, drop it into TC.
    if cfg.enable_et_rule and et.sum() < cfg.et_min_volume_voxels:
        et = np.zeros_like(et)

    return np.stack([tc, wt, et], axis=0)
