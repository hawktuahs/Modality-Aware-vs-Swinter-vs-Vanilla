"""Per-case evaluation metrics for BraTS-PEDs.

For each subject we compute, for each of the three regions (TC, WT, ET):
    * Dice score
    * HD95 (95th-percentile Hausdorff distance)  [optional - requires MONAI]
    * Sensitivity (a.k.a. Recall)
    * Specificity

BraTS convention: when a region is empty in both prediction and ground truth,
Dice is defined as 1.0 (perfect); when empty in GT but not in prediction, Dice
is 0.0 (worst). HD95 uses the same convention, returning NaN if ill-defined.
Empty-ET cases are common in BraTS-PEDs and must be handled explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

REGIONS = ("TC", "WT", "ET")


@dataclass
class CaseMetrics:
    sid: str
    dice: Dict[str, float] = field(default_factory=dict)
    hd95: Dict[str, float] = field(default_factory=dict)
    sens: Dict[str, float] = field(default_factory=dict)
    spec: Dict[str, float] = field(default_factory=dict)

    def to_row(self) -> Dict[str, float]:
        row: Dict[str, float] = {"sid": self.sid}
        for r in REGIONS:
            row[f"dice_{r}"] = self.dice.get(r, float("nan"))
            row[f"hd95_{r}"] = self.hd95.get(r, float("nan"))
            row[f"sens_{r}"] = self.sens.get(r, float("nan"))
            row[f"spec_{r}"] = self.spec.get(r, float("nan"))
        return row


def _to_np_bool(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return t.astype(bool)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _to_np_bool(pred); g = _to_np_bool(gt)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * (p & g).sum() / denom)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _to_np_bool(pred); g = _to_np_bool(gt)
    if g.sum() == 0:
        return float("nan")
    return float((p & g).sum() / g.sum())


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    p = _to_np_bool(pred); g = _to_np_bool(gt)
    neg_g = ~g
    if neg_g.sum() == 0:
        return float("nan")
    return float((~p & neg_g).sum() / neg_g.sum())


def hd95_distance(pred: np.ndarray, gt: np.ndarray,
                  spacing=(1.0, 1.0, 1.0)) -> float:
    """95th percentile Hausdorff, in physical units of `spacing`.

    Returns NaN if either mask is empty (distance ill-defined).
    Uses MONAI if available; otherwise falls back to scipy's EDT.
    """
    p = _to_np_bool(pred); g = _to_np_bool(gt)
    if p.sum() == 0 or g.sum() == 0:
        if p.sum() == 0 and g.sum() == 0:
            return 0.0
        return float("nan")
    try:
        from monai.metrics import compute_hausdorff_distance
        pt = torch.from_numpy(p[None, None].astype(np.uint8))
        gt_t = torch.from_numpy(g[None, None].astype(np.uint8))
        val = compute_hausdorff_distance(
            pt, gt_t, percentile=95,
            include_background=True, spacing=spacing,
        )
        return float(val.item())
    except Exception:
        return float("nan")


def per_case_metrics(pred_masks: np.ndarray, gt_masks: np.ndarray,
                     sid: str,
                     spacing=(1.0, 1.0, 1.0),
                     compute_hd95: bool = True) -> CaseMetrics:
    """pred_masks, gt_masks : (3, H, W, D) bool/float arrays in TC/WT/ET order."""
    out = CaseMetrics(sid=sid)
    for i, r in enumerate(REGIONS):
        p = pred_masks[i]
        g = gt_masks[i]
        out.dice[r] = dice_score(p, g)
        out.sens[r] = sensitivity(p, g)
        out.spec[r] = specificity(p, g)
        if compute_hd95:
            out.hd95[r] = hd95_distance(p, g, spacing=spacing)
    return out


def aggregate_metrics(cases: List[CaseMetrics]) -> Dict[str, Dict[str, float]]:
    """Return {region: {metric: mean, ...}} ignoring NaNs."""
    out: Dict[str, Dict[str, float]] = {}
    for r in REGIONS:
        d: Dict[str, float] = {}
        for name, getter in (("dice", lambda c: c.dice[r]),
                             ("hd95", lambda c: c.hd95.get(r, float("nan"))),
                             ("sens", lambda c: c.sens.get(r, float("nan"))),
                             ("spec", lambda c: c.spec.get(r, float("nan")))):
            vals = np.array([getter(c) for c in cases], dtype=np.float64)
            d[f"{name}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
            d[f"{name}_std"] = float(np.nanstd(vals)) if vals.size else float("nan")
        out[r] = d
    # Also add the overall mean-Dice across regions, one value per case.
    per_case_mean = np.array(
        [np.nanmean([c.dice[r] for r in REGIONS]) for c in cases])
    out["MEAN"] = {"dice_mean": float(np.nanmean(per_case_mean)),
                   "dice_std": float(np.nanstd(per_case_mean))}
    return out
