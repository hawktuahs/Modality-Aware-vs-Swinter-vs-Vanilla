"""Ablation framework.

Three ablation axes are supported:
    (A) embedding   : with vs without the modality embedding  (same checkpoint,
                      just bypass the hook at inference time).
    (B) modality    : 15 missing-modality combinations (|{1..4}| \\ empty),
                      measured at inference on the val set of each fold.
    (C) p_drop      : requires re-training; launched via scripts/train.py with
                      different `p_modality_dropout` values.

Produces per-case CSVs and an aggregate summary, ready for the paper.
"""
from __future__ import annotations

import csv
import logging
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.data import DataLoader

from .inference import InferenceConfig, predict_probs
from .metrics import REGIONS, CaseMetrics, aggregate_metrics, per_case_metrics
from .postprocess import PostProcessConfig, postprocess
from .stats import wilcoxon_signed_rank


MODALITY_NAMES = ("T1c", "T1n", "T2f", "T2w")


def all_modality_masks(num: int = 4, include_full: bool = True,
                       include_empty: bool = False) -> List[Tuple[int, ...]]:
    """Return every distinct modality-availability vector."""
    combos: List[Tuple[int, ...]] = []
    for bits in product((0, 1), repeat=num):
        if sum(bits) == 0 and not include_empty:
            continue
        if all(b == 1 for b in bits) and not include_full:
            continue
        combos.append(bits)
    if include_full and (1,) * num not in combos:
        combos.insert(0, (1,) * num)
    return combos


def evaluate_on_loader(model: torch.nn.Module,
                       loader: DataLoader,
                       device: torch.device,
                       mask_vec: Sequence[int],
                       cfg: InferenceConfig,
                       pp_cfg: Optional[PostProcessConfig] = None
                       ) -> List[CaseMetrics]:
    """Evaluate `model` on `loader`, forcing the given modality mask.

    Images' channels that correspond to 0s in `mask_vec` are zeroed *before*
    inference, matching the training-time corruption protocol.
    """
    mask_t = torch.tensor(mask_vec, dtype=torch.float32,
                          device=device).unsqueeze(0)        # (1, C)
    results: List[CaseMetrics] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            B = images.shape[0]
            # Zero missing channels
            chan_mask = mask_t.view(1, -1, 1, 1, 1)
            images_masked = images * chan_mask
            mm = mask_t.expand(B, -1)

            probs = predict_probs(model, images_masked, mm, cfg, device=device)
            probs_np = probs.cpu().numpy()
            gt_np = labels.cpu().numpy().astype(np.uint8)
            sids = batch.get("sid", [f"case_{i}" for i in range(B)])
            for b in range(B):
                pred_bin = postprocess(probs_np[b], pp_cfg)
                sid = sids[b] if isinstance(sids, (list, tuple)) else str(sids)
                results.append(per_case_metrics(
                    pred_bin, gt_np[b], sid=str(sid),
                    spacing=batch.get("spacing", (1.0, 1.0, 1.0))
                    if "spacing" in batch else (1.0, 1.0, 1.0),
                    compute_hd95=False,  # flip to True for final paper table
                ))
    return results


def ablation_missing_modalities(model: torch.nn.Module,
                                loader: DataLoader,
                                device: torch.device,
                                out_dir: Path,
                                cfg: InferenceConfig,
                                pp_cfg: Optional[PostProcessConfig] = None
                                ) -> Dict[str, List[CaseMetrics]]:
    """Run the full 15-condition ablation and save a per-case CSV per condition.

    Returns {condition_name: [CaseMetrics, ...]} for downstream stats tests.
    """
    log = logging.getLogger(__name__)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_masks = all_modality_masks(num=4, include_full=True, include_empty=False)
    results: Dict[str, List[CaseMetrics]] = {}
    n_cond = len(all_masks)
    n_subj = len(loader)
    t0_total = time.time()
    for cond_idx, mv in enumerate(all_masks, 1):
        name = "".join(MODALITY_NAMES[i] if b else "_" for i, b in enumerate(mv))
        t0 = time.time()
        log.info("[%d/%d] Condition: %-20s  (%d subjects) ...",
                 cond_idx, n_cond, name, n_subj)
        cases = evaluate_on_loader(model, loader, device, mv, cfg, pp_cfg)
        elapsed = time.time() - t0
        total_elapsed = time.time() - t0_total
        avg_per_cond = total_elapsed / cond_idx
        eta = avg_per_cond * (n_cond - cond_idx)
        log.info("  done in %.1fs | ETA: %.0fs (%.1fmin)",
                 elapsed, eta, eta / 60)
        results[name] = cases
        _save_cases_csv(cases, out_dir / f"cases_{name}.csv")
    _save_aggregate_csv(results, out_dir / "summary.csv")
    return results


def _save_cases_csv(cases: List[CaseMetrics], path: Path) -> None:
    if not cases:
        path.write_text("")
        return
    fieldnames = list(cases[0].to_row().keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for c in cases:
            w.writerow(c.to_row())


def _save_aggregate_csv(results: Dict[str, List[CaseMetrics]],
                        path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "region", "dice_mean", "dice_std",
                    "sens_mean", "spec_mean"])
        for cond, cases in results.items():
            agg = aggregate_metrics(cases)
            for r in list(REGIONS) + ["MEAN"]:
                d = agg[r]
                w.writerow([
                    cond, r,
                    f"{d.get('dice_mean', float('nan')):.4f}",
                    f"{d.get('dice_std', float('nan')):.4f}",
                    f"{d.get('sens_mean', float('nan')):.4f}"
                        if r != "MEAN" else "",
                    f"{d.get('spec_mean', float('nan')):.4f}"
                        if r != "MEAN" else "",
                ])


def pairwise_wilcoxon(cases_a: List[CaseMetrics], cases_b: List[CaseMetrics],
                      region: str = "MEAN") -> Dict[str, float]:
    """Paired Wilcoxon on per-case Dice between two conditions.

    Requires that the two lists are aligned by subject ID. If not, we align by
    sid automatically.
    """
    index_a = {c.sid: c for c in cases_a}
    index_b = {c.sid: c for c in cases_b}
    shared = sorted(set(index_a).intersection(index_b))
    def _val(c: CaseMetrics) -> float:
        if region == "MEAN":
            return float(np.nanmean([c.dice[r] for r in REGIONS]))
        return c.dice.get(region, float("nan"))
    a = [_val(index_a[s]) for s in shared]
    b = [_val(index_b[s]) for s in shared]
    return wilcoxon_signed_rank(a, b)
