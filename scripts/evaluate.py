#!/usr/bin/env python
"""Evaluate a trained checkpoint on a fold's val set (or the held-out test set).

Emits per-case CSV with Dice / HD95 / Sens / Spec for TC / WT / ET plus an
aggregate JSON summary.

Usage:
    python scripts/evaluate.py \
        --config configs/segresnet_modality.yaml \
        --ckpt outputs/runs/segresnet_modality_fold0/best.pth \
        --split val --fold 0
    python scripts/evaluate.py --config ... --ckpt ... --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from neuroseg.utils import (
    Config, inject_cuda_dlls, seed_everything, pick_device, get_logger,
    load_json, save_json,
)

inject_cuda_dlls()

import numpy as np
import torch
from monai.data import DataLoader, Dataset as MonaiDataset

from neuroseg.data import discover_subjects, to_monai_list
from neuroseg.inference import InferenceConfig, predict_probs
from neuroseg.metrics import aggregate_metrics, per_case_metrics, REGIONS
from neuroseg.models import build_model, load_checkpoint
from neuroseg.postprocess import PostProcessConfig, postprocess
from neuroseg.transforms import LabelMap, build_val_transforms


def _subjects_from_sids(all_subjects, sids):
    idx = {s.sid: s for s in all_subjects}
    return [idx[s] for s in sids if s in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--splits", default=str(BASE_DIR / "splits.json"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to write evaluation results.")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-pp", action="store_true",
                        help="Disable post-processing (raw threshold).")
    parser.add_argument("--compute-hd95", action="store_true",
                        help="Compute HD95 (slower).")
    args = parser.parse_args()

    log = get_logger(level="INFO")
    cfg = Config.from_yaml(Path(args.config))
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    device = pick_device("auto")

    dataset_dir = BASE_DIR / cfg["dataset"]["root"]
    all_subjects = discover_subjects(dataset_dir, require_label=True)

    # --- choose subjects
    sp = load_json(Path(args.splits))
    if args.split == "val":
        sids = sp["folds"][args.fold]["val"]
    else:
        sids = sp["test"]
    subs = _subjects_from_sids(all_subjects, sids)
    log.info("%s split: %d subjects", args.split, len(subs))

    label_map = LabelMap.from_dict(cfg["dataset"].get("label_map"))
    val_tf = build_val_transforms(cfg["data"], label_map)
    ds = MonaiDataset(data=to_monai_list(subs), transform=val_tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg["model"], device=device)
    load_checkpoint(model, args.ckpt, device=device, strict=False)

    infer_cfg_dict = dict(cfg.get("inference", {}))
    icfg = InferenceConfig(
        roi_size=tuple(infer_cfg_dict.get("roi_size", (128, 128, 128))),
        sw_batch_size=int(infer_cfg_dict.get("sw_batch_size", 4)),
        overlap=float(infer_cfg_dict.get("overlap", 0.5)),
        mode=infer_cfg_dict.get("mode", "gaussian"),
        use_tta=(not args.no_tta) and bool(infer_cfg_dict.get("use_tta", True)),
        flip_axes=tuple(infer_cfg_dict.get("flip_axes", (2, 3, 4))),
    )
    pp_dict = cfg.get("postprocess", {})
    pp_cfg = None if args.no_pp else PostProcessConfig(
        threshold=float(pp_dict.get("threshold", 0.5)),
        min_component_voxels=dict(pp_dict.get("min_component_voxels",
                                              {"TC": 100, "WT": 100, "ET": 50})),
        et_min_volume_voxels=int(pp_dict.get("et_min_volume_voxels", 200)),
        enable_et_rule=bool(pp_dict.get("enable_et_rule", True)),
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        sid = batch.get("sid", ["unknown"])[0]
        B = images.shape[0]
        mm = torch.ones(B, cfg["model"].get("num_modalities", 4),
                        dtype=torch.float32, device=device)
        probs = predict_probs(model, images, mm, icfg, device=device)
        probs_np = probs[0].cpu().numpy()
        gt_np = labels[0].cpu().numpy().astype(np.uint8)
        pred_bin = postprocess(probs_np, pp_cfg) if pp_cfg is not None \
                   else (probs_np >= 0.5).astype(np.uint8)
        cases.append(per_case_metrics(pred_bin, gt_np, sid=str(sid),
                                      compute_hd95=args.compute_hd95))
        log.info("  %s  Dice TC=%.3f WT=%.3f ET=%.3f",
                 sid, cases[-1].dice["TC"], cases[-1].dice["WT"],
                 cases[-1].dice["ET"])

    # Per-case CSV
    if cases:
        fieldnames = list(cases[0].to_row().keys())
        with open(out_dir / "per_case.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for c in cases:
                w.writerow(c.to_row())

    agg = aggregate_metrics(cases)
    save_json(agg, out_dir / "summary.json")

    log.info("AGGREGATE  mean-Dice=%.4f   TC=%.4f  WT=%.4f  ET=%.4f",
             agg["MEAN"]["dice_mean"], agg["TC"]["dice_mean"],
             agg["WT"]["dice_mean"], agg["ET"]["dice_mean"])
    log.info("Wrote -> %s", out_dir)


if __name__ == "__main__":
    main()
