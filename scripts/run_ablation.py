#!/usr/bin/env python
"""Run the full missing-modality ablation plus paired Wilcoxon tests.

Usage:
    python scripts/run_ablation.py \
        --config configs/segresnet_modality.yaml \
        --ckpt   outputs/runs/segresnet_modality_fold0/best.pth \
        --fold 0 --split val
    python scripts/run_ablation.py --config ... --ckpt ... --split test
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

import torch
from monai.data import DataLoader, Dataset as MonaiDataset

from neuroseg.data import discover_subjects, to_monai_list
from neuroseg.ablation import (
    MODALITY_NAMES, ablation_missing_modalities, evaluate_on_loader,
    pairwise_wilcoxon,
)
from neuroseg.inference import InferenceConfig
from neuroseg.metrics import REGIONS, aggregate_metrics
from neuroseg.models import build_model, load_checkpoint
from neuroseg.postprocess import PostProcessConfig
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
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--disable-embedding-control",
                        action="store_true",
                        help="Also evaluate same checkpoint with "
                             "use_embedding=False (ablation A).")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable test-time augmentation (8x faster).")
    parser.add_argument("--overlap", type=float, default=None,
                        help="Override sliding-window overlap (e.g. 0.25 for speed).")
    args = parser.parse_args()

    log = get_logger(level="INFO")
    cfg = Config.from_yaml(Path(args.config))
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    device = pick_device("auto")

    dataset_dir = BASE_DIR / cfg["dataset"]["root"]
    all_subjects = discover_subjects(dataset_dir, require_label=True)
    sp = load_json(Path(args.splits))
    sids = (sp["folds"][args.fold]["val"]
            if args.split == "val" else sp["test"])
    subs = _subjects_from_sids(all_subjects, sids)
    log.info("Ablation on %d subjects (%s)", len(subs), args.split)

    label_map = LabelMap.from_dict(cfg["dataset"].get("label_map"))
    ds = MonaiDataset(data=to_monai_list(subs),
                      transform=build_val_transforms(cfg["data"], label_map))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg["model"], device=device)
    load_checkpoint(model, args.ckpt, device=device, strict=False)

    # Inference / post-process config
    infer = cfg.get("inference", {})
    use_tta = bool(infer.get("use_tta", True)) and not args.no_tta
    overlap  = args.overlap if args.overlap is not None else float(infer.get("overlap", 0.5))
    icfg = InferenceConfig(
        roi_size=tuple(infer.get("roi_size", (128, 128, 128))),
        sw_batch_size=int(infer.get("sw_batch_size", 4)),
        overlap=overlap,
        mode=infer.get("mode", "gaussian"),
        use_tta=use_tta,
        flip_axes=tuple(infer.get("flip_axes", (2, 3, 4))),
    )
    log.info("Inference: TTA=%s  overlap=%.2f  sw_batch=%d",
             use_tta, overlap, icfg.sw_batch_size)
    pp = cfg.get("postprocess", {})
    pp_cfg = PostProcessConfig(
        threshold=float(pp.get("threshold", 0.5)),
        min_component_voxels=dict(pp.get("min_component_voxels",
                                         {"TC": 100, "WT": 100, "ET": 50})),
        et_min_volume_voxels=int(pp.get("et_min_volume_voxels", 200)),
        enable_et_rule=bool(pp.get("enable_et_rule", True)),
    )

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(args.ckpt).parent / f"ablation_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # (B) Missing-modality ablation with embedding ON
    model.use_embedding = True if hasattr(model, "use_embedding") else None
    results_emb = ablation_missing_modalities(
        model, loader, device, out_dir / "with_embedding",
        cfg=icfg, pp_cfg=pp_cfg)

    # (A) Embedding control (same checkpoint, embedding OFF)
    if args.disable_embedding_control and hasattr(model, "use_embedding"):
        log.info("Running embedding-off control...")
        model.use_embedding = False
        results_noemb = ablation_missing_modalities(
            model, loader, device, out_dir / "without_embedding",
            cfg=icfg, pp_cfg=pp_cfg)
        # Pairwise Wilcoxon per condition
        rows = []
        for cond in results_emb:
            stat = pairwise_wilcoxon(results_emb[cond],
                                     results_noemb.get(cond, []))
            rows.append({"condition": cond, **stat})
        with open(out_dir / "wilcoxon_emb_vs_noemb.csv", "w",
                  newline="", encoding="utf-8") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        model.use_embedding = True

    log.info("Ablation done -> %s", out_dir)


if __name__ == "__main__":
    main()
