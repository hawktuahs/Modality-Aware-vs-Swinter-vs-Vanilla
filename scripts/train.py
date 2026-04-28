#!/usr/bin/env python
"""Train one fold of one model on BraTS-PEDs.

Usage:
    python scripts/train.py --config configs/segresnet_modality.yaml \
                            --fold 0 \
                            --splits splits.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from neuroseg.utils import (
    Config, inject_cuda_dlls, seed_everything, pick_device,
    describe_device, get_logger, count_parameters,
)

inject_cuda_dlls()

import torch
from monai.data import DataLoader, Dataset as MonaiDataset
from neuroseg.data import Subject, discover_subjects, to_monai_list
from neuroseg.transforms import (
    LabelMap, build_train_transforms, build_val_transforms,
)
from neuroseg.models import build_model
from neuroseg.losses import build_base_loss
from neuroseg.trainer import Trainer, TrainerConfig
from neuroseg.utils import load_json


def _subjects_from_sids(all_subjects, sids):
    idx = {s.sid: s for s in all_subjects}
    return [idx[s] for s in sids if s in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default=str(BASE_DIR / "splits.json"))
    parser.add_argument("--fold", type=int, default=None,
                        help="Override dataset.fold in the config")
    parser.add_argument("--run-suffix", type=str, default="",
                        help="Appended to the run name, e.g. '_fold0'")
    args = parser.parse_args()

    log = get_logger(level="INFO")
    cfg = Config.from_yaml(Path(args.config))
    ds_cfg = cfg["dataset"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    fold = args.fold if args.fold is not None else int(ds_cfg.get("fold", 0))
    seed = int(cfg["experiment"].get("seed", 42))
    seed_everything(seed)

    device = pick_device("auto")
    log.info("Device: %s", describe_device(device))

    # -------------- Splits (prefer file; else auto-build) -----------
    dataset_dir = BASE_DIR / ds_cfg["root"]
    all_subjects = discover_subjects(dataset_dir, require_label=True)

    if Path(args.splits).exists():
        sp = load_json(Path(args.splits))
        fold_sp = sp["folds"][fold]
        train_subs = _subjects_from_sids(all_subjects, fold_sp["train"])
        val_subs   = _subjects_from_sids(all_subjects, fold_sp["val"])
    else:
        log.warning("splits.json not found; building on-the-fly (run prepare_folds.py!)")
        from neuroseg.data import holdout_test_split, stratified_kfold_splits
        dev, _ = holdout_test_split(all_subjects, test_frac=ds_cfg.get("test_frac", 0.1),
                                    seed=ds_cfg.get("test_seed", 12345))
        train_subs, val_subs = stratified_kfold_splits(
            dev, n_splits=ds_cfg.get("n_folds", 5), seed=seed)[fold]

    log.info("Train %d | Val %d (fold=%d)", len(train_subs), len(val_subs), fold)

    # -------------- Transforms & loaders ----------------------------
    label_map = LabelMap.from_dict(ds_cfg.get("label_map"))
    train_tf = build_train_transforms(data_cfg, label_map)
    val_tf   = build_val_transforms(data_cfg, label_map)
    train_ds = MonaiDataset(data=to_monai_list(train_subs), transform=train_tf)
    val_ds   = MonaiDataset(data=to_monai_list(val_subs),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=data_cfg.get("batch_size", 1),
                              shuffle=True, num_workers=data_cfg.get("num_workers", 2),
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=data_cfg.get("val_batch_size", 1),
                            shuffle=False, num_workers=data_cfg.get("num_workers", 2),
                            pin_memory=True)

    # ------------------------- Model --------------------------------
    model = build_model(model_cfg, device=device)
    log.info("Model params: %s", f"{count_parameters(model):,}")
    loss_fn = build_base_loss(cfg.get("loss", {"name": "dicece"}))

    # --------------------- Trainer config ---------------------------
    run_name = f"{cfg['experiment']['name']}_fold{fold}{args.run_suffix}"
    tcfg = TrainerConfig(
        max_epochs=train_cfg.get("max_epochs", 200),
        val_every=train_cfg.get("val_every", 1),
        early_stop_patience=train_cfg.get("early_stop_patience", 30),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        amp=train_cfg.get("amp", True),
        base_lr=train_cfg.get("base_lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        min_lr=train_cfg.get("min_lr", 1e-6),
        roi_size=tuple(data_cfg.get("roi_size", (128, 128, 128))),
        sw_batch_size=train_cfg.get("sw_batch_size", 4),
        overlap=train_cfg.get("overlap", 0.5),
        num_modalities=model_cfg.get("num_modalities", 4),
        out_dir=BASE_DIR / cfg["experiment"].get("out_dir", "outputs") / "runs",
        run_name=run_name,
    )

    trainer = Trainer(model, loss_fn, train_loader, val_loader, tcfg, device)
    result = trainer.train()
    log.info("FINISHED best_mean_dice=%.4f epochs=%d time=%.0fs",
             result["best_mean_dice"], result["epochs_trained"],
             result["total_sec"])

    # Dump a manifest next to the checkpoint
    import json
    manifest_path = trainer.run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({
            "config_path": str(args.config),
            "fold": fold,
            "train_sids": [s.sid for s in train_subs],
            "val_sids":   [s.sid for s in val_subs],
            "result": result,
        }, fh, indent=2)
    log.info("Manifest written -> %s", manifest_path)


if __name__ == "__main__":
    main()
