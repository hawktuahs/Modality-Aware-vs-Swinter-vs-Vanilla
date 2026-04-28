#!/usr/bin/env python
"""Freeze the test set and K-fold splits as JSON so every experiment uses
the *same* data partitioning. Run ONCE per dataset.

Usage:
    python scripts/prepare_folds.py --config configs/segresnet_modality.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from neuroseg.data import (
    Subject, discover_subjects, holdout_test_split, stratified_kfold_splits,
)
from neuroseg.utils import Config, get_logger, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default=str(BASE_DIR / "splits.json"))
    args = parser.parse_args()

    log = get_logger(level="INFO")
    cfg = Config.from_yaml(Path(args.config))
    ds = cfg["dataset"]
    dataset_dir = BASE_DIR / ds["root"]
    log.info("Dataset dir : %s", dataset_dir)

    subjects = discover_subjects(dataset_dir, require_label=True)
    log.info("Discovered %d labelled subjects", len(subjects))
    if not subjects:
        raise SystemExit("No subjects found. Check dataset root.")

    dev, test = holdout_test_split(
        subjects, test_frac=float(ds.get("test_frac", 0.1)),
        seed=int(ds.get("test_seed", 12345)))
    log.info("Held-out test: %d | dev pool: %d", len(test), len(dev))

    folds = stratified_kfold_splits(dev, n_splits=int(ds.get("n_folds", 5)),
                                    seed=int(cfg["experiment"].get("seed", 42)))
    payload = {
        "dataset_root": str(dataset_dir),
        "test_frac": ds.get("test_frac", 0.1),
        "test_seed": ds.get("test_seed", 12345),
        "n_folds": ds.get("n_folds", 5),
        "test": [s.sid for s in test],
        "dev": [s.sid for s in dev],
        "folds": [
            {"train": [s.sid for s in tr], "val": [s.sid for s in va]}
            for tr, va in folds
        ],
    }
    save_json(payload, Path(args.out))
    log.info("Wrote splits -> %s", args.out)
    for i, f in enumerate(payload["folds"]):
        log.info("  fold %d: train=%d val=%d", i, len(f["train"]), len(f["val"]))


if __name__ == "__main__":
    main()
