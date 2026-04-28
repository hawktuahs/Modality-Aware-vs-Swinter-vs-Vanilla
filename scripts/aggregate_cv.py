#!/usr/bin/env python
"""Aggregate per-fold results into mean ± std tables with bootstrap CIs.

Reads each fold's eval summary.json + per_case.csv and produces:
    outputs/cv_summary/<run>/cv_mean_std.json
    outputs/cv_summary/<run>/cv_bootstrap_ci.json

Usage:
    python scripts/aggregate_cv.py --run segresnet_modality --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import numpy as np
from neuroseg.stats import bootstrap_ci
from neuroseg.utils import get_logger, save_json


def _load_per_case(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = {"sid": row["sid"]}
            for k, v in row.items():
                if k == "sid":
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = float("nan")
            rows.append(parsed)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="Run name prefix, e.g. segresnet_modality")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--eval-subdir", default="eval_val")
    args = parser.parse_args()

    log = get_logger(level="INFO")
    all_rows: List[Dict[str, float]] = []
    fold_means: List[Dict[str, float]] = []

    for fold in args.folds:
        run_dir = BASE_DIR / args.runs_dir / f"{args.run}_fold{fold}"
        per_case_csv = run_dir / args.eval_subdir / "per_case.csv"
        if not per_case_csv.exists():
            log.warning("Missing %s", per_case_csv)
            continue
        rows = _load_per_case(per_case_csv)
        all_rows.extend(rows)
        means = {}
        for r in ("TC", "WT", "ET"):
            means[f"dice_{r}"] = float(np.nanmean([row[f"dice_{r}"]
                                                   for row in rows]))
        means["dice_MEAN"] = float(np.nanmean(
            [np.nanmean([row[f"dice_{r}"] for r in ("TC", "WT", "ET")])
             for row in rows]))
        means["fold"] = fold
        means["n"] = len(rows)
        fold_means.append(means)
        log.info("fold %d  meanD=%.4f  TC=%.4f  WT=%.4f  ET=%.4f (n=%d)",
                 fold, means["dice_MEAN"], means["dice_TC"],
                 means["dice_WT"], means["dice_ET"], means["n"])

    if not fold_means:
        log.error("No folds found. Did you run scripts/evaluate.py per fold?")
        return

    out_dir = BASE_DIR / "outputs" / "cv_summary" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_mean_std: Dict[str, Dict[str, float]] = {}
    for r in ("TC", "WT", "ET", "MEAN"):
        key = f"dice_{r}"
        vals = np.array([fm[key] for fm in fold_means])
        cv_mean_std[r] = {
            "fold_mean": float(vals.mean()),
            "fold_std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        }
    save_json({"per_fold": fold_means, "cv_mean_std": cv_mean_std},
              out_dir / "cv_mean_std.json")

    bootstrap: Dict[str, Dict[str, float]] = {}
    for r in ("TC", "WT", "ET"):
        vals = [row[f"dice_{r}"] for row in all_rows]
        m, lo, hi = bootstrap_ci(vals, n_boot=10_000)
        bootstrap[r] = {"mean": m, "ci_low": lo, "ci_high": hi,
                        "n_subjects": len(vals)}
    mean_vals = [np.nanmean([row[f"dice_{r}"] for r in ("TC", "WT", "ET")])
                 for row in all_rows]
    m, lo, hi = bootstrap_ci(mean_vals, n_boot=10_000)
    bootstrap["MEAN"] = {"mean": m, "ci_low": lo, "ci_high": hi,
                          "n_subjects": len(mean_vals)}
    save_json(bootstrap, out_dir / "cv_bootstrap_ci.json")

    log.info("Cross-validation summary written -> %s", out_dir)
    for r, d in bootstrap.items():
        log.info("  %-4s  %.4f  [%.4f, %.4f]  (n=%d)",
                 r, d["mean"], d["ci_low"], d["ci_high"], d["n_subjects"])


if __name__ == "__main__":
    main()
