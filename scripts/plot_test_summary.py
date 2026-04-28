"""Bar chart summarising test-set Dice across the three trained models
(NeuroSeg 5-fold, vanilla fold-0, SwinUNETR fold-0). Reads the
per-fold eval_test/summary.json files.

Usage:
    python scripts/plot_test_summary.py \
        --out outputs/figures/test_summary.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


def load(p: str) -> dict:
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # NeuroSeg: 5 folds
    neuro = {"TC": [], "WT": [], "ET": []}
    for f in range(5):
        s = load(f"outputs/runs/segresnet_modality_fold{f}/eval_test/summary.json")
        for k in neuro:
            neuro[k].append(s[k]["dice_mean"])

    van = load("outputs/runs/segresnet_vanilla_fold0/eval_test/summary.json")
    swin = load("outputs/runs/swin_unetr_pretrained_fold0/eval_test/summary.json")

    regions = ["TC", "WT", "ET"]
    x = np.arange(len(regions))
    width = 0.27

    neuro_mean = [mean(neuro[k]) for k in regions]
    neuro_std  = [stdev(neuro[k]) for k in regions]
    van_v      = [van[k]["dice_mean"] for k in regions]
    swin_v     = [swin[k]["dice_mean"] for k in regions]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    b1 = ax.bar(x - width, van_v, width, label="SegResNet vanilla (fold 0)",
                color="#d9534f", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x, neuro_mean, width, yerr=neuro_std,
                label="NeuroSeg (SegResNet + MAE, 5-fold mean)",
                color="#3a83b8", edgecolor="black", linewidth=0.6,
                capsize=4, error_kw={"elinewidth": 1.0})
    b3 = ax.bar(x + width, swin_v, width, label="SwinUNETR + adult SSL (fold 0)",
                color="#5cb85c", edgecolor="black", linewidth=0.6)

    for bars, vals in [(b1, van_v), (b2, neuro_mean), (b3, swin_v)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=11)
    ax.set_ylabel("Test-set Dice")
    ax.set_ylim(0, 1.0)
    ax.set_title("BraTS-PEDs 2023 — held-out test set (n = 26)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
