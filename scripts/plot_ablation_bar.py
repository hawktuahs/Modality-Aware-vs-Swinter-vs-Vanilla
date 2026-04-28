"""Bar-chart figure of vanilla vs. modality-aware Dice across all 15 modality
patterns. Reads the comparison CSV produced by the previous analysis step.

Usage:
    python scripts/plot_ablation_bar.py \
        --comparison-csv outputs/ablation/comparison_f0/ablation_comparison_f0.csv \
        --out outputs/figures/ablation_15patterns.png
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "pattern":      r["pattern"],
                "n_present":    int(r["n_present"]),
                "vanilla":      float(r["vanilla_mean"]),
                "modality":     float(r["modality_mean"]),
                "delta":        float(r["delta_mean"]),
                "p":            float(r["p_one_sided"]),
            })
    # Sort: by num present desc, then by pattern alpha
    rows.sort(key=lambda r: (-r["n_present"], r["pattern"]))
    return rows


def stars(p: float) -> str:
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Missing-modality robustness — vanilla vs modality-aware (fold 0, n=47)")
    args = ap.parse_args()

    rows = load_rows(args.comparison_csv)
    labels = [r["pattern"] for r in rows]
    van = np.array([r["vanilla"] for r in rows])
    mod = np.array([r["modality"] for r in rows])

    x = np.arange(len(rows))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6.5))
    b1 = ax.bar(x - width / 2, van, width, label="Vanilla SegResNet",
                color="#d9534f", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, mod, width, label="Modality-Aware (MAE + 30% dropout)",
                color="#3a83b8", edgecolor="black", linewidth=0.6)

    # Significance stars above the higher bar of each group
    for i, r in enumerate(rows):
        s = stars(r["p"]) if r["delta"] > 0 else ""
        # invert: if vanilla > mod-aware, use the inverse one-sided p
        if r["delta"] < 0:
            inv_p = 1.0 - r["p"]
            s = stars(inv_p)
        if not s:
            continue
        ymax = max(van[i], mod[i])
        ax.text(i, ymax + 0.012, s, ha="center", va="bottom",
                fontsize=10, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Mean Dice (TC/WT/ET avg)")
    ax.set_ylim(0, 1.0)
    ax.set_title(args.title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)

    # Annotate "all 4" baseline and "1 mod" extremes for paper narrative
    ax.axvspan(-0.5, 0.5, color="#fff2cc", alpha=0.4, zorder=0)
    ax.text(0, 0.96, "all 4", ha="center", fontsize=8, color="#777")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
