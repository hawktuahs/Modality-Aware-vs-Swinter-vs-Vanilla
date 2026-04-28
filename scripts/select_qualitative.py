"""Pick three representative subjects from a per-case evaluation CSV:
one easy (high Dice), one typical (median Dice), one hard (low but non-zero Dice).

Usage:
    python scripts/select_qualitative.py \
        --per-case outputs/runs/segresnet_modality_fold0/eval_test/per_case.csv \
        --n 3
"""
from __future__ import annotations

import argparse
import csv
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case", required=True)
    ap.add_argument("--n", type=int, default=3, help="how many to pick (currently always returns easy/typical/hard)")
    args = ap.parse_args()

    rows = []
    with open(args.per_case, newline="") as f:
        for r in csv.DictReader(f):
            tc = float(r["dice_TC"]); wt = float(r["dice_WT"]); et = float(r["dice_ET"])
            mean = (tc + wt + et) / 3.0
            rows.append((r["sid"], mean, tc, wt, et))

    # Drop the empty / fully-failed cases (they make bad qualitative figures)
    valid = [r for r in rows if r[1] > 0.10]
    valid.sort(key=lambda r: r[1])

    if not valid:
        raise SystemExit("No subjects with mean Dice > 0.10 — check the eval CSV.")

    n = len(valid)
    hard = valid[max(0, n // 6)]
    typical = valid[n // 2]
    easy = valid[-max(1, n // 6)]

    print(f"{'role':<10} {'sid':<28} {'mean':>6} {'TC':>6} {'WT':>6} {'ET':>6}")
    print("-" * 70)
    for role, r in [("hard", hard), ("typical", typical), ("easy", easy)]:
        print(f"{role:<10} {r[0]:<28} {r[1]:.4f}  {r[2]:.4f}  {r[3]:.4f}  {r[4]:.4f}")

    print()
    print("Run the predict step on each, e.g.:")
    for role, r in [("hard", hard), ("typical", typical), ("easy", easy)]:
        print(f"  python scripts/predict.py --config configs/segresnet_modality.yaml ^")
        print(f"      --ckpt outputs/runs/segresnet_modality_fold0/best.pth ^")
        print(f"      --subject {r[0]} ^")
        print(f"      --out-dir outputs/figures/qualitative/{role}")


if __name__ == "__main__":
    main()
