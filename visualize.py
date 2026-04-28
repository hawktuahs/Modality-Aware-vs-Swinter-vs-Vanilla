#!/usr/bin/env python
# ============================================================
# visualize.py  —  Overlay visualizations for BraTS-PEDs predictions
# ============================================================
# Usage:
#   python visualize.py --subject_dir "BraTS-PEDs-v1\Training\BraTS-PED-00001-000"
# ============================================================

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import nibabel as nib

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"


def find_best_slice(mask_3d):
    """Return the axial slice index with the most tumour voxels."""
    sums = mask_3d.sum(axis=(0, 1))   # sum over H, W → shape (D,)
    idx  = int(sums.argmax())
    return idx if sums[idx] > 0 else mask_3d.shape[-1] // 2


def create_overlay(img_np, gt_np, pred_np, out_path: str, slice_idx: int = None):
    """
    Saves a 2-row × 4-column comparison figure.
    Row 0: Ground Truth overlaid on T1c (red)
    Row 1: Prediction    overlaid on T1c (blue / green / orange)

    img_np  : (4, H, W, D)  raw MRI (T1c, T1n, T2f, T2w)
    gt_np   : (3, H, W, D)  or None  — GT masks (TC, WT, ET)
    pred_np : (3, H, W, D)  — predicted binary masks (TC, WT, ET)
    out_path: where to save the PNG
    """
    channel_names = ["TC (Tumor Core)", "WT (Whole Tumor)", "ET (Enhancing Tumor)"]
    colors_gt   = ["Reds",   "Purples", "Oranges"]
    colors_pred = ["Blues",  "Greens",  "YlOrBr"]

    # T1c is channel 0 — best anatomical reference
    t1c = img_np[0]

    if slice_idx is None:
        # Pick slice with most combined tumour signal
        combined = pred_np.sum(axis=0)
        slice_idx = find_best_slice(combined)

    flair_sl  = t1c[:, :, slice_idx]
    n_rows = 1 if gt_np is None else 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows),
                             facecolor="#111111")

    def _show(ax, bg, overlay, cmap, title, alpha=0.45):
        ax.imshow(bg, cmap="gray", aspect="equal")
        if overlay is not None:
            masked = np.ma.masked_where(overlay == 0, overlay)
            ax.imshow(masked, cmap=cmap, alpha=alpha, aspect="equal",
                      vmin=0.1, vmax=1.0)
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.axis("off")

    # ── Row 0: Ground Truth ────────────────────────────────────────
    if gt_np is not None:
        ax_row = axes[0] if n_rows == 2 else axes
        _show(ax_row[0], flair_sl, None, None, "T1c (reference)")
        for ch in range(3):
            _show(ax_row[ch + 1],
                  flair_sl,
                  gt_np[ch, :, :, slice_idx],
                  colors_gt[ch],
                  f"GT — {channel_names[ch]}")
        ax_row[0].set_ylabel("Ground Truth", color="white", fontsize=11,
                              labelpad=8)

    # ── Row 1: Prediction ──────────────────────────────────────────
    ax_row = axes[1] if gt_np is not None else axes
    if n_rows == 1:
        ax_row = axes
    _show(ax_row[0], flair_sl, None, None, "T1c (reference)")
    for ch in range(3):
        _show(ax_row[ch + 1],
              flair_sl,
              pred_np[ch, :, :, slice_idx],
              colors_pred[ch],
              f"Pred — {channel_names[ch]}")
    ax_row[0].set_ylabel("Prediction", color="white", fontsize=11, labelpad=8)

    # ── Legend patches ─────────────────────────────────────────────
    patch_gt   = mpatches.Patch(color="#e05252", label="Ground Truth")
    patch_pred = mpatches.Patch(color="#5282e0", label="Prediction")
    fig.legend(handles=[patch_gt, patch_pred], loc="lower center",
               ncol=2, fontsize=10, facecolor="#222222", labelcolor="white",
               framealpha=0.8)

    fig.suptitle(f"Axial Slice #{slice_idx} — Segmentation Overlay",
                 color="white", fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Overlay saved → {out_path}")


def create_multi_slice_grid(img_np, pred_np, out_path: str, n_slices: int = 6):
    """
    Creates a grid showing n_slices evenly-spaced axial slices with the
    colour-coded tumour predictions (TC=blue, WT=green, ET=orange) overlaid.
    """
    D  = img_np.shape[-1]
    t1 = img_np[0]
    idxs = np.linspace(D * 0.2, D * 0.8, n_slices, dtype=int)

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4.5),
                             facecolor="#0d0d0d")

    cmap_info = [("Blues", "TC"), ("Greens", "WT"), ("Oranges", "ET")]

    for col, sl in enumerate(idxs):
        ax = axes[col]
        ax.imshow(t1[:, :, sl], cmap="gray", aspect="equal")
        for i, (cmap, _) in enumerate(cmap_info):
            m = np.ma.masked_where(pred_np[i, :, :, sl] == 0,
                                   pred_np[i, :, :, sl])
            ax.imshow(m, cmap=cmap, alpha=0.50, aspect="equal",
                      vmin=0.1, vmax=1.0)
        ax.set_title(f"Slice {sl}", color="white", fontsize=9)
        ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=c, label=l)
               for c, l in [("#4d88ff", "TC"), ("#3db85c", "WT"), ("#e07533", "ET")]]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
               facecolor="#1a1a1a", labelcolor="white", framealpha=0.9)

    fig.suptitle("Multi-Slice Prediction View", color="white", fontsize=12, y=1.01)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ Multi-slice grid saved → {out_path}")


def _load_subject(subject_dir: Path):
    """Load MRI and (optionally) segmentation from a patient folder."""
    sid  = subject_dir.name
    mods = ["t1c", "t1n", "t2f", "t2w"]
    imgs = []
    ref_affine = None
    for m in mods:
        fp = subject_dir / f"{sid}-{m}.nii.gz"
        nib_img = nib.load(str(fp))
        arr = nib_img.get_fdata(dtype=np.float32)
        if ref_affine is None:
            ref_affine = nib_img.affine
        # normalise nonzero
        nz = arr != 0
        if nz.any():
            arr[nz] = (arr[nz] - arr[nz].mean()) / (arr[nz].std() + 1e-8)
        imgs.append(arr)
    img_np = np.stack(imgs, axis=0)   # (4, H, W, D)

    seg_path = subject_dir / f"{sid}-seg.nii.gz"
    gt_np = None
    if seg_path.exists():
        seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
        tc  = ((seg == 1) | (seg == 3)).astype(np.float32)
        wt  = ((seg == 1) | (seg == 2) | (seg == 3) | (seg == 4)).astype(np.float32)
        et  = (seg == 3).astype(np.float32)
        gt_np = np.stack([tc, wt, et], axis=0)
    return img_np, gt_np, ref_affine


def main():
    parser = argparse.ArgumentParser(description="Visualisation script for BraTS-PEDs predictions")
    parser.add_argument("--subject_dir", type=str, required=False,
                        help="Path to patient folder")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Path to saved NIfTI prediction folder (optional, "
                             "otherwise calls predict.py internally)")
    parser.add_argument("--slice_idx", type=int, default=None,
                        help="Axial slice index (auto-selected if omitted)")
    parser.add_argument("--multi", action="store_true",
                        help="Also produce multi-slice grid")
    args = parser.parse_args()

    # ── Determine subject dir ──────────────────────────────────────
    if args.subject_dir is None:
        train_dir = BASE_DIR / "BraTS-PEDs-v1" / "Training"
        subject   = sorted(train_dir.iterdir())[0]
    else:
        subject = Path(args.subject_dir)

    sid = subject.name
    print(f"Visualising subject: {sid}")

    img_np, gt_np, _ = _load_subject(subject)

    # ── Load prediction NIfTI if available ────────────────────────
    pred_dir = Path(args.pred_dir) if args.pred_dir else OUTPUT_DIR / sid
    pred_files = {
        "TC": pred_dir / f"{sid}-pred-TC.nii.gz",
        "WT": pred_dir / f"{sid}-pred-WT.nii.gz",
        "ET": pred_dir / f"{sid}-pred-ET.nii.gz",
    }
    if all(p.exists() for p in pred_files.values()):
        pred_np = np.stack(
            [nib.load(str(pred_files[k])).get_fdata(dtype=np.float32)
             for k in ["TC","WT","ET"]], axis=0
        )
        print("✓ Loaded pre-computed predictions from NIfTI files")
    else:
        print("No saved predictions found — running inference first...")
        # Import predict and run
        sys.path.insert(0, str(BASE_DIR))
        try:
            import site
            for p in site.getsitepackages() + [site.getusersitepackages()]:
                for module in ['cudnn', 'cublas', 'cuda_nvrtc']:
                    bin_path = os.path.join(p, 'nvidia', module, 'bin')
                    if os.path.exists(bin_path):
                        os.add_dll_directory(bin_path)
        except Exception: pass

        import torch
        from predict import load_model, predict_subject
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = load_model(BASE_DIR / "checkpoints" / "best_model.pth", device)
        pred_np, _ = predict_subject(subject, model, device, OUTPUT_DIR)

    if pred_np is None:
        print("ERROR: Could not obtain predictions.")
        return

    # ── Save overlay ───────────────────────────────────────────────
    out_overlay = str(OUTPUT_DIR / sid / f"{sid}-overlay.png")
    create_overlay(img_np, gt_np, pred_np, out_path=out_overlay,
                   slice_idx=args.slice_idx)

    if args.multi:
        out_multi = str(OUTPUT_DIR / sid / f"{sid}-multislice.png")
        create_multi_slice_grid(img_np, pred_np, out_path=out_multi)


if __name__ == "__main__":
    main()
