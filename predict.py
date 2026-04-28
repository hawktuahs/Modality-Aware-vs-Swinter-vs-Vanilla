#!/usr/bin/env python
# ============================================================
# predict.py  —  Run inference on a BraTS-PEDs patient folder
# ============================================================
# Usage:
#   python predict.py --subject_dir "BraTS-PEDs-v1\Training\BraTS-PED-00001-000"
#   python predict.py --subject_dir "BraTS-PEDs-v1\Training\BraTS-PED-00001-000" --visualize
# ============================================================

import os, sys, argparse, time
import numpy as np
import nibabel as nib
from pathlib import Path

# ── DLL path injection (Windows cuDNN fix) ────────────────────────────
try:
    import site
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        for module in ['cudnn', 'cublas', 'cuda_nvrtc']:
            bin_path = os.path.join(p, 'nvidia', module, 'bin')
            if os.path.exists(bin_path):
                os.add_dll_directory(bin_path)
                os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')
except Exception:
    pass

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    EnsureTyped, Spacingd, Orientationd
)
from monai.inferers import sliding_window_inference

# ── Re-use the same model classes from train.py ───────────────────────
BASE_DIR       = Path(__file__).resolve().parent
CHECKPOINT     = BASE_DIR / "checkpoints" / "best_model.pth"
OUTPUT_DIR     = BASE_DIR / "outputs" / "predictions"
ROI_SIZE       = (128, 128, 128)
NUM_MODALITIES = 4

class ModalityEmbedding(nn.Module):
    def __init__(self, num_modalities=4, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_modalities, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
    def forward(self, mask):
        return self.net(mask)

class ModalityAwareSegResNetFixed(nn.Module):
    def __init__(self, blocks_down=(1,2,2,4), blocks_up=(1,1,1),
                 init_filters=16, in_channels=4, out_channels=3,
                 dropout_prob=0.2, emb_dim=128, num_modalities=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.backbone = SegResNet(
            blocks_down=blocks_down, blocks_up=blocks_up,
            init_filters=init_filters, in_channels=in_channels,
            out_channels=out_channels, dropout_prob=dropout_prob,
        )
        self.mod_emb  = ModalityEmbedding(num_modalities, emb_dim)
        self._current_emb = None
        bottleneck_filters = init_filters * (2 ** (len(blocks_down) - 1))
        self.emb_proj = nn.Linear(emb_dim, bottleneck_filters)
        self._hook_handle = self._register_bottleneck_hook()

    def _register_bottleneck_hook(self):
        last_block = list(self.backbone.down_layers)[-1]
        def _hook(module, input, output):
            if self._current_emb is None:
                return output
            B, C, *spatial = output.shape
            proj = self.emb_proj(self._current_emb).view(B, C, *([1]*len(spatial)))
            return output + proj
        return last_block.register_forward_hook(_hook)

    def forward(self, x, modality_mask=None):
        B = x.shape[0]
        if modality_mask is None:
            modality_mask = torch.ones(B, 4, dtype=torch.float32, device=x.device)
        self._current_emb = self.mod_emb(modality_mask)
        logits = self.backbone(x)
        self._current_emb = None
        return logits

    def __del__(self):
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()


def load_model(checkpoint_path, device):
    model = ModalityAwareSegResNetFixed(
        blocks_down=(1,2,2,4), blocks_up=(1,1,1),
        init_filters=16, in_channels=4, out_channels=3,
        dropout_prob=0.2, emb_dim=128, num_modalities=4,
    ).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✓ Model loaded from: {checkpoint_path}")
    return model


def predict_subject(subject_dir: Path, model, device, output_dir: Path,
                    modality_mask=None, save_nifti=True):
    """
    Run inference on one patient folder. Saves TC/WT/ET .nii.gz masks
    and returns the prediction array.
    """
    sid  = subject_dir.name
    mods = ["t1c", "t1n", "t2f", "t2w"]
    paths = {m: str(subject_dir / f"{sid}-{m}.nii.gz") for m in mods}
    seg_path = subject_dir / f"{sid}-seg.nii.gz"

    # Check which modalities are present
    present = [1 if Path(v).exists() else 0 for v in paths.values()]
    print(f"\nSubject   : {sid}")
    print(f"Modalities: T1c={present[0]} T1n={present[1]} T2f={present[2]} T2w={present[3]}")

    transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])

    # Use available modalities; missing ones will be zeroed post-load
    available = [str(subject_dir / f"{sid}-{m}.nii.gz")
                 for m, ok in zip(mods, present) if ok]
    if not available:
        print("  [ERROR] No modality files found!")
        return None, None

    # Load available modalities; pad zeros for missing ones
    all_imgs = []
    ref_nib  = None
    for m, ok in zip(mods, present):
        fpath = subject_dir / f"{sid}-{m}.nii.gz"
        if ok:
            nib_img = nib.load(str(fpath))
            arr     = nib_img.get_fdata(dtype=np.float32)
            if ref_nib is None:
                ref_nib = nib_img
        else:
            arr = np.zeros_like(nib.load(str(available[0])).get_fdata(dtype=np.float32))
        all_imgs.append(arr)

    # Stack → (4, H, W, D)
    img_np = np.stack(all_imgs, axis=0)

    # Normalize each channel nonzero
    for c in range(4):
        ch = img_np[c]
        mask_nz = ch != 0
        if mask_nz.any():
            ch[mask_nz] = (ch[mask_nz] - ch[mask_nz].mean()) / (ch[mask_nz].std() + 1e-8)
        img_np[c] = ch

    img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1,4,H,W,D)

    # Modality mask
    if modality_mask is None:
        mod_mask_t = torch.tensor(present, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        mod_mask_t = torch.tensor(modality_mask, dtype=torch.float32, device=device).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        outputs = sliding_window_inference(
            inputs        = img_tensor,
            roi_size      = ROI_SIZE,
            sw_batch_size = 1,
            predictor     = lambda x: model(x, mod_mask_t.expand(x.shape[0], -1)),
            overlap       = 0.5,
        )
    elapsed = time.time() - t0

    pred_binary = (outputs.sigmoid() > 0.5).float().squeeze(0).cpu().numpy()  # (3,H,W,D)

    # Calculate Dice if ground-truth exists
    label_names = ["TC (Tumor Core)", "WT (Whole Tumor)", "ET (Enhancing Tumor)"]
    if seg_path.exists():
        from monai.transforms import ConvertToMultiChannelBasedOnBratsClassesd
        seg_nib  = nib.load(str(seg_path))
        seg_np   = seg_nib.get_fdata(dtype=np.float32)
        # Convert BraTS multi-class to TC/WT/ET
        seg_tc = ((seg_np == 1) | (seg_np == 3)).astype(np.float32)
        seg_wt = ((seg_np == 1) | (seg_np == 2) | (seg_np == 3) | (seg_np == 4)).astype(np.float32)
        seg_et = (seg_np == 3).astype(np.float32)
        gt = np.stack([seg_tc, seg_wt, seg_et], axis=0)
        print(f"\nDice Scores (Inference took {elapsed:.1f}s):")
        for i, name in enumerate(label_names):
            p = pred_binary[i]; g = gt[i]
            inter = (p * g).sum()
            dice  = (2 * inter + 1e-5) / (p.sum() + g.sum() + 1e-5)
            print(f"  {name:26s}: {dice:.4f}")
    else:
        print(f"\nInference completed in {elapsed:.1f}s (no GT mask found for Dice).")

    # Save output masks as NIfTI
    if save_nifti and ref_nib is not None:
        out_subj = output_dir / sid
        out_subj.mkdir(parents=True, exist_ok=True)
        channel_names = ["TC", "WT", "ET"]
        affine = ref_nib.affine
        for i, cname in enumerate(channel_names):
            out_path = out_subj / f"{sid}-pred-{cname}.nii.gz"
            nib.save(nib.Nifti1Image(pred_binary[i], affine), str(out_path))
        combo_path = out_subj / f"{sid}-pred-combined.nii.gz"
        combined   = np.zeros_like(pred_binary[0])
        combined[pred_binary[2] == 1] = 3   # ET
        combined[pred_binary[0] == 1] = 1   # TC
        combined[pred_binary[1] == 1] = 2   # WT (only where not already TC/ET)
        nib.save(nib.Nifti1Image(combined, affine), str(combo_path))
        print(f"\n✓ Prediction masks saved to: {out_subj}")

    return pred_binary, img_np


def main():
    parser = argparse.ArgumentParser(description="BraTS-PEDs Inference Script")
    parser.add_argument("--subject_dir",  type=str, default=None,
                        help="Path to patient folder containing NIfTI files")
    parser.add_argument("--checkpoint",   type=str, default=str(CHECKPOINT),
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--modality_mask", type=int, nargs=4, default=None,
                        metavar=("T1C","T1N","T2F","T2W"),
                        help="Binary mask e.g. 1 0 1 1 to simulate missing T1n")
    parser.add_argument("--visualize",    action="store_true",
                        help="Also generate visualization overlay")
    parser.add_argument("--no_save",      action="store_true",
                        help="Skip saving NIfTI output masks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model  = load_model(Path(args.checkpoint), device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # If no subject_dir given, pick the first available training subject
    if args.subject_dir is None:
        train_dir = BASE_DIR / "BraTS-PEDs-v1" / "Training"
        subjects  = sorted(train_dir.iterdir())
        subject   = subjects[0]
        print(f"No --subject_dir given. Using first subject: {subject}")
    else:
        subject = Path(args.subject_dir)

    pred_binary, img_np = predict_subject(
        subject_dir   = subject,
        model         = model,
        device        = device,
        output_dir    = OUTPUT_DIR,
        modality_mask = args.modality_mask,
        save_nifti    = not args.no_save,
    )

    if args.visualize and pred_binary is not None:
        # Inline call to visualize
        sys.path.insert(0, str(BASE_DIR))
        from visualize import create_overlay
        sid  = subject.name
        out_path = str(OUTPUT_DIR / sid / f"{sid}-overlay.png")
        seg_path = subject / f"{sid}-seg.nii.gz"
        import nibabel as nib
        if seg_path.exists():
            seg_np = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
            seg_tc = ((seg_np == 1) | (seg_np == 3)).astype(np.float32)
            seg_wt = ((seg_np == 1) | (seg_np == 2) | (seg_np == 3) | (seg_np == 4)).astype(np.float32)
            seg_et = (seg_np == 3).astype(np.float32)
            gt = np.stack([seg_tc, seg_wt, seg_et], axis=0)
        else:
            gt = None
        create_overlay(img_np, gt, pred_binary, out_path=out_path)
        print(f"✓ Visualization saved to: {out_path}")

if __name__ == "__main__":
    main()
