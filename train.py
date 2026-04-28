#!/usr/bin/env python
# ============================================================
# train.py  —  Brain Tumor Segmentation (BraTS-PEDs v1)
# Merged from cell01..cell19 (Colab → Local, RTX 4060)
# ============================================================

# ============================================================
# Cell 2: Imports
# ============================================================
import os
import random as _random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandSpatialCropd, RandFlipd, RandRotate90d, ToTensord,
    EnsureTyped, ConvertToMultiChannelBasedOnBratsClassesd,
    Spacingd, Orientationd, AsDiscreted,
    MapTransform
)
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiLoader, decollate_batch
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
import warnings
warnings.filterwarnings("ignore")

set_determinism(seed=42)

# ============================================================
# Helpers & Classes (Must be top-level for pickling)
# ============================================================
def build_data_list(dataset_dir: Path):
    """
    Returns a list of dicts:
      { "image": [t1c, t1n, t2f, t2w], "label": seg }
    """
    subjects = sorted(dataset_dir.iterdir())
    data_list = []
    for subj_dir in subjects:
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name                        # e.g. BraTS-PED-00001-000
        t1c = subj_dir / f"{sid}-t1c.nii.gz"
        t1n = subj_dir / f"{sid}-t1n.nii.gz"
        t2f = subj_dir / f"{sid}-t2f.nii.gz"
        t2w = subj_dir / f"{sid}-t2w.nii.gz"
        seg = subj_dir / f"{sid}-seg.nii.gz"
        if all(p.exists() for p in [t1c, t1n, t2f, t2w, seg]):
            data_list.append({
                "image": [str(t1c), str(t1n), str(t2f), str(t2w)],
                "label": str(seg),
            })
        else:
            missing = [p for p in [t1c,t1n,t2f,t2w,seg] if not p.exists()]
            print(f"  [SKIP] {sid} — missing: {[p.name for p in missing]}")
    return data_list

class ModalityDropoutd(MapTransform):
    """
    Randomly zero out one or more MRI modalities during training.
    Stores a binary mask [t1c_ok, t1n_ok, t2f_ok, t2w_ok] in
    batch["modality_mask"] for the model to consume.
    """
    def __init__(self, keys, p_drop=0.30, min_present=1):
        super().__init__(keys)
        self.p_drop     = p_drop
        self.min_present = min_present

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]                    # (4, H, W, D)
            C   = img.shape[0]
            while True:
                mask = torch.ones(C, dtype=torch.float32)
                for c in range(C):
                    if _random.random() < self.p_drop:
                        mask[c] = 0.0
                if mask.sum() >= self.min_present:
                    break
            # zero out dropped modalities
            for c in range(C):
                if mask[c] == 0:
                    img[c] = img[c] * 0.0
            d[key] = img
            d["modality_mask"] = mask       # (4,)
        return d

class ModalityEmbedding(nn.Module):
    """
    2-layer MLP: binary mask (N,) → embedding (emb_dim,)
    """
    def __init__(self, num_modalities=4, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_modalities, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, mask):          # mask: (B, N)
        return self.net(mask)         # (B, emb_dim)

class ModalityAwareSegResNetFixed(nn.Module):
    """
    SegResNet backbone with a ModalityEmbedding injected at the
    encoder bottleneck via a forward hook.
    """
    def __init__(self,
                 blocks_down=(1,2,2,4),
                 blocks_up=(1,1,1),
                 init_filters=16,
                 in_channels=4,
                 out_channels=3,
                 dropout_prob=0.2,
                 emb_dim=128,
                 num_modalities=4):
        super().__init__()
        self.emb_dim = emb_dim

        self.backbone = SegResNet(
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )

        self.mod_emb = ModalityEmbedding(num_modalities, emb_dim)
        self._current_emb = None

        # Identify the bottleneck layer to hook into
        # SegResNet bottleneck = last encoder block
        bottleneck_filters = init_filters * (2 ** (len(blocks_down) - 1))
        self.emb_proj = nn.Linear(emb_dim, bottleneck_filters)

        # Register hook on the final encoder block
        self._hook_handle = self._register_bottleneck_hook()

    def _register_bottleneck_hook(self):
        # Walk to the last conv block in the encoder
        encoder_blocks = list(self.backbone.down_layers)
        last_block      = encoder_blocks[-1]

        def _hook(module, input, output):
            if self._current_emb is None:
                return output
            B, C, *spatial = output.shape
            proj = self.emb_proj(self._current_emb)    # (B, C)
            proj = proj.view(B, C, *([1]*len(spatial)))
            return output + proj

        return last_block.register_forward_hook(_hook)

    def forward(self, x, modality_mask=None):
        B = x.shape[0]
        if modality_mask is None:
            # Create tensor explicitly assigned to x.device, handled below or outside if possible
            modality_mask = torch.ones(B, 4, dtype=torch.float32, device=x.device)
            
        self._current_emb = self.mod_emb(modality_mask)   # (B, emb_dim)
        logits = self.backbone(x)
        self._current_emb = None
        return logits

    def __del__(self):
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()

# ============================================================
# Main execution function
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ============================================================
    # Cell 3: Paths & Hyperparameters
    # ============================================================
    BASE_DIR       = Path(__file__).resolve().parent
    DATASET_DIR    = BASE_DIR / "BraTS-PEDs-v1" / "Training"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    OUTPUT_DIR     = BASE_DIR / "outputs"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MAX_EPOCHS      = 100
    BATCH_SIZE      = 1          
    VAL_BATCH_SIZE  = 1
    LR              = 1e-4
    ROI_SIZE        = (128, 128, 128)
    NUM_MODALITIES  = 4
    NUM_CLASSES     = 3          
    WORKERS         = 1          
    VAL_SPLIT       = 0.20       

    print(f"Dataset  : {DATASET_DIR}")
    print(f"Subjects : {len(list(DATASET_DIR.iterdir()))}")
    print(f"ROI size : {ROI_SIZE}")
    print(f"Epochs   : {MAX_EPOCHS}")

    all_data = build_data_list(DATASET_DIR)
    print(f"Valid subjects found: {len(all_data)}")

    train_files, val_files = train_test_split(
        all_data, test_size=VAL_SPLIT, random_state=42
    )
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    # ============================================================
    # Cell 5: MONAI Transforms
    # ============================================================
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),   
        EnsureChannelFirstd(keys=["label"]),  
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=ROI_SIZE, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        EnsureChannelFirstd(keys=["label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    train_transforms_with_dropout = Compose([
        *train_transforms.transforms,          
        ModalityDropoutd(keys=["image"], p_drop=0.30, min_present=1),
    ])

    # ============================================================
    # Cell 7: DataLoaders
    # ============================================================
    train_ds = MonaiDataset(data=train_files, transform=train_transforms_with_dropout)
    val_ds   = MonaiDataset(data=val_files,   transform=val_transforms)

    # Note: num_workers triggers pickle on Windows.
    train_loader = MonaiLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=WORKERS, pin_memory=True)
    val_loader = MonaiLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                             num_workers=WORKERS, pin_memory=True)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ============================================================
    # Cell 8: Model — ModalityAwareSegResNet
    # ============================================================
    model = ModalityAwareSegResNetFixed(
        blocks_down=(1,2,2,4), blocks_up=(1,1,1),
        init_filters=16, in_channels=4, out_channels=3,
        dropout_prob=0.2, emb_dim=128, num_modalities=4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ============================================================
    # Cell 9: Loss, Optimizer, Scheduler, Metrics
    # ============================================================
    loss_fn    = DiceCELoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5, squared_pred=True)
    optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    print("Loss     : DiceCELoss (sigmoid)")
    print("Optimizer: AdamW lr=1e-4")
    print("Scheduler: CosineAnnealingLR")

    # ============================================================
    # Cell 10: Training Loop
    # ============================================================
    print("Starting training loop...")
    log_dir = str(OUTPUT_DIR / "runs" / "modality_aware_segresnet")
    writer  = SummaryWriter(log_dir=log_dir)

    best_mean_dice     = -1.0
    train_loss_history = []
    val_dice_history   = []

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        epoch_loss   = 0.0
        epoch_n_mods = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)     # (B, 4, H, W, D)
            labels = batch["label"].to(device)     # (B, 3, H, W, D)

            if "modality_mask" in batch:
                mod_mask = batch["modality_mask"].to(device)   
            else:
                mod_mask = torch.ones(images.shape[0], NUM_MODALITIES, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            outputs = model(images, modality_mask=mod_mask)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_n_mods += mod_mask.sum(dim=1).mean().item()

            print(f"  Epoch {epoch:03d}/{MAX_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        epoch_loss   /= len(train_loader)
        epoch_n_mods /= len(train_loader)
        train_loss_history.append(epoch_loss)
        scheduler.step()

        writer.add_scalar("Loss/train",                   epoch_loss,   epoch)
        writer.add_scalar("Train/mean_modalities_present",epoch_n_mods, epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Starting validation for Epoch {epoch:03d}...", flush=True)
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    val_outputs = sliding_window_inference(
                        inputs        = val_images,
                        roi_size      = ROI_SIZE,
                        sw_batch_size = 2,      
                        predictor     = lambda x: model(
                            x,
                            torch.ones(x.shape[0], NUM_MODALITIES, dtype=torch.float32, device=device)
                        ),
                        overlap       = 0.5,
                    )
                    val_preds = (val_outputs.sigmoid() > 0.5).float()
                    dice_metric(y_pred=val_preds, y=val_labels)

            metric_values = dice_metric.aggregate()   
            dice_metric.reset()
            mean_dice = metric_values.mean().item()
            tc_dice   = metric_values[0].item()
            wt_dice   = metric_values[1].item()
            et_dice   = metric_values[2].item()
            val_dice_history.append(mean_dice)

            writer.add_scalar("Dice/mean", mean_dice, epoch)
            writer.add_scalar("Dice/TC",   tc_dice,   epoch)
            writer.add_scalar("Dice/WT",   wt_dice,   epoch)
            writer.add_scalar("Dice/ET",   et_dice,   epoch)

            print(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Loss: {epoch_loss:.4f} | "
                f"AvgMod: {epoch_n_mods:.2f}/4 | "
                f"Dice mean: {mean_dice:.4f} "
                f"TC: {tc_dice:.4f} WT: {wt_dice:.4f} ET: {et_dice:.4f}",
                flush=True
            )

            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                ckpt_path = str(CHECKPOINT_DIR / "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f" --> New best model saved (Dice={best_mean_dice:.4f})", flush=True)
        else:
            print(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Loss: {epoch_loss:.4f} | "
                f"AvgMod: {epoch_n_mods:.2f}/4",
                flush=True
            )

    writer.close()
    print(f"\nTraining complete. Best mean Dice: {best_mean_dice:.4f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(train_loss_history) + 1), train_loss_history, color="royalblue")
    axes[0].set_title("Training Loss (DiceCE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    val_epochs = [e for e in range(1, MAX_EPOCHS + 1) if e % 5 == 0 or e == 1]
    axes[1].plot(val_epochs[:len(val_dice_history)], val_dice_history,
                 color="darkorange", marker="o")
    axes[1].set_title("Validation Mean Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    plt.tight_layout()
    curves_path = str(OUTPUT_DIR / "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Curves saved to {curves_path}")

    print("""
    === Novel Pipeline Summary ===
    Architecture  : ModalityAwareSegResNet
      Backbone    : 3-D SegResNet (MONAI, blocks [1,2,2,4] / [1,1,1])
      Novel module: ModalityEmbedding (2-layer MLP, emb_dim=128)
                    injected at encoder bottleneck via forward hook
    Input         : 4 MRI modalities (T1c, T1n, T2f, T2w)
                  + binary modality availability mask (4,)
    Output        : 3 binary segmentation masks (TC, WT, ET)
    Loss          : DiceCELoss (sigmoid, multi-label)
    Optimizer     : AdamW lr=1e-4
    Scheduler     : CosineAnnealingLR (T_max=100)
    Augmentation  : ModalityDropoutd (p_drop=0.30, min_present=1)
    Metrics       : Dice Score (TC, WT, ET)
    Dataset       : BraTS-PEDs v1 (local Training split 80/20)
    Checkpoints   : checkpoints/best_model.pth
    """)

if __name__ == "__main__":
    main()
