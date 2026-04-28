# NeuroSeg — Modality-Aware Pediatric Brain Tumor Segmentation

A research-grade 3D segmentation pipeline for the **BraTS-PEDs 2023**
dataset with support for:

- Modality-availability embedding (MAE) injected at the encoder
  bottleneck — produces robust predictions when MRI modalities are
  missing.
- Two backbones: **SegResNet** (light baseline) and **SwinUNETR** with
  MONAI's self-supervised BraTS-Adult pretrained encoder weights
  ("base" for transfer learning).
- 5-fold cross-validation on a frozen held-out test set.
- Heavy MRI-specific augmentation (bias field, Gibbs noise, Gaussian
  smoothing, elastic affine) + modality dropout.
- Test-time augmentation (8-way flip) + connected-component
  post-processing + the "small-ET rule".
- Per-case Dice, HD95, Sensitivity, Specificity; paired Wilcoxon tests;
  bootstrap 95 % CIs.
- Flask web app for clinical-style drag-and-drop inference (with TTA
  and overlay figures).

## Repository layout

```
src/neuroseg/              # Importable Python package
  data.py                  # Subject discovery, K-fold splits
  transforms.py            # MONAI pipelines + ModalityDropoutd + label converter
  models.py                # ModalityAware SegResNet + SwinUNETR
  losses.py                # DiceCELoss + deep-supervision wrapper
  metrics.py               # Per-case Dice/HD95/Sens/Spec + aggregation
  tta.py                   # 8-way flip test-time augmentation
  postprocess.py           # Connected-component filtering + ET rule
  trainer.py               # AMP training loop + early stopping + CSV logs
  inference.py             # Sliding-window + TTA + post-process
  ablation.py              # Missing-modality grid + paired Wilcoxon helpers
  stats.py                 # Wilcoxon + bootstrap CIs
  utils.py                 # Config YAML + seed + logging

configs/                   # Experiment YAML configs
  segresnet_modality.yaml      # Primary model (SegResNet + MAE + p_drop=0.3)
  segresnet_vanilla.yaml       # Control (no MAE, p_drop=0)
  swin_unetr_pretrained.yaml   # Transfer model ("the base")

scripts/
  prepare_folds.py         # Freeze train/val/test into splits.json
  train.py                 # Train one fold
  evaluate.py              # Evaluate a checkpoint on val or test
  predict.py               # Single-subject inference
  run_ablation.py          # Full missing-modality + emb control
  aggregate_cv.py          # Mean ± std + bootstrap CIs across folds

app.py                     # Flask demo (http://127.0.0.1:5000)
paper/RESEARCH_PAPER.md    # Research-paper scaffold (fill [[FILL]] tags)
docs/EXPERIMENT_SCHEDULE.md# Day-by-day compute plan for 1-2 weeks
requirements.txt           # Python deps
legacy/                    # Original notebook-merged scripts (kept for reference)
```

## Quickstart

### 1. Install
```powershell
python -m venv venv
venv\Scripts\activate
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 ^
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Freeze splits
```powershell
python scripts/prepare_folds.py --config configs/segresnet_modality.yaml
```

### 3. Train a fold
```powershell
python scripts/train.py --config configs/segresnet_modality.yaml --fold 0
```

### 4. Evaluate
```powershell
python scripts/evaluate.py --config configs/segresnet_modality.yaml ^
    --ckpt outputs/runs/segresnet_modality_fold0/best.pth ^
    --split val --fold 0 --compute-hd95
```

### 5. Ablation study
```powershell
python scripts/run_ablation.py --config configs/segresnet_modality.yaml ^
    --ckpt outputs/runs/segresnet_modality_fold0/best.pth ^
    --split val --fold 0 --disable-embedding-control
```

### 6. Single-subject inference
```powershell
python scripts/predict.py --config configs/segresnet_modality.yaml ^
    --ckpt outputs/runs/segresnet_modality_fold0/best.pth ^
    --subject-dir BraTS-PEDs-v1/Training/BraTS-PED-00001-000 ^
    --modality-mask 1 0 1 1
```

### 7. Web app
```powershell
$env:NEUROSEG_CKPT = "outputs/runs/segresnet_modality_fold0/best.pth"
python app.py
```
Open http://127.0.0.1:5000

## Using the original `best_model.pth`

The existing `checkpoints/best_model.pth` was trained with the *old*
(adult) label convention. You can still evaluate it with the new pipeline
by creating a one-off config whose `dataset.label_map` mirrors the adult
convention:

```yaml
dataset:
  label_map:
    et: [4]
    tc: [1, 4]
    wt: [1, 2, 4]
```

For the published paper you should re-train with the correct pediatric
label map; the baseline numbers on the old checkpoint are not
directly comparable.

## Citation

If you use this code in a paper please cite the accompanying preprint
(`paper/RESEARCH_PAPER.md`) and the BraTS-PEDs 2023 challenge paper.

## Acknowledgments

Built on [MONAI](https://monai.io/) and [PyTorch](https://pytorch.org/).
SwinUNETR pretrained weights courtesy of MONAI's SSL release.
