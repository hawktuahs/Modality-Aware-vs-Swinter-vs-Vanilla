#!/usr/bin/env python
"""Single-subject inference. Saves NIfTI masks and (optionally) PNG overlay.

Usage:
    python scripts/predict.py \
        --config configs/segresnet_modality.yaml \
        --ckpt   outputs/runs/segresnet_modality_fold0/best.pth \
        --subject-dir BraTS-PEDs-v1/Training/BraTS-PED-00001-000 \
        [--modality-mask 1 0 1 1]  [--no-tta]  [--visualize]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from neuroseg.utils import (
    Config, inject_cuda_dlls, seed_everything, pick_device, get_logger,
)
inject_cuda_dlls()

import numpy as np
import torch
import nibabel as nib
from monai.transforms import Compose

from neuroseg.inference import InferenceConfig, predict_probs
from neuroseg.metrics import per_case_metrics
from neuroseg.models import build_model, load_checkpoint
from neuroseg.postprocess import PostProcessConfig, postprocess
from neuroseg.transforms import LabelMap, build_inference_transforms

MODALITIES = ("t1c", "t1n", "t2f", "t2w")


def build_single_subject_input(subject_dir: Path, modality_mask_flags,
                               pixdim=(1.0, 1.0, 1.0)):
    """Load the 4 modalities (zero-filling any that the user marked absent).

    Returns: (image_tensor(1,4,H,W,D), ref_nib, affine, modalities_present)
    """
    sid = subject_dir.name
    paths = [subject_dir / f"{sid}-{m}.nii.gz" for m in MODALITIES]

    present_disk = [int(p.exists()) for p in paths]
    if modality_mask_flags is None:
        modality_mask_flags = present_disk
    # Final mask: intersection of on-disk existence and user-requested.
    final_mask = [int(a * b) for a, b in zip(present_disk, modality_mask_flags)]

    ref_nib = None
    volumes = []
    for i, p in enumerate(paths):
        if final_mask[i]:
            nb = nib.load(str(p))
            ref_nib = ref_nib or nb
            arr = nb.get_fdata(dtype=np.float32)
        else:
            arr = None
        volumes.append(arr)

    if ref_nib is None:
        raise RuntimeError(f"No usable modalities for {sid}")

    # Fill missing modalities with zeros of the reference shape.
    ref_shape = ref_nib.get_fdata(dtype=np.float32).shape
    filled = []
    for v, mk in zip(volumes, final_mask):
        if v is None:
            filled.append(np.zeros(ref_shape, dtype=np.float32))
        else:
            filled.append(v.astype(np.float32))
    stacked = np.stack(filled, axis=0)        # (4, H, W, D)

    # Per-channel nonzero z-score normalisation.
    for c in range(4):
        ch = stacked[c]
        nz = ch != 0
        if nz.any():
            ch[nz] = (ch[nz] - ch[nz].mean()) / (ch[nz].std() + 1e-8)
            stacked[c] = ch

    tensor = torch.from_numpy(stacked).unsqueeze(0)
    return tensor, ref_nib, ref_nib.affine, final_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--subject-dir", required=True)
    parser.add_argument("--modality-mask", nargs=4, type=int, default=None,
                        help="e.g. 1 0 1 1 to simulate missing T1n")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-pp", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    log = get_logger(level="INFO")
    cfg = Config.from_yaml(Path(args.config))
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    device = pick_device("auto")

    subject_dir = Path(args.subject_dir)
    image, ref_nib, affine, mask_vec = build_single_subject_input(
        subject_dir, args.modality_mask,
        pixdim=tuple(cfg["data"].get("pixdim", (1.0, 1.0, 1.0))))
    image = image.to(device)
    log.info("Subject %s | mask %s", subject_dir.name, mask_vec)

    model = build_model(cfg["model"], device=device)
    load_checkpoint(model, args.ckpt, device=device, strict=False)

    infer_cfg_dict = cfg.get("inference", {})
    icfg = InferenceConfig(
        roi_size=tuple(infer_cfg_dict.get("roi_size", (128, 128, 128))),
        sw_batch_size=int(infer_cfg_dict.get("sw_batch_size", 4)),
        overlap=float(infer_cfg_dict.get("overlap", 0.5)),
        mode=infer_cfg_dict.get("mode", "gaussian"),
        use_tta=(not args.no_tta) and bool(infer_cfg_dict.get("use_tta", True)),
        flip_axes=tuple(infer_cfg_dict.get("flip_axes", (2, 3, 4))),
    )
    pp_dict = cfg.get("postprocess", {})
    pp_cfg = None if args.no_pp else PostProcessConfig(
        threshold=float(pp_dict.get("threshold", 0.5)),
        min_component_voxels=dict(pp_dict.get("min_component_voxels",
                                              {"TC": 100, "WT": 100, "ET": 50})),
        et_min_volume_voxels=int(pp_dict.get("et_min_volume_voxels", 200)),
        enable_et_rule=bool(pp_dict.get("enable_et_rule", True)),
    )

    t0 = time.time()
    mm = torch.tensor(mask_vec, dtype=torch.float32,
                      device=device).unsqueeze(0)
    probs = predict_probs(model, image, mm, icfg, device=device)
    probs_np = probs[0].cpu().numpy()
    bin_masks = (postprocess(probs_np, pp_cfg) if pp_cfg is not None
                 else (probs_np >= 0.5).astype(np.uint8))
    log.info("Inference took %.1fs", time.time() - t0)

    out_dir = Path(args.out_dir) if args.out_dir else BASE_DIR / "outputs" / "predictions" / subject_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    region_names = ("TC", "WT", "ET")
    for i, name in enumerate(region_names):
        nib.save(nib.Nifti1Image(bin_masks[i].astype(np.uint8), affine),
                 str(out_dir / f"{subject_dir.name}-pred-{name}.nii.gz"))
        nib.save(nib.Nifti1Image(probs_np[i].astype(np.float32), affine),
                 str(out_dir / f"{subject_dir.name}-prob-{name}.nii.gz"))

    # Combined competition-style mask: 3=ET, 1=TC\ET, 2=WT\(TC).
    combo = np.zeros_like(bin_masks[0], dtype=np.uint8)
    combo[bin_masks[1] == 1] = 2
    combo[(bin_masks[0] == 1) & (bin_masks[2] == 0)] = 1
    combo[bin_masks[2] == 1] = 3
    nib.save(nib.Nifti1Image(combo, affine),
             str(out_dir / f"{subject_dir.name}-pred-combined.nii.gz"))

    # Ground-truth Dice if label available.
    seg_path = subject_dir / f"{subject_dir.name}-seg.nii.gz"
    if seg_path.exists():
        seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
        lm = LabelMap.from_dict(cfg["dataset"].get("label_map"))
        def to_region(vals):
            m = np.zeros_like(seg, dtype=np.uint8)
            for v in vals:
                m[seg == v] = 1
            return m
        gt = np.stack([to_region(lm.tc), to_region(lm.wt), to_region(lm.et)],
                      axis=0)
        cm = per_case_metrics(bin_masks, gt, sid=subject_dir.name,
                              compute_hd95=False)
        log.info("Dice  TC=%.4f  WT=%.4f  ET=%.4f",
                 cm.dice["TC"], cm.dice["WT"], cm.dice["ET"])

    log.info("Wrote masks -> %s", out_dir)


if __name__ == "__main__":
    main()
