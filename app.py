#!/usr/bin/env python
"""NeuroSeg Flask demo: drop in 4 NIfTI modalities (+optional seg) and get a
TTA-averaged segmentation with per-region Dice (if seg provided), plus an
anatomical overlay figure.

The configuration is read from `configs/segresnet_modality.yaml` unless the
env var `NEUROSEG_CONFIG` points to something else. The checkpoint defaults
to `checkpoints/best_model.pth` but can be overridden via `NEUROSEG_CKPT`.

Usage:
    python app.py
    # visit http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from neuroseg.utils import (
    Config, inject_cuda_dlls, pick_device, get_logger, seed_everything,
)
inject_cuda_dlls()

import torch
import nibabel as nib

from neuroseg.inference import InferenceConfig, predict_probs
from neuroseg.metrics import per_case_metrics, REGIONS
from neuroseg.models import build_model, load_checkpoint
from neuroseg.postprocess import PostProcessConfig, postprocess
from neuroseg.transforms import LabelMap

log = get_logger("neuroseg.app")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024   # 1 GB uploads

MODALITY_ORDER = ("t1c", "t1n", "t2f", "t2w")
STATE: Dict[str, object] = {"model": None, "device": None, "cfg": None,
                            "icfg": None, "pp_cfg": None, "label_map": None}


def lazy_init() -> None:
    if STATE["model"] is not None:
        return
    cfg_path = os.environ.get("NEUROSEG_CONFIG",
                              str(BASE_DIR / "configs" / "segresnet_modality.yaml"))
    ckpt_path = os.environ.get("NEUROSEG_CKPT",
                               str(BASE_DIR / "checkpoints" / "best_model.pth"))
    log.info("config: %s", cfg_path)
    log.info("ckpt  : %s", ckpt_path)
    cfg = Config.from_yaml(Path(cfg_path))
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    device = pick_device("auto")

    model = build_model(cfg["model"], device=device)
    if Path(ckpt_path).exists():
        load_checkpoint(model, ckpt_path, device=device, strict=False)
    else:
        log.warning("Checkpoint not found at %s; predictions will be random.",
                    ckpt_path)
    model.eval()

    infer = cfg.get("inference", {})
    icfg = InferenceConfig(
        roi_size=tuple(infer.get("roi_size", (128, 128, 128))),
        sw_batch_size=int(infer.get("sw_batch_size", 4)),
        overlap=float(infer.get("overlap", 0.5)),
        mode=infer.get("mode", "gaussian"),
        use_tta=bool(infer.get("use_tta", True)),
        flip_axes=tuple(infer.get("flip_axes", (2, 3, 4))),
    )
    pp = cfg.get("postprocess", {})
    pp_cfg = PostProcessConfig(
        threshold=float(pp.get("threshold", 0.5)),
        min_component_voxels=dict(pp.get("min_component_voxels",
                                         {"TC": 100, "WT": 100, "ET": 50})),
        et_min_volume_voxels=int(pp.get("et_min_volume_voxels", 200)),
        enable_et_rule=bool(pp.get("enable_et_rule", True)),
    )
    STATE.update(model=model, device=device, cfg=cfg,
                 icfg=icfg, pp_cfg=pp_cfg,
                 label_map=LabelMap.from_dict(cfg["dataset"].get("label_map")))


def _normalize_channelwise(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for c in range(arr.shape[0]):
        nz = arr[c] != 0
        if nz.any():
            out[c][nz] = (arr[c][nz] - arr[c][nz].mean()) / (arr[c][nz].std() + 1e-8)
    return out


def _load_uploads(files) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, ...], object]:
    """Parse uploaded files. Keys must be 't1c','t1n','t2f','t2w'; 'seg' optional."""
    mods: List[Optional[np.ndarray]] = [None, None, None, None]
    ref_nib = None
    seg_arr = None
    for i, name in enumerate(MODALITY_ORDER):
        fs = files.get(name)
        if fs is None or fs.filename == "":
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        fs.save(tmp.name)
        tmp.close()
        nb = nib.load(tmp.name)
        ref_nib = ref_nib or nb
        mods[i] = nb.get_fdata(dtype=np.float32)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if ref_nib is None:
        raise ValueError("At least one MRI modality must be uploaded.")
    ref_idx = next(i for i, m in enumerate(mods) if m is not None)
    ref_shape = mods[ref_idx].shape
    mask_vec = tuple(int(m is not None) for m in mods)
    filled = [m if m is not None else np.zeros(ref_shape, dtype=np.float32)
              for m in mods]
    image = _normalize_channelwise(np.stack(filled, axis=0))

    seg_file = files.get("seg")
    if seg_file is not None and seg_file.filename:
        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        seg_file.save(tmp.name)
        tmp.close()
        seg_arr = nib.load(tmp.name).get_fdata().astype(np.int16)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return image, seg_arr, mask_vec, ref_nib


def _seg_to_regions(seg: np.ndarray, lm: LabelMap) -> np.ndarray:
    def mk(vals):
        m = np.zeros_like(seg, dtype=np.uint8)
        for v in vals:
            m[seg == v] = 1
        return m
    return np.stack([mk(lm.tc), mk(lm.wt), mk(lm.et)], axis=0)


def _overlay_png(image: np.ndarray, gt: Optional[np.ndarray],
                 pred: np.ndarray) -> str:
    """Return a base64-encoded PNG of a 6-slice axial overlay on T1c."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t1c = image[0]  # show on T1c
    z_idxs = np.linspace(t1c.shape[-1] // 4, 3 * t1c.shape[-1] // 4, 6).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()
    for ax, z in zip(axes, z_idxs):
        base = t1c[:, :, z]
        ax.imshow(np.rot90(base), cmap="gray")
        pred_z = pred[:, :, :, z]
        rgba = np.zeros(base.shape + (4,), dtype=np.float32)
        rgba[..., 0] = pred_z[0]    # TC -> R
        rgba[..., 1] = pred_z[1]    # WT -> G
        rgba[..., 2] = pred_z[2]    # ET -> B
        rgba[..., 3] = np.clip(pred_z.max(axis=0) * 0.5, 0, 1)
        ax.imshow(np.rot90(rgba), interpolation="none")
        ax.set_title(f"z={int(z)}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


INDEX_HTML = """
<!doctype html>
<html><head>
  <meta charset="utf-8">
  <title>NeuroSeg - Pediatric Brain Tumor Segmentation</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Arial, sans-serif;
           max-width: 920px; margin: 2em auto; padding: 0 1em; color: #222; }
    h1 { color: #0b4a7f; margin-bottom: 0.2em; }
    .sub { color: #666; margin-top: 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5em 1em; }
    label { font-weight: 600; }
    input[type=file] { width: 100%; }
    button { background: #0b4a7f; color: white; border: 0;
             padding: 0.7em 1.4em; border-radius: 6px;
             font-size: 1em; margin-top: 1em; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: wait; }
    .card { background: #f6f8fa; padding: 1em 1.4em; border-radius: 8px;
            margin: 1.4em 0; }
    .pill { display: inline-block; background: #0b4a7f; color:white;
            padding: 0.05em 0.6em; border-radius: 10px; font-size: 0.85em; }
  </style>
</head><body>
  <h1>NeuroSeg</h1>
  <p class="sub">Pediatric brain tumor segmentation &middot;
    <span class="pill">BraTS-PEDs</span>
    <span class="pill">TTA</span>
    <span class="pill">Modality-aware</span></p>

  <div class="card">
    <p>Upload the four MRI modalities (NIfTI). Leave any modality blank to
    simulate a missing scan - the model embeds the availability vector at the
    bottleneck and still produces a prediction.</p>
    <form method="POST" action="/predict" enctype="multipart/form-data"
          onsubmit="this.querySelector('button').disabled=true;">
      <div class="grid">
        <div><label>T1c&nbsp;</label>
          <input type="file" name="t1c" accept=".nii,.nii.gz"></div>
        <div><label>T1n&nbsp;</label>
          <input type="file" name="t1n" accept=".nii,.nii.gz"></div>
        <div><label>T2f (FLAIR)&nbsp;</label>
          <input type="file" name="t2f" accept=".nii,.nii.gz"></div>
        <div><label>T2w&nbsp;</label>
          <input type="file" name="t2w" accept=".nii,.nii.gz"></div>
        <div style="grid-column: 1 / span 2;">
          <label>Ground truth (optional, for Dice):</label>
          <input type="file" name="seg" accept=".nii,.nii.gz">
        </div>
      </div>
      <button type="submit">Run segmentation</button>
    </form>
  </div>
  <p style="color:#888;font-size:0.85em;">
    Model: {{ model_name }} &middot; Device: {{ device }} &middot;
    TTA: {{ tta }} &middot; Post-process: connected-component filter + ET rule
  </p>
</body></html>
"""


RESULT_HTML = """
<!doctype html>
<html><head>
<meta charset="utf-8">
<title>NeuroSeg result</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif;
         max-width: 1020px; margin: 2em auto; padding: 0 1em; color:#222; }
  h2 { color: #0b4a7f; }
  table { border-collapse: collapse; margin: 1em 0; }
  th, td { padding: 0.3em 1em; border-bottom: 1px solid #eee; }
  .card { background: #f6f8fa; padding: 1em 1.4em; border-radius: 8px; }
  img { max-width: 100%; border-radius: 6px; }
</style>
</head><body>
  <h2>Segmentation result</h2>
  <p>Inference time: {{ elapsed }} s &middot;
     Modality mask: {{ mask }} ({{ mods_present }}/4 present)</p>

  {% if dice %}
  <div class="card">
    <h3>Dice scores vs ground truth</h3>
    <table>
      <tr><th>Region</th><th>Dice</th><th>Sensitivity</th><th>Specificity</th></tr>
      {% for r in regions %}
      <tr>
        <td><strong>{{ r }}</strong></td>
        <td>{{ "%.4f"|format(dice[r]) }}</td>
        <td>{{ "%.4f"|format(sens[r]) if sens[r] == sens[r] else "n/a" }}</td>
        <td>{{ "%.4f"|format(spec[r]) if spec[r] == spec[r] else "n/a" }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  <h3>Overlay on T1c</h3>
  <p><em>TC = red &middot; WT = green &middot; ET = blue</em></p>
  <img src="data:image/png;base64,{{ overlay_b64 }}">

  <p><a href="/">&larr; new upload</a></p>
</body></html>
"""


@app.route("/")
def index():
    lazy_init()
    return render_template_string(INDEX_HTML,
        model_name=STATE["cfg"]["model"]["name"],
        device=str(STATE["device"]),
        tta="on" if STATE["icfg"].use_tta else "off")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        lazy_init()
        image, seg, mask_vec, ref_nib = _load_uploads(request.files)
        t0 = time.time()
        tensor = torch.from_numpy(image).unsqueeze(0).to(STATE["device"])
        mm = torch.tensor(mask_vec, dtype=torch.float32,
                          device=STATE["device"]).unsqueeze(0)
        probs = predict_probs(STATE["model"], tensor, mm,
                              STATE["icfg"], device=STATE["device"])
        probs_np = probs[0].cpu().numpy()
        bin_masks = postprocess(probs_np, STATE["pp_cfg"])
        elapsed = f"{time.time() - t0:.1f}"

        dice = sens = spec = None
        if seg is not None:
            gt = _seg_to_regions(seg, STATE["label_map"])
            cm = per_case_metrics(bin_masks, gt, sid="user_upload",
                                  compute_hd95=False)
            dice = cm.dice
            sens = cm.sens
            spec = cm.spec

        overlay_b64 = _overlay_png(image, seg, bin_masks)
        return render_template_string(
            RESULT_HTML,
            elapsed=elapsed,
            mask=list(mask_vec),
            mods_present=sum(mask_vec),
            dice=dice, sens=sens, spec=spec,
            regions=list(REGIONS),
            overlay_b64=overlay_b64,
        )
    except Exception as e:  # pragma: no cover
        log.exception("predict failed")
        return jsonify({"error": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/health")
def health():
    lazy_init()
    return jsonify({
        "device": str(STATE["device"]),
        "model": STATE["cfg"]["model"]["name"],
        "tta": STATE["icfg"].use_tta,
    })


if __name__ == "__main__":
    lazy_init()
    app.run(host="127.0.0.1", port=5000, debug=False)
