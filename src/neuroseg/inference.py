"""Sliding-window inference wrapper + TTA + post-processing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .postprocess import PostProcessConfig, postprocess
from .tta import tta_predict


@dataclass
class InferenceConfig:
    roi_size: Tuple[int, int, int] = (128, 128, 128)
    sw_batch_size: int = 4
    overlap: float = 0.5
    mode: str = "gaussian"
    use_tta: bool = True
    flip_axes: Sequence[int] = (2, 3, 4)


def predict_probs(model: torch.nn.Module,
                  image: torch.Tensor,
                  modality_mask: Optional[torch.Tensor],
                  cfg: InferenceConfig,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """Run sliding-window (+TTA) inference.

    Args
    ----
    image            : (B, C, H, W, D) tensor.
    modality_mask    : (B, C) float {0,1} mask (usually all-ones at test time).
    cfg              : InferenceConfig
    device           : target device; falls back to image.device.

    Returns the averaged sigmoid probability map, (B, 3, H, W, D).
    """
    model.eval()
    device = device or image.device
    image = image.to(device)
    if modality_mask is None:
        modality_mask = torch.ones(image.shape[0], 4, dtype=torch.float32,
                                    device=device)
    else:
        modality_mask = modality_mask.to(device)

    def _forward(patch: torch.Tensor) -> torch.Tensor:
        # modality_mask shape is [B, C] where B=1 (one image at a time).
        # sliding_window_inference sends sw_batch_size patches per call.
        # expand() stretches [1,C] → [sw_batch_size, C] with no memory copy.
        mm = modality_mask.expand(patch.shape[0], -1)
        return model(patch, mm)

    def _sliding(x: torch.Tensor) -> torch.Tensor:
        return sliding_window_inference(
            inputs=x,
            roi_size=cfg.roi_size,
            sw_batch_size=cfg.sw_batch_size,
            predictor=_forward,
            overlap=cfg.overlap,
            mode=cfg.mode,
        )

    with torch.no_grad():
        if cfg.use_tta:
            probs = tta_predict(_sliding, image,
                                flip_axes=cfg.flip_axes,
                                activation="sigmoid")
        else:
            logits = _sliding(image)
            probs = torch.sigmoid(logits)
    return probs


def predict_binary(model: torch.nn.Module,
                   image: torch.Tensor,
                   modality_mask: Optional[torch.Tensor],
                   cfg: InferenceConfig,
                   pp_cfg: Optional[PostProcessConfig] = None,
                   device: Optional[torch.device] = None) -> np.ndarray:
    """Return (B, 3, H, W, D) binary masks after TTA + post-processing."""
    probs = predict_probs(model, image, modality_mask, cfg, device)
    probs_np = probs.cpu().numpy()
    out = np.stack([postprocess(p, pp_cfg) for p in probs_np], axis=0)
    return out
