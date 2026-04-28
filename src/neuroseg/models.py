"""Models for NeuroSeg.

Three architectures, all optionally conditioned on a per-sample modality
availability mask via a learned embedding injected at the encoder bottleneck:

    1. ModalityAwareSegResNet  - lightweight 3D CNN baseline (MONAI SegResNet).
    2. ModalityAwareSwinUNETR  - strong transformer backbone; supports loading
       MONAI's self-supervised pretrained encoder (BraTS-Adult ssl weights)
       for transfer learning to the pediatric domain.
    3. VanillaSegResNet / VanillaSwinUNETR - controls without the embedding.

The embedding is a 2-layer MLP over the (C,) binary mask; a linear projection
adapts the embedding dimension to the bottleneck channel count and is broadcast
additively across spatial dims. This matches the intuition that "which
modalities are present" is a global, per-sample conditioning signal.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

# Project root = two levels up from this file (src/neuroseg/models.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR

log = logging.getLogger("neuroseg.models")


# ---------------------------------------------------------------------------
# Modality embedding
# ---------------------------------------------------------------------------
class ModalityEmbedding(nn.Module):
    """Binary availability mask -> dense embedding vector."""
    def __init__(self, num_modalities: int = 4, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_modalities, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:  # (B, N) -> (B, D)
        return self.net(mask)


# ---------------------------------------------------------------------------
# SegResNet with embedding
# ---------------------------------------------------------------------------
class ModalityAwareSegResNet(nn.Module):
    """SegResNet with a modality-embedding added at the deepest encoder block."""
    def __init__(self, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                 init_filters: int = 16, in_channels: int = 4,
                 out_channels: int = 3, dropout_prob: float = 0.2,
                 emb_dim: int = 128, num_modalities: int = 4,
                 use_embedding: bool = True):
        super().__init__()
        self.num_modalities = num_modalities
        self.emb_dim = emb_dim
        self.use_embedding = use_embedding

        self.backbone = SegResNet(
            blocks_down=blocks_down, blocks_up=blocks_up,
            init_filters=init_filters, in_channels=in_channels,
            out_channels=out_channels, dropout_prob=dropout_prob,
        )
        self.mod_emb = ModalityEmbedding(num_modalities, emb_dim)
        bottleneck_c = init_filters * (2 ** (len(blocks_down) - 1))
        self.emb_proj = nn.Linear(emb_dim, bottleneck_c)

        self._current_emb: Optional[torch.Tensor] = None
        self._hook_handle = self._register_hook()

    def _register_hook(self):
        last_block = list(self.backbone.down_layers)[-1]

        def _hook(module, inputs, output):
            if (self._current_emb is None) or (not self.use_embedding):
                return output
            B, C, *spatial = output.shape
            proj = self.emb_proj(self._current_emb)       # (B, C)
            proj = proj.view(B, C, *([1] * len(spatial))) # broadcast
            return output + proj

        return last_block.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor,
                modality_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        if modality_mask is None:
            modality_mask = torch.ones(B, self.num_modalities,
                                       dtype=torch.float32, device=x.device)
        self._current_emb = self.mod_emb(modality_mask)
        try:
            logits = self.backbone(x)
        finally:
            self._current_emb = None
        return logits

    def __del__(self):
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# SwinUNETR with embedding (supports pretrained weights)
# ---------------------------------------------------------------------------
class ModalityAwareSwinUNETR(nn.Module):
    """3D SwinUNETR backbone + modality embedding at the deepest stage.

    Pretrained encoder weights: MONAI publishes self-supervised pretrained
    SwinUNETR weights trained on a BraTS-Adult corpus
    (model_swinvit.pt / ssl_pretrained.pth). Pass `pretrained_ssl_path` to
    warm-start the encoder.
    """
    def __init__(self, img_size=(128, 128, 128), in_channels: int = 4,
                 out_channels: int = 3, feature_size: int = 48,
                 use_checkpoint: bool = True, emb_dim: int = 128,
                 num_modalities: int = 4, use_embedding: bool = True,
                 pretrained_ssl_path: Optional[str] = None):
        super().__init__()
        self.num_modalities = num_modalities
        self.emb_dim = emb_dim
        self.use_embedding = use_embedding

        # `use_v2=True` disables deprecated behaviour; MONAI >=1.3 supports it.
        try:
            self.backbone = SwinUNETR(
                img_size=img_size, in_channels=in_channels,
                out_channels=out_channels, feature_size=feature_size,
                use_checkpoint=use_checkpoint, use_v2=True,
            )
        except TypeError:
            # Older MONAI doesn't accept use_v2
            self.backbone = SwinUNETR(
                img_size=img_size, in_channels=in_channels,
                out_channels=out_channels, feature_size=feature_size,
                use_checkpoint=use_checkpoint,
            )

        # SwinUNETR's deepest features are 16x feature_size channels.
        bottleneck_c = 16 * feature_size
        self.mod_emb = ModalityEmbedding(num_modalities, emb_dim)
        self.emb_proj = nn.Linear(emb_dim, bottleneck_c)

        self._current_emb: Optional[torch.Tensor] = None
        self._hook_handle = self._register_hook()

        if pretrained_ssl_path is not None:
            self.load_ssl_pretrained(pretrained_ssl_path)

    def _register_hook(self):
        # swinViT has `.layers4` as the last stage in MONAI; hook it.
        swin = self.backbone.swinViT
        target = swin.layers4

        def _hook(module, inputs, output):
            if (self._current_emb is None) or (not self.use_embedding):
                return output
            if isinstance(output, (list, tuple)):
                return output  # safety: don't modify multi-tuple outputs
            B, C, *spatial = output.shape
            proj = self.emb_proj(self._current_emb)
            proj = proj.view(B, C, *([1] * len(spatial)))
            return output + proj

        return target.register_forward_hook(_hook)

    def load_ssl_pretrained(self, path: str) -> None:
        """Warm-start the SwinViT encoder from MONAI's SSL checkpoint.

        Accepts either the raw `model_swinvit.pt` state_dict or a wrapped
        dict with a `state_dict` / `model` key.

        The SSL checkpoint was pretrained with a **single-channel** input, so
        ``patch_embed.proj.weight`` has shape [C_out, 1, kD, kH, kW].  We
        inflate it to [C_out, in_channels, kD, kH, kW] by repeating across
        the channel axis and dividing by in_channels to preserve the expected
        activation scale (equivalent to the "average initialisation" used by
        ViT and BEiT papers).  Any remaining shape-mismatched keys are logged
        and silently skipped so the rest of the encoder still loads.
        """
        # Resolve relative paths from the project root so this works
        # regardless of the shell's current working directory.
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        if not resolved.exists():
            raise FileNotFoundError(
                f"SSL pretrained weights not found at: {resolved}\n"
                f"  (original path in config: {path!r})\n"
                f"  Download from: https://github.com/Project-MONAI/"
                f"MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
            )
        ckpt = torch.load(str(resolved), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt

        # Strip common key prefixes from MONAI SSL checkpoints.
        clean: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            nk = k
            for p in ("module.", "swinViT.", "swin_vit.", "net."):
                if nk.startswith(p):
                    nk = nk[len(p):]
            clean[nk] = v

        # Get the target model's state dict to compare shapes.
        model_sd = self.backbone.swinViT.state_dict()
        in_ch = self.backbone.swinViT.patch_embed.proj.weight.shape[1]

        filtered: Dict[str, torch.Tensor] = {}
        skipped = []
        for k, v in clean.items():
            if k not in model_sd:
                continue  # extra key — just skip (handled by strict=False)
            target_shape = model_sd[k].shape
            if v.shape == target_shape:
                filtered[k] = v
            elif k == "patch_embed.proj.weight" and v.shape[1] == 1:
                # Inflate 1-channel → in_channels by repeating and rescaling.
                filtered[k] = v.repeat(1, in_ch, 1, 1, 1) / in_ch
                log.info(
                    "  patch_embed.proj.weight: inflated 1-ch → %d-ch "
                    "(÷%d rescaling)", in_ch, in_ch)
            else:
                skipped.append(f"{k}: ckpt={tuple(v.shape)} "
                               f"model={tuple(target_shape)}")

        if skipped:
            log.warning("  Skipped %d shape-mismatched key(s):", len(skipped))
            for s in skipped:
                log.warning("    %s", s)

        missing, unexpected = self.backbone.swinViT.load_state_dict(
            filtered, strict=False)
        log.info("Loaded SSL pretrained SwinViT from %s", path)
        log.info("  missing keys: %d | unexpected: %d",
                 len(missing), len(unexpected))

    def forward(self, x: torch.Tensor,
                modality_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        if modality_mask is None:
            modality_mask = torch.ones(B, self.num_modalities,
                                       dtype=torch.float32, device=x.device)
        self._current_emb = self.mod_emb(modality_mask)
        try:
            logits = self.backbone(x)
        finally:
            self._current_emb = None
        return logits

    def __del__(self):
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(cfg: Dict, device: Optional[torch.device] = None) -> nn.Module:
    name = cfg.get("name", "segresnet").lower()
    common = dict(in_channels=cfg.get("in_channels", 4),
                  out_channels=cfg.get("out_channels", 3),
                  num_modalities=cfg.get("num_modalities", 4),
                  emb_dim=cfg.get("emb_dim", 128),
                  use_embedding=cfg.get("use_embedding", True))
    if name in ("segresnet", "modalityawaresegresnet"):
        model = ModalityAwareSegResNet(
            blocks_down=tuple(cfg.get("blocks_down", (1, 2, 2, 4))),
            blocks_up=tuple(cfg.get("blocks_up", (1, 1, 1))),
            init_filters=cfg.get("init_filters", 16),
            dropout_prob=cfg.get("dropout_prob", 0.2),
            **common,
        )
    elif name in ("swinunetr", "modalityawareswinunetr"):
        model = ModalityAwareSwinUNETR(
            img_size=tuple(cfg.get("img_size", (128, 128, 128))),
            feature_size=cfg.get("feature_size", 48),
            use_checkpoint=cfg.get("use_checkpoint", True),
            pretrained_ssl_path=cfg.get("pretrained_ssl_path"),
            **common,
        )
    else:
        raise ValueError(f"Unknown model.name: {name!r}")

    if device is not None:
        model = model.to(device)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str,
                    device: Optional[torch.device] = None,
                    strict: bool = False) -> nn.Module:
    """Load a training checkpoint. Tolerates minor key mismatches by default."""
    ckpt = torch.load(ckpt_path, map_location=device or "cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    log.info("Loaded checkpoint %s (missing=%d unexpected=%d)",
             ckpt_path, len(missing), len(unexpected))
    return model
