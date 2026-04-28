"""Utilities: seeding, device handling, Windows-DLL injection, logging, config."""
from __future__ import annotations

import json
import logging
import os
import random
import site
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

LOGGER_NAME = "neuroseg"


# ---------------------------------------------------------------------------
# Windows CUDA DLL shim (same logic as user's original scripts, cleaned up).
# ---------------------------------------------------------------------------
def inject_cuda_dlls() -> None:
    """On Windows, some NVIDIA pip wheels ship DLLs in site-packages. Add them
    to PATH / DLL directories so torch can find cuDNN, cuBLAS, cuda_nvrtc.

    No-op on non-Windows or if the NVIDIA site-packages layout is not present.
    """
    if os.name != "nt":
        return
    try:
        candidates = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
        for p in candidates:
            for mod in ("cudnn", "cublas", "cuda_nvrtc", "cuda_runtime"):
                bin_path = os.path.join(p, "nvidia", mod, "bin")
                if os.path.isdir(bin_path):
                    try:
                        os.add_dll_directory(bin_path)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        # Best-effort; never crash training on DLL path issues.
        pass


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Prioritise throughput on a 4060.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    try:
        from monai.utils import set_determinism
        set_determinism(seed=seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"CUDA:{device.index or 0} {p.name} ({p.total_memory / 1e9:.1f} GB)"
    return "CPU"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = LOGGER_NAME, level: str = "INFO",
               log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid double-adding handlers on repeat calls
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Flat dictionary-style config with attribute access and YAML I/O."""
    data: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["data"]:
            return self.__dict__["data"][key]
        raise AttributeError(key)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        self.data.update(other)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cls(data=cfg)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.data, fh, sort_keys=False)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (not trainable_only) or p.requires_grad)
