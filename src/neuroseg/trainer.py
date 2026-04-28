"""Training loop for NeuroSeg.

Features:
    * AMP / autocast on CUDA
    * Gradient clipping
    * Per-epoch validation with sliding-window inference
    * Early stopping on mean-Dice plateau
    * Cosine LR with optional warmup
    * Checkpoint saving (best + last) with full training state
    * CSV + TensorBoard logging
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from monai.data import DataLoader, Dataset as MonaiDataset
from monai.inferers import sliding_window_inference
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .metrics import aggregate_metrics, per_case_metrics
from .utils import get_logger


# ---------------------------------------------------------------------------
# Scheduler with warmup
# ---------------------------------------------------------------------------
class WarmupCosine:
    def __init__(self, optimizer, max_epochs: int, warmup_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            t = (epoch - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * t))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


@dataclass
class TrainerConfig:
    max_epochs: int = 200
    val_every: int = 1
    early_stop_patience: int = 30
    grad_clip: float = 1.0
    amp: bool = True
    base_lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    roi_size: tuple = (128, 128, 128)
    sw_batch_size: int = 4
    overlap: float = 0.5
    num_modalities: int = 4
    out_dir: Path = Path("outputs")
    run_name: str = "run"
    save_every_best_only: bool = True


class Trainer:
    def __init__(self, model: torch.nn.Module, loss_fn: Callable,
                 train_loader: DataLoader, val_loader: DataLoader,
                 cfg: TrainerConfig, device: torch.device,
                 optimizer: Optional[torch.optim.Optimizer] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.optimizer = optimizer or AdamW(
            model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        self.scheduler = WarmupCosine(self.optimizer, cfg.max_epochs,
                                      cfg.warmup_epochs, cfg.base_lr,
                                      cfg.min_lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and
                                           device.type == "cuda")
        self.log = get_logger("neuroseg.trainer")
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_best = self.run_dir / "best.pth"
        self.ckpt_last = self.run_dir / "last.pth"
        self.csv_path = self.run_dir / "metrics.csv"
        self._init_csv()

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        except Exception:
            self.tb = None

    # ---------------------------------------------------------------- I/O
    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "epoch", "lr", "train_loss",
                    "val_mean_dice", "val_dice_TC", "val_dice_WT", "val_dice_ET",
                    "elapsed_sec",
                ])

    def _log_row(self, row):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(row)

    # -------------------------------------------------------------- train
    def train(self) -> Dict[str, float]:
        best_mean = -1.0
        epochs_since_best = 0
        history: List[Dict[str, float]] = []
        t0 = time.time()

        for epoch in range(self.cfg.max_epochs):
            lr = self.scheduler.step(epoch)
            train_loss = self._train_one_epoch(epoch)

            do_val = ((epoch + 1) % self.cfg.val_every == 0)
            if do_val:
                metrics = self._validate()
                mean_d = metrics["MEAN"]["dice_mean"]
                tc_d = metrics["TC"]["dice_mean"]
                wt_d = metrics["WT"]["dice_mean"]
                et_d = metrics["ET"]["dice_mean"]
                self.log.info(
                    "epoch %03d | lr=%.2e | loss=%.4f | "
                    "meanD=%.4f TC=%.4f WT=%.4f ET=%.4f",
                    epoch + 1, lr, train_loss,
                    mean_d, tc_d, wt_d, et_d)
                if self.tb is not None:
                    self.tb.add_scalar("loss/train", train_loss, epoch)
                    self.tb.add_scalar("dice/mean", mean_d, epoch)
                    self.tb.add_scalar("dice/TC", tc_d, epoch)
                    self.tb.add_scalar("dice/WT", wt_d, epoch)
                    self.tb.add_scalar("dice/ET", et_d, epoch)
                self._log_row([epoch + 1, f"{lr:.6e}", f"{train_loss:.6f}",
                               f"{mean_d:.6f}", f"{tc_d:.6f}",
                               f"{wt_d:.6f}", f"{et_d:.6f}",
                               f"{time.time() - t0:.1f}"])
                history.append({"epoch": epoch + 1, "train_loss": train_loss,
                                "mean_dice": mean_d})

                is_best = mean_d > best_mean
                if is_best:
                    best_mean = mean_d
                    epochs_since_best = 0
                    self._save_checkpoint(epoch, mean_d, self.ckpt_best)
                else:
                    epochs_since_best += 1

                # Always save last
                self._save_checkpoint(epoch, mean_d, self.ckpt_last)

                if epochs_since_best >= self.cfg.early_stop_patience:
                    self.log.info("Early stopping: no improvement for %d epochs",
                                  self.cfg.early_stop_patience)
                    break
            else:
                self.log.info("epoch %03d | lr=%.2e | loss=%.4f",
                              epoch + 1, lr, train_loss)

        return {"best_mean_dice": best_mean,
                "epochs_trained": epoch + 1,
                "total_sec": time.time() - t0}

    # ----------------------------------------------------- single epoch
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total = 0.0
        n_batches = 0
        n_total = len(self.train_loader)
        t_epoch_start = time.time()

        print(f"\n{'─'*65}", flush=True)
        print(f"  Epoch {epoch + 1:03d}/{self.cfg.max_epochs}  "
              f"[{n_total} batches]", flush=True)
        print(f"{'─'*65}", flush=True)

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            mod = batch.get("modality_mask")
            if mod is None:
                mod = torch.ones(images.shape[0], self.cfg.num_modalities,
                                 dtype=torch.float32, device=self.device)
            else:
                mod = mod.to(self.device, non_blocking=True)
                if mod.ndim == 1:
                    mod = mod.unsqueeze(0)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                logits = self.model(images, mod)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += float(loss.detach().item())
            n_batches += 1

            # ── per-batch progress ────────────────────────────────────
            elapsed = time.time() - t_epoch_start
            avg_t   = elapsed / n_batches
            eta_sec = avg_t * (n_total - n_batches)
            eta_str = time.strftime("%M:%S", time.gmtime(eta_sec))
            bar_w   = 30
            filled  = int(bar_w * n_batches / n_total)
            bar     = "█" * filled + "░" * (bar_w - filled)
            print(
                f"  [{bar}] {n_batches:3d}/{n_total}  "
                f"loss={total/n_batches:.4f}  "
                f"ETA {eta_str}",
                end="\r", flush=True,
            )

        print(flush=True)   # newline after final \r
        return total / max(1, n_batches)

    # ------------------------------------------------------- validation
    def _validate(self) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        cases = []
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                B = images.shape[0]
                mm = torch.ones(B, self.cfg.num_modalities,
                                dtype=torch.float32, device=self.device)

                def predictor(x):
                    # x may have sw_batch_size rows (e.g. 4) while mm has B rows (1).
                    # expand (not repeat) is zero-copy and handles the size mismatch.
                    mm_exp = mm.expand(x.shape[0], -1)
                    return self.model(x, mm_exp)

                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=self.cfg.roi_size,
                    sw_batch_size=self.cfg.sw_batch_size,
                    predictor=predictor, overlap=self.cfg.overlap,
                    mode="gaussian",
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                gt = labels.cpu().numpy()
                sids = batch.get("sid", [f"val_{i}" for i in range(B)])
                if isinstance(sids, torch.Tensor):
                    sids = sids.tolist()
                if isinstance(sids, (list, tuple)) and len(sids) != B:
                    sids = [str(s) for s in sids]
                for b in range(B):
                    pred_bin = (probs[b] >= 0.5).astype(np.uint8)
                    cases.append(per_case_metrics(
                        pred_bin, gt[b].astype(np.uint8),
                        sid=str(sids[b]) if isinstance(sids, (list, tuple))
                            else str(sids),
                        compute_hd95=False,  # fast path during training
                    ))
        return aggregate_metrics(cases)

    # ------------------------------------------------------------ ckpt
    def _save_checkpoint(self, epoch: int, mean_dice: float, path: Path):
        torch.save({
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "mean_dice": mean_dice,
        }, path)
