"""Dataset discovery and cross-validation splits for BraTS-PEDs-v1.

Label convention (BraTS-PEDs 2024 / 2023 challenge):
    The raw segmentation volume contains integer labels {0, 1, 2, 3, 4}.
    The canonical label semantics for BraTS-PEDs are:
        1  ET   (Enhancing Tumor)
        2  NET  (Non-Enhancing Tumor Core)
        3  CC   (Cystic Component)
        4  ED   (Peritumoral Edema)
    Evaluation sub-regions:
        WT (Whole Tumor)  = {1, 2, 3, 4}
        TC (Tumor Core)   = {1, 2, 3}          (ET + NET + CC)
        ET (Enhancing)    = {1}
IMPORTANT: Verify this mapping against the data card that came with your
download. The mapping is exposed in configs/*.yaml under `dataset.label_map`
so it can be overridden without editing code if a different convention applies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold, train_test_split


MODALITIES = ("t1c", "t1n", "t2f", "t2w")


@dataclass
class Subject:
    sid: str
    images: List[str]          # [t1c, t1n, t2f, t2w]
    label: Optional[str]       # seg path or None (challenge val set)

    def to_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {"sid": self.sid, "image": list(self.images)}
        if self.label is not None:
            d["label"] = self.label
        return d


def discover_subjects(dataset_dir: Path, require_label: bool = True,
                      modalities: Sequence[str] = MODALITIES) -> List[Subject]:
    """Scan a BraTS-PEDs split directory and return one Subject per folder.

    Skips any subject missing one of the requested modalities (or the seg mask
    when `require_label=True`).
    """
    subjects: List[Subject] = []
    for subj_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        sid = subj_dir.name
        imgs = [subj_dir / f"{sid}-{m}.nii.gz" for m in modalities]
        seg = subj_dir / f"{sid}-seg.nii.gz"
        if not all(p.exists() for p in imgs):
            continue
        if require_label and not seg.exists():
            continue
        subjects.append(Subject(
            sid=sid,
            images=[str(p) for p in imgs],
            label=str(seg) if seg.exists() else None,
        ))
    return subjects


def stratified_kfold_splits(subjects: List[Subject], n_splits: int = 5,
                            seed: int = 42) -> List[Tuple[List[Subject], List[Subject]]]:
    """Return [(train_list, val_list), ...] of length n_splits.

    BraTS-PEDs does not come with per-subject clinical labels for
    stratification, so we use deterministic KFold on sorted subject IDs. This
    is reproducible and the splits are stable across machines.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    subjects = list(subjects)  # KFold needs indexable
    splits: List[Tuple[List[Subject], List[Subject]]] = []
    for train_idx, val_idx in kf.split(np.arange(len(subjects))):
        train = [subjects[i] for i in train_idx]
        val = [subjects[i] for i in val_idx]
        splits.append((train, val))
    return splits


def holdout_test_split(subjects: List[Subject], test_frac: float = 0.1,
                       seed: int = 42) -> Tuple[List[Subject], List[Subject]]:
    """Carve a permanent test set off the provided subject list.

    The remaining 'dev' pool is what you run K-fold CV over. This is the
    standard methodology: K-fold for HPO/model selection on `dev`, then a
    single final evaluation on `test` for the paper.
    """
    dev, test = train_test_split(
        subjects, test_size=test_frac, random_state=seed,
        shuffle=True,
    )
    return dev, test


def to_monai_list(subjects: List[Subject]) -> List[Dict[str, object]]:
    """Convert to the list-of-dicts format MONAI transforms expect."""
    out: List[Dict[str, object]] = []
    for s in subjects:
        d: Dict[str, object] = {"image": list(s.images), "sid": s.sid}
        if s.label is not None:
            d["label"] = s.label
        out.append(d)
    return out
