"""Test-time augmentation utilities.

The standard BraTS TTA is the 8-way flip set: for each combination of flipping
along the three spatial axes, run the network, un-flip the output, and
average. This costs 8x inference time but typically yields +0.005 - +0.02 Dice.

For efficiency the caller is expected to wrap the forward pass in a sliding-
window inferrer (see `inference.py`).
"""
from __future__ import annotations

from itertools import product
from typing import Callable, List, Sequence, Tuple

import torch

FlipAxes = Tuple[int, ...]


def flip_combinations(axes: Sequence[int] = (2, 3, 4)) -> List[FlipAxes]:
    """All 2^|axes| subsets of axes to flip (including the empty set)."""
    subsets: List[FlipAxes] = [()]
    for k in range(1, len(axes) + 1):
        for combo in _combinations(axes, k):
            subsets.append(tuple(combo))
    return subsets


def _combinations(items, k):
    if k == 0:
        yield ()
        return
    n = len(items)
    indices = list(range(k))
    while True:
        yield tuple(items[i] for i in indices)
        for i in reversed(range(k)):
            if indices[i] != i + n - k:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, k):
            indices[j] = indices[j - 1] + 1


def tta_predict(predictor: Callable[[torch.Tensor], torch.Tensor],
                x: torch.Tensor,
                flip_axes: Sequence[int] = (2, 3, 4),
                activation: str = "sigmoid") -> torch.Tensor:
    """Run the predictor 8 times over the flip group; average after inverting.

    `predictor` must accept a single tensor and return logits of the same
    spatial shape. The returned tensor is the averaged probability map.
    """
    combos = flip_combinations(flip_axes)
    probs: List[torch.Tensor] = []
    for combo in combos:
        xi = torch.flip(x, dims=list(combo)) if combo else x
        with torch.no_grad():
            yi = predictor(xi)
        if activation == "sigmoid":
            yi = torch.sigmoid(yi)
        elif activation == "softmax":
            yi = torch.softmax(yi, dim=1)
        if combo:
            yi = torch.flip(yi, dims=list(combo))
        probs.append(yi)
    return torch.stack(probs, dim=0).mean(dim=0)
