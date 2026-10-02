"""Pooled Panoptic Quality totals and the paired bootstrap that compares them.

The competition pools TP, FP, FN and the matched IoU sum over every annotator's
view of every observation. Keeping those six totals per observation, rather
than a per-observation PQ, is what lets comparisons resample observations and
still compute the pooled score exactly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .metrics import evaluate_image
from .rle import Rle

#: Per-observation totals, in this order, summed over the observation's views.
COUNT_FIELDS = ("n_gt", "n_pred", "tp", "fp", "fn", "iou_sum")


def view_totals(views: Sequence[str], gt: dict[str, list[Rle]], pred_rles: list[Rle]) -> np.ndarray:
    """One observation's totals: its predictions scored against each view, summed."""
    out = np.zeros(len(COUNT_FIELDS), dtype=np.float64)
    for image_id in views:
        r = evaluate_image(image_id, gt[image_id], pred_rles, with_fragmentation=False)
        out += (r.n_gt, r.n_pred, r.tp, r.fp, r.fn, r.iou_sum)
    return out


def pq_of(totals: np.ndarray) -> np.ndarray:
    """Pooled PQ from summed totals; works on any leading shape."""
    tp, fp, fn, iou_sum = (totals[..., COUNT_FIELDS.index(k)] for k in ("tp", "fp", "fn", "iou_sum"))
    denominator = tp + 0.5 * fp + 0.5 * fn
    return np.divide(iou_sum, denominator, out=np.zeros_like(iou_sum), where=denominator > 0)


def paired_bootstrap(
    per_stem_a: np.ndarray, per_stem_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """PQ(a) - PQ(b) on all observations, with a 95% interval from resampling them.

    Observations, not views, are resampled -- an observation's views share one
    set of predictions -- and both settings see the same resample each time.
    """
    n = per_stem_a.shape[0]
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    # einsum rather than `@`: macOS's Accelerate BLAS raises spurious
    # floating-point warnings in matmul, and this is far too small to need BLAS.
    deltas = (pq_of(np.einsum("bs,sk->bk", weights, per_stem_a))
              - pq_of(np.einsum("bs,sk->bk", weights, per_stem_b)))
    full = float(pq_of(per_stem_a.sum(axis=0)) - pq_of(per_stem_b.sum(axis=0)))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return full, float(low), float(high)
