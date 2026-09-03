"""Local re-implementation of the competition's evaluation.

Leaderboard metric (competition Overview -> "Leaderboard Ranking"):

    PQ(Y, Y_hat) = sum_{(y, y_hat) in TP} IoU(y, y_hat)
                   / (|TP| + 0.5 |FP| + 0.5 |FN|)

which is exactly standard Panoptic Quality (Kirillov et al., CVPR 2019) with the
usual IoU > 0.5 matching rule. Above 0.5 the match is provably unique, so greedy
assignment by descending IoU is optimal.

Two design notes:

* **Aggregation is ambiguous.** The competition states the formula over sets but
  does not say whether TP/FP/FN are pooled across the whole test set or whether
  per-image PQ is averaged. Both are computed here (``pooled`` and
  ``per_image``); calibrate which one matches by comparing a submission's local
  score against the public leaderboard. Treat this as an open question until
  confirmed -- it materially changes how much small images matter.

* **The rubric wants more than PQ.** 70% of the final score is quantitative and
  explicitly includes the *distributions* of Dice and IoU plus the counts of
  one-to-many (fragmentation) and many-to-one (over-merge) relations. Those are
  produced here too, so the report can be written from real numbers.

Marginal-value rule worth remembering while tuning post-processing: a predicted
instance that fails to reach IoU 0.5 costs 0.5 FP + 0.5 FN = 1.0 in the
denominator, while simply not predicting it costs only 0.5. Emitting a doubtful
instance is *twice* as expensive as staying silent, so a prediction is only
worth making when its chance of clearing the threshold is above roughly
``PQ / (2 * E[IoU | hit])`` -- around 30% at a PQ of 0.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import FRAGMENT_OVERLAP_THRESHOLD, PQ_IOU_THRESHOLD
from .rle import Rle, iou_matrix, rle_areas, rle_intersection_area


@dataclass
class ImageResult:
    """Per-image matching outcome."""

    image_id: str
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    matched_ious: list[float] = field(default_factory=list)
    #: Ground-truth segments split across >= 2 predictions (fragmentation).
    one_to_many: int = 0
    #: Predictions covering >= 2 ground-truth segments (over-merge).
    many_to_one: int = 0

    @property
    def iou_sum(self) -> float:
        return float(sum(self.matched_ious))

    @property
    def denominator(self) -> float:
        return self.tp + 0.5 * self.fp + 0.5 * self.fn

    @property
    def pq(self) -> float:
        if self.denominator == 0:
            # No ground truth and no prediction: nothing to get wrong.
            return 1.0
        return self.iou_sum / self.denominator

    @property
    def matched_dices(self) -> list[float]:
        # Dice and IoU are algebraically linked: Dice = 2*IoU / (1 + IoU).
        return [2.0 * i / (1.0 + i) for i in self.matched_ious]


def match_instances(
    gt_rles: Sequence[Rle],
    pred_rles: Sequence[Rle],
    iou_threshold: float = PQ_IOU_THRESHOLD,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Greedily match predictions to ground truth by descending IoU.

    Returns ``(matches, iou_matrix)`` where each match is ``(gt_i, pred_j, iou)``.
    """
    ious = iou_matrix(gt_rles, pred_rles)
    matches: list[tuple[int, int, float]] = []
    if ious.size == 0:
        return matches, ious

    candidates = np.argwhere(ious > iou_threshold)
    order = np.argsort([-ious[g, p] for g, p in candidates]) if len(candidates) else []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for idx in order:
        g, p = candidates[idx]
        g, p = int(g), int(p)
        if g in used_gt or p in used_pred:
            continue
        used_gt.add(g)
        used_pred.add(p)
        matches.append((g, p, float(ious[g, p])))
    return matches, ious


def count_fragmentation(
    gt_rles: Sequence[Rle],
    pred_rles: Sequence[Rle],
    overlap_threshold: float = FRAGMENT_OVERLAP_THRESHOLD,
) -> tuple[int, int]:
    """Count one-to-many and many-to-one relations.

    A ground-truth segment is *fragmented* when at least two predictions each
    cover more than ``overlap_threshold`` of its area. A prediction is an
    *over-merge* when it covers more than ``overlap_threshold`` of at least two
    ground-truth segments.
    """
    if not gt_rles or not pred_rles:
        return 0, 0

    gt_area = rle_areas(gt_rles)
    inter = np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    ious = iou_matrix(gt_rles, pred_rles)
    for g in range(len(gt_rles)):
        for p in range(len(pred_rles)):
            if ious[g, p] > 0:
                inter[g, p] = rle_intersection_area(gt_rles[g], pred_rles[p])

    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(gt_area[:, None] > 0, inter / gt_area[:, None], 0.0)

    significant = coverage > overlap_threshold
    one_to_many = int((significant.sum(axis=1) >= 2).sum())
    many_to_one = int((significant.sum(axis=0) >= 2).sum())
    return one_to_many, many_to_one


def evaluate_image(
    image_id: str,
    gt_rles: Sequence[Rle],
    pred_rles: Sequence[Rle],
    iou_threshold: float = PQ_IOU_THRESHOLD,
    with_fragmentation: bool = True,
) -> ImageResult:
    matches, _ = match_instances(gt_rles, pred_rles, iou_threshold)
    tp = len(matches)
    result = ImageResult(
        image_id=image_id,
        n_gt=len(gt_rles),
        n_pred=len(pred_rles),
        tp=tp,
        fp=len(pred_rles) - tp,
        fn=len(gt_rles) - tp,
        matched_ious=[m[2] for m in matches],
    )
    if with_fragmentation:
        result.one_to_many, result.many_to_one = count_fragmentation(gt_rles, pred_rles)
    return result


def summarize(results: Sequence[ImageResult]) -> dict:
    """Aggregate per-image results into the numbers the rubric asks for."""
    if not results:
        return {"n_images": 0}

    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    iou_sum = sum(r.iou_sum for r in results)
    denom = tp + 0.5 * fp + 0.5 * fn

    all_ious = np.array([i for r in results for i in r.matched_ious], dtype=np.float64)
    all_dices = np.array([d for r in results for d in r.matched_dices], dtype=np.float64)
    per_image = np.array([r.pq for r in results], dtype=np.float64)

    def distribution(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"n": 0}
        percentiles = np.percentile(values, [5, 25, 50, 75, 95])
        return {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p05": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "median": float(percentiles[2]),
            "p75": float(percentiles[3]),
            "p95": float(percentiles[4]),
            "max": float(values.max()),
        }

    return {
        "n_images": len(results),
        # The two candidate aggregations -- see the module docstring.
        "pq_pooled": float(iou_sum / denom) if denom else float("nan"),
        "pq_per_image_mean": float(per_image.mean()),
        # Standard PQ = SQ * RQ decomposition, useful for diagnosis:
        # SQ says "how good are the masks I got right", RQ says "how often am I right".
        "sq": float(iou_sum / tp) if tp else 0.0,
        "rq": float(tp / denom) if denom else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": sum(r.n_gt for r in results),
        "n_pred": sum(r.n_pred for r in results),
        "one_to_many": sum(r.one_to_many for r in results),
        "many_to_one": sum(r.many_to_one for r in results),
        "iou_distribution": distribution(all_ious),
        "dice_distribution": distribution(all_dices),
        "pq_per_image_distribution": distribution(per_image),
    }


def evaluate(
    gt: dict[str, Sequence[Rle]],
    pred: dict[str, Sequence[Rle]],
    iou_threshold: float = PQ_IOU_THRESHOLD,
    with_fragmentation: bool = True,
) -> tuple[dict, list[ImageResult]]:
    """Score a prediction dict against a ground-truth dict, keyed by image."""
    results = [
        evaluate_image(
            image_id,
            gt[image_id],
            pred.get(image_id, []),
            iou_threshold=iou_threshold,
            with_fragmentation=with_fragmentation,
        )
        for image_id in sorted(gt)
    ]
    # Predictions for images absent from the ground truth are pure false positives.
    for image_id in sorted(set(pred) - set(gt)):
        n_pred = len(pred[image_id])
        results.append(
            ImageResult(image_id=image_id, n_gt=0, n_pred=n_pred, tp=0, fp=n_pred, fn=0)
        )
    return summarize(results), results


def format_report(summary: dict) -> str:
    """Human-readable summary for the terminal and the technical report."""
    if not summary.get("n_images"):
        return "no images evaluated"

    def dist_line(name: str, dist: dict) -> str:
        if not dist.get("n"):
            return f"  {name:<6} (none)"
        return (
            f"  {name:<6} n={dist['n']:<5} mean={dist['mean']:.3f}  "
            f"p05={dist['p05']:.3f} p25={dist['p25']:.3f} med={dist['median']:.3f} "
            f"p75={dist['p75']:.3f} p95={dist['p95']:.3f}"
        )

    return "\n".join(
        [
            f"images={summary['n_images']}  gt={summary['n_gt']}  pred={summary['n_pred']}",
            f"PQ (pooled)      {summary['pq_pooled']:.4f}",
            f"PQ (per-image)   {summary['pq_per_image_mean']:.4f}",
            f"SQ={summary['sq']:.4f}  RQ={summary['rq']:.4f}",
            f"TP={summary['tp']}  FP={summary['fp']}  FN={summary['fn']}",
            f"fragmentation: one-to-many={summary['one_to_many']}  "
            f"many-to-one={summary['many_to_one']}",
            "distributions over matched pairs:",
            dist_line("IoU", summary["iou_distribution"]),
            dist_line("Dice", summary["dice_distribution"]),
            dist_line("PQ/img", summary["pq_per_image_distribution"]),
        ]
    )
