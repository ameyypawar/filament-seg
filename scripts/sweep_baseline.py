"""Grid-search the baseline detector's thresholds against local Panoptic Quality.

Tuning against Dice would be a mistake here: the first baseline run scored
SQ 0.63 (the masks it did match were fine) against RQ 0.15 (it almost never
matched), so the whole loss was detection precision, which Dice cannot see.

The expensive half of the pipeline -- reading the image, finding the disk,
dividing out limb darkening -- does not depend on any swept parameter, so it is
computed once per image and reused across the whole grid.

    python scripts/sweep_baseline.py --n-images 40
    python scripts/sweep_baseline.py --n-images 40 --k 2.0 2.4 2.8 --min-area 800 1500
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import itertools
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from filament_seg.config import (
    SPLIT_PATH,
    TRAIN_ANNOTATIONS,
    TRAIN_IMAGE_DIR,
    ensure_output_dir,
)
from filament_seg.data import deduplicate_by_file, load_annotations
from filament_seg.disk import detect_disk, flatten, read_image
from filament_seg.metrics import evaluate_image
from filament_seg.postprocess import PostprocessParams, binary_to_instances
from filament_seg.rle import Rle, labels_to_rles

_GT: dict[str, list[Rle]] = {}
_GRID: list[dict] = []


def _init(gt: dict, grid: list[dict]) -> None:
    global _GT, _GRID
    _GT, _GRID = gt, grid


def _score_one_image(payload: tuple[str, str]) -> dict[int, tuple]:
    """Score every grid point on a single image, sharing the preprocessing."""
    image_id, path = payload
    image = read_image(path)
    disk = detect_disk(image)
    disk_mask = disk.mask(image.shape, shrink=0.96)
    flat = cv2.GaussianBlur(flatten(image, disk), (0, 0), sigmaX=2.0)

    inside = flat[disk_mask]
    median = float(np.median(inside))
    mad = float(np.median(np.abs(inside - median)))
    sigma = 1.4826 * mad if mad > 0 else float(inside.std())

    gt = _GT[image_id]
    out: dict[int, tuple] = {}
    for index, point in enumerate(_GRID):
        binary = (flat < median - point["k"] * max(sigma, 1e-6)) & disk_mask
        labels = binary_to_instances(
            binary,
            PostprocessParams(
                open_radius=point["open_radius"],
                close_radius=point["close_radius"],
                bridge_gap=point["bridge_gap"],
                min_area=point["min_area"],
            ),
            restrict_to=disk_mask,
        )
        result = evaluate_image(
            image_id, gt, labels_to_rles(labels), with_fragmentation=False
        )
        out[index] = (
            result.n_gt, result.n_pred, result.tp, result.fp, result.fn,
            result.iou_sum,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--images", default=str(TRAIN_IMAGE_DIR))
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--k", type=float, nargs="+", default=[1.6, 2.0, 2.4, 2.8, 3.2])
    parser.add_argument("--min-area", type=int, nargs="+", default=[400, 900, 1600])
    parser.add_argument("--bridge-gap", type=int, nargs="+", default=[12])
    parser.add_argument("--close-radius", type=int, nargs="+", default=[5])
    parser.add_argument("--open-radius", type=int, nargs="+", default=[2])
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    image_ids = deduplicate_by_file(annotations, split["val"])
    random.Random(args.seed).shuffle(image_ids)
    image_ids = sorted(image_ids[: args.n_images])

    image_dir = Path(args.images)
    payloads: list[tuple[str, str]] = []
    gt: dict[str, list[Rle]] = {}
    for image_id in image_ids:
        record = annotations.images[image_id]
        matches = list(image_dir.glob(record.stem + ".*"))
        if not matches:
            continue
        payloads.append((image_id, str(matches[0])))
        gt[image_id] = annotations.gt_rles(image_id)

    grid = [
        {
            "k": k,
            "min_area": min_area,
            "bridge_gap": bridge_gap,
            "close_radius": close_radius,
            "open_radius": open_radius,
        }
        for k, min_area, bridge_gap, close_radius, open_radius in itertools.product(
            args.k, args.min_area, args.bridge_gap, args.close_radius, args.open_radius
        )
    ]
    print(f"{len(payloads)} images x {len(grid)} grid points")

    totals = {i: np.zeros(6, dtype=np.float64) for i in range(len(grid))}
    with ProcessPoolExecutor(
        max_workers=args.workers or None, initializer=_init, initargs=(gt, grid)
    ) as pool:
        for n, per_image in enumerate(pool.map(_score_one_image, payloads), start=1):
            for index, values in per_image.items():
                totals[index] += np.asarray(values, dtype=np.float64)
            if n % 10 == 0 or n == len(payloads):
                print(f"  {n}/{len(payloads)}", flush=True)

    rows = []
    for index, point in enumerate(grid):
        n_gt, n_pred, tp, fp, fn, iou_sum = totals[index]
        denominator = tp + 0.5 * fp + 0.5 * fn
        rows.append(
            {
                **point,
                "pq": float(iou_sum / denominator) if denominator else 0.0,
                "sq": float(iou_sum / tp) if tp else 0.0,
                "rq": float(tp / denominator) if denominator else 0.0,
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
                "n_pred": int(n_pred), "n_gt": int(n_gt),
            }
        )
    rows.sort(key=lambda r: -r["pq"])

    header = (
        f"{'k':>5} {'min_area':>9} {'gap':>4} {'close':>6} "
        f"{'PQ':>7} {'SQ':>6} {'RQ':>6} {'TP':>5} {'FP':>6} {'FN':>5} {'pred/gt':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows[: args.top]:
        print(
            f"{row['k']:>5.1f} {row['min_area']:>9} {row['bridge_gap']:>4} "
            f"{row['close_radius']:>6} {row['pq']:>7.4f} {row['sq']:>6.3f} "
            f"{row['rq']:>6.3f} {row['tp']:>5} {row['fp']:>6} {row['fn']:>5} "
            f"{row['n_pred'] / max(row['n_gt'], 1):>8.2f}"
        )

    out = ensure_output_dir() / "sweep_baseline.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
