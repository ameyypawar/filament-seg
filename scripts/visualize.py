"""Render ground-truth and predicted filaments over the H-alpha image.

30% of the final score is qualitative, and one of its three stated criteria is
"the apparent morphology of predicted segmentations on H-Alpha images". This
produces the figures for that -- and, more usefully day to day, it is the
fastest way to see *why* a Panoptic Quality number is what it is. Fragmentation
and over-merge are obvious at a glance and invisible in a scalar.

    # worst validation images by PQ, ground truth vs prediction side by side
    python scripts/visualize.py --submission outputs/val_baseline.csv \\
        --subset val --sort worst --limit 12

    # just the raw detections on a few test images
    python scripts/visualize.py --subset test --limit 6
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from filament_seg.baseline import BaselineParams, segment
from filament_seg.config import (
    SPLIT_PATH,
    TEST_IMAGE_DIR,
    TRAIN_ANNOTATIONS,
    TRAIN_IMAGE_DIR,
    ensure_output_dir,
)
from filament_seg.data import deduplicate_by_file, load_annotations, test_image_paths
from filament_seg.disk import read_image
from filament_seg.metrics import evaluate_image
from filament_seg.rle import Rle, labels_to_rles, read_submission, rle_to_mask

# Distinct colours so neighbouring instances never blur into one blob.
_PALETTE = np.array(
    [
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
        [0, 206, 209], [154, 205, 50],
    ],
    dtype=np.float32,
)


def overlay(image: np.ndarray, rles: list[Rle], alpha: float = 0.55) -> np.ndarray:
    """Tint each instance a different colour over the grayscale image."""
    canvas = np.repeat(image[:, :, None].astype(np.float32), 3, axis=2)
    for i, rle in enumerate(rles):
        mask = rle_to_mask(rle)
        colour = _PALETTE[i % len(_PALETTE)]
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * colour
    return np.clip(canvas, 0, 255).astype(np.uint8)


def panel(ax, image: np.ndarray, rles: list[Rle], title: str) -> None:
    ax.imshow(overlay(image, rles))
    ax.set_title(f"{title}  ({len(rles)} filaments)", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["test", "train", "val"], default="val")
    parser.add_argument("--submission", default=None, help="CSV of predictions")
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--images", default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--sort",
        choices=["worst", "best", "name"],
        default="worst",
        help="worst/best need both a submission and ground truth",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or (ensure_output_dir() / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_submission(args.submission) if args.submission else None

    # --- Test subset: no ground truth, so just show the detections ---------
    if args.subset == "test":
        paths = test_image_paths(args.images or TEST_IMAGE_DIR)[: args.limit]
        for path in paths:
            image = read_image(path)
            if predictions is not None:
                rles = predictions.get(path.stem, [])
            else:
                labels, _ = segment(image, BaselineParams())
                rles = labels_to_rles(labels)
            fig, ax = plt.subplots(figsize=(6, 6))
            panel(ax, image, rles, path.stem)
            fig.tight_layout()
            fig.savefig(out_dir / f"{path.stem}.png", dpi=args.dpi)
            plt.close(fig)
            print(f"  wrote {out_dir / (path.stem + '.png')}")
        return

    # --- Train/val: ground truth alongside the prediction ------------------
    annotations = load_annotations(args.annotations)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    image_ids = deduplicate_by_file(annotations, split[args.subset])

    scored: list[tuple[float, str]] = []
    for image_id in image_ids:
        stem = annotations.images[image_id].stem
        if predictions is None:
            scored.append((0.0, image_id))
            continue
        result = evaluate_image(
            image_id,
            annotations.gt_rles(image_id),
            predictions.get(stem, []),
            with_fragmentation=False,
        )
        scored.append((result.pq, image_id))

    if args.sort == "worst" and predictions is not None:
        scored.sort(key=lambda t: t[0])
    elif args.sort == "best" and predictions is not None:
        scored.sort(key=lambda t: -t[0])
    else:
        scored.sort(key=lambda t: t[1])

    image_dir = Path(args.images or TRAIN_IMAGE_DIR)
    for pq, image_id in scored[: args.limit]:
        record = annotations.images[image_id]
        path = next(
            (p for p in [image_dir / record.file_name] + list(
                image_dir.glob(record.stem + ".*")) if p.exists()),
            None,
        )
        if path is None:
            print(f"  skipping {record.stem}: image file not found")
            continue

        image = read_image(path)
        gt = annotations.gt_rles(image_id)

        if predictions is None:
            labels, _ = segment(image, BaselineParams())
            pred = labels_to_rles(labels)
            title = "baseline (on the fly)"
        else:
            pred = predictions.get(record.stem, [])
            title = f"prediction  PQ={pq:.3f}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        panel(axes[0], image, gt, f"ground truth  {record.annotator}")
        panel(axes[1], image, pred, title)
        fig.suptitle(record.stem, fontsize=11)
        fig.tight_layout()
        name = f"{pq:.3f}_{record.stem}.png" if predictions else f"{record.stem}.png"
        fig.savefig(out_dir / name, dpi=args.dpi)
        plt.close(fig)
        print(f"  wrote {out_dir / name}")


if __name__ == "__main__":
    main()
