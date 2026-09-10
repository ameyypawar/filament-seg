"""Grid-search post-processing knobs against local Panoptic Quality, for the model.

``PostprocessParams`` currently defaults to ``open_radius=2, close_radius=5,
bridge_gap=12, min_area=400``. Those were tuned for the classical baseline's
speckly threshold output (see scripts/sweep_baseline.py). A neural network's
probability map is far cleaner, and morphological opening with radius 2 wipes
out anything thinner than about 5 px -- filaments are only a handful of pixels
wide, so those defaults may be actively destroying thin filaments and barbs.
Training's per-epoch validation (scripts/train.py) uses these same defaults,
so the reported val PQ is probably understated.

The expensive half of the pipeline -- tiled full-resolution inference
(``filament_seg.model.tiled_predict``) -- does not depend on any swept
parameter, so it is run once per validation observation and the resulting
logit map is cached to disk as float16 (2048x2048 float16 is 8 MB); the whole
grid then reuses that cache. This is the same trick scripts/sweep_baseline.py
uses for disk detection and flattening.

    python scripts/sweep_postprocess.py --checkpoint outputs/model_best.pt
    python scripts/sweep_postprocess.py --checkpoint outputs/model_best.pt --n-images 40 --open-radius 0 1
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
import torch

from filament_seg.config import (
    OUTPUT_ROOT,
    REPO_ROOT,
    SPLIT_PATH,
    TRAIN_ANNOTATIONS,
    ensure_output_dir,
)
from filament_seg.data import deduplicate_by_file, load_annotations
from filament_seg.dataset import load_disk_geometry
from filament_seg.disk import Disk
from filament_seg.metrics import evaluate_image
from filament_seg.model import build_model, tiled_predict
from filament_seg.postprocess import PostprocessParams, binary_to_instances
from filament_seg.rle import Rle, labels_to_rles

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"

_GT: dict[str, list[Rle]] = {}
_GRID: list[dict] = []
_DISK: dict[str, Disk] = {}
_OUT_DIR = Path()


def _logit(p: float) -> float:
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9
    return float(np.log(p / (1.0 - p)))


def _cache_path(out_dir: Path, stem: str) -> Path:
    return out_dir / f"{stem}.npy"


def ensure_logits(
    payloads: list[tuple[str, str]],
    checkpoint_path: str,
    disk_geometry: dict[str, Disk],
    flat_dir: Path,
    out_dir: Path,
    tile: int,
    overlap: int,
    device: torch.device,
    force: bool,
) -> None:
    """Cache ``tiled_predict``'s full-resolution logits for every stem, once."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted({stem for _, stem in payloads})
    pending = [s for s in stems if force or not _cache_path(out_dir, s).exists()]
    if not pending:
        print(f"logits: {len(stems)}/{len(stems)} already cached, skipping model load")
        return

    print(f"device: {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        encoder=checkpoint.get("encoder", "resnet34"),
        in_channels=checkpoint.get("in_channels", 2),
        weights=None,  # trained weights load next -- pretrained ones would just be overwritten
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"computing logits for {len(pending)}/{len(stems)} observations")
    for n, stem in enumerate(pending, start=1):
        flat_u8 = cv2.imread(str(flat_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if flat_u8 is None:
            print(f"  warning: no cached flat image for {stem}, skipping")
            continue
        disk = disk_geometry[stem]
        radius = disk.radius_map(flat_u8.shape).astype(np.float32)
        x = np.stack([flat_u8.astype(np.float32) / 255.0, radius], axis=0)
        logits = tiled_predict(model, x, tile=tile, overlap=overlap, device=device)
        np.save(_cache_path(out_dir, stem), logits.astype(np.float16))
        if n % 5 == 0 or n == len(pending):
            print(f"  {n}/{len(pending)}", flush=True)


def _init(gt: dict, grid: list[dict], disk_geometry: dict, out_dir: Path) -> None:
    global _GT, _GRID, _DISK, _OUT_DIR
    _GT, _GRID, _DISK, _OUT_DIR = gt, grid, disk_geometry, out_dir


def _score_one_image(payload: tuple[str, str]) -> dict[int, tuple]:
    """Score every grid point on one image, sharing its cached logit map."""
    image_id, stem = payload
    logits = np.load(_cache_path(_OUT_DIR, stem)).astype(np.float32)
    disk = _DISK[stem]
    disk_mask = disk.mask(logits.shape)

    gt = _GT[image_id]
    out: dict[int, tuple] = {}
    for index, point in enumerate(_GRID):
        binary = ((logits > point["threshold_logit"]) & disk_mask).astype(np.uint8)
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_ROOT / "val_logits"))
    parser.add_argument("--n-images", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="recompute cached logits even if present"
    )
    parser.add_argument(
        "--threshold", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7]
    )
    parser.add_argument("--min-area", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--bridge-gap", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--close-radius", type=int, nargs="+", default=[3, 5, 7])
    # 0 disables opening entirely (filament_seg.postprocess.clean_binary skips
    # it) -- worth trying since opening is exactly what can erase thin barbs.
    parser.add_argument("--open-radius", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device)

    annotations = load_annotations(args.annotations)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    image_ids = deduplicate_by_file(annotations, split["val"])
    random.Random(args.seed).shuffle(image_ids)
    image_ids = sorted(image_ids[: args.n_images])

    cache_dir = Path(args.cache_dir)
    disk_geometry = load_disk_geometry(cache_dir / "disk.json")
    flat_dir = cache_dir / "flat"
    out_dir = Path(args.out_dir)

    payloads: list[tuple[str, str]] = []
    gt: dict[str, list[Rle]] = {}
    for image_id in image_ids:
        record = annotations.images[image_id]
        stem = record.stem
        if stem not in disk_geometry:
            continue
        payloads.append((image_id, stem))
        gt[image_id] = annotations.gt_rles(image_id)

    ensure_logits(
        payloads, args.checkpoint, disk_geometry, flat_dir, out_dir,
        args.tile, args.overlap, device, args.force,
    )

    grid = [
        {
            "threshold": threshold,
            "threshold_logit": _logit(threshold),
            "min_area": min_area,
            "bridge_gap": bridge_gap,
            "close_radius": close_radius,
            "open_radius": open_radius,
        }
        for threshold, min_area, bridge_gap, close_radius, open_radius in itertools.product(
            args.threshold, args.min_area, args.bridge_gap, args.close_radius,
            args.open_radius,
        )
    ]
    print(f"{len(payloads)} images x {len(grid)} grid points")

    totals = {i: np.zeros(6, dtype=np.float64) for i in range(len(grid))}
    with ProcessPoolExecutor(
        max_workers=args.workers or None,
        initializer=_init, initargs=(gt, grid, disk_geometry, out_dir),
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
                "threshold": point["threshold"],
                "min_area": point["min_area"],
                "bridge_gap": point["bridge_gap"],
                "close_radius": point["close_radius"],
                "open_radius": point["open_radius"],
                "pq": float(iou_sum / denominator) if denominator else 0.0,
                "sq": float(iou_sum / tp) if tp else 0.0,
                "rq": float(tp / denominator) if denominator else 0.0,
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
                "n_pred": int(n_pred), "n_gt": int(n_gt),
            }
        )
    rows.sort(key=lambda r: -r["pq"])

    header = (
        f"{'thr':>5} {'min_area':>9} {'gap':>4} {'close':>6} {'open':>5} "
        f"{'PQ':>7} {'SQ':>6} {'RQ':>6} {'TP':>5} {'FP':>6} {'FN':>5} {'pred/gt':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows[: args.top]:
        print(
            f"{row['threshold']:>5.2f} {row['min_area']:>9} {row['bridge_gap']:>4} "
            f"{row['close_radius']:>6} {row['open_radius']:>5} {row['pq']:>7.4f} "
            f"{row['sq']:>6.3f} {row['rq']:>6.3f} {row['tp']:>5} {row['fp']:>6} "
            f"{row['fn']:>5} {row['n_pred'] / max(row['n_gt'], 1):>8.2f}"
        )

    out = ensure_output_dir() / "sweep_postprocess.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
