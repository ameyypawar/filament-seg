"""Train the U-Net filament segmenter and select checkpoints on Panoptic Quality.

Training runs on random 2-channel crops (``filament_seg.dataset.FilamentCrops``);
validation, after every epoch, runs full-resolution tiled inference
(``filament_seg.model.tiled_predict``) over a fixed subset of held-out
observations, forms instances the same way the submission pipeline will
(``filament_seg.postprocess.binary_to_instances``), and scores them with the
real evaluator (``filament_seg.metrics``).

The selection metric is PQ, not the training loss's Dice term: Dice cannot
see detection precision (missing or hallucinating a whole filament barely
moves it), which is exactly the failure mode the competition punishes most.

    python scripts/train.py --epochs 30 --crop-size 512 --encoder resnet34
    python scripts/train.py --epochs 1 --limit-val 4 --crops-per-image 2  # smoke test
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from filament_seg.config import REPO_ROOT, SPLIT_PATH, TRAIN_ANNOTATIONS
from filament_seg.data import Annotations, deduplicate_by_file, load_annotations
from filament_seg.dataset import FilamentCrops, load_disk_geometry
from filament_seg.disk import Disk
from filament_seg.metrics import evaluate_image, summarize
from filament_seg.model import DiceBCELoss, build_model, select_device, tiled_predict
from filament_seg.postprocess import PostprocessParams, binary_to_instances
from filament_seg.rle import labels_to_rles

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"

#: Validation tiling knobs are fixed rather than exposed as flags: they must
#: match what scripts/predict.py will actually submit with, or "best val PQ"
#: stops meaning anything.
VAL_TILE = 512
VAL_OVERLAP = 128


def run_validation(
    model: torch.nn.Module,
    val_ids: list[str],
    annotations: Annotations,
    disk_geometry: dict[str, Disk],
    flat_dir: Path,
    device: torch.device,
    postprocess_params: PostprocessParams,
) -> dict:
    model.eval()
    results = []
    for image_id in val_ids:
        record = annotations.images[image_id]
        stem = record.stem
        shape = (record.height, record.width)

        flat_u8 = cv2.imread(str(flat_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if flat_u8 is None:
            print(f"  warning: no cached flat image for {stem}, skipping")
            continue

        disk = disk_geometry[stem]
        radius = disk.radius_map(shape).astype(np.float32)
        x = np.stack([flat_u8.astype(np.float32) / 255.0, radius], axis=0)

        logits = tiled_predict(model, x, tile=VAL_TILE, overlap=VAL_OVERLAP, device=device)
        disk_mask = disk.mask(shape)
        binary = ((logits > 0.0) & disk_mask).astype(np.uint8)  # logit > 0 <=> prob > 0.5
        labels = binary_to_instances(binary, postprocess_params, restrict_to=disk_mask)

        pred_rles = labels_to_rles(labels)
        gt_rles = annotations.gt_rles(image_id)
        results.append(evaluate_image(image_id, gt_rles, pred_rles))

    return summarize(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crops-per-image", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "model_best.pt"))
    parser.add_argument("--limit-val", type=int, default=24)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = select_device()
    print(f"device: {device}")

    annotations = load_annotations(args.annotations)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_ids = deduplicate_by_file(annotations, split["train"], seed=args.seed)
    # Sorted (not shuffled) so the same subset is scored every epoch -- the
    # point is a stable trend, not a representative sample.
    val_ids = deduplicate_by_file(annotations, split["val"], seed=args.seed)[: args.limit_val]
    print(f"train observations: {len(train_ids)}  val observations (capped): {len(val_ids)}")

    cache_dir = Path(args.cache_dir)
    disk_geometry = load_disk_geometry(cache_dir / "disk.json")

    dataset = FilamentCrops(
        train_ids,
        annotations,
        cache_dir=cache_dir,
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        augment=True,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    model = build_model(encoder=args.encoder, in_channels=2, weights="imagenet").to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    postprocess_params = PostprocessParams()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_pq = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        start = time.perf_counter()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        val_summary = run_validation(
            model, val_ids, annotations, disk_geometry, cache_dir / "flat", device,
            postprocess_params,
        )
        val_pq = val_summary.get("pq_pooled", 0.0)
        elapsed = time.perf_counter() - start

        improved = val_pq > best_pq
        if improved:
            best_pq = val_pq
            torch.save(
                {
                    "model": model.state_dict(),
                    "encoder": args.encoder,
                    "in_channels": 2,
                    "crop_size": args.crop_size,
                    "epoch": epoch,
                    "val_pq": val_pq,
                },
                out_path,
            )

        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_pq={val_pq:.4f}  sq={val_summary.get('sq', 0.0):.4f}  "
            f"rq={val_summary.get('rq', 0.0):.4f}  ({elapsed:.1f}s)"
            + ("  * saved" if improved else "")
        )

    print(f"best val PQ: {best_pq:.4f}  -> {out_path}")


if __name__ == "__main__":
    main()
