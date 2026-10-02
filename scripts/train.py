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
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from filament_seg.config import CACHE_DIR, REPO_ROOT, SPLIT_PATH, TRAIN_ANNOTATIONS
from filament_seg.data import (
    Annotations,
    deduplicate_by_file,
    load_annotations,
    load_split,
    records_for_stems,
    sample_stems,
)
from filament_seg.dataset import FilamentCrops, load_disk_geometry, load_model_input
from filament_seg.disk import Disk
from filament_seg.metrics import evaluate_image, summarize
from filament_seg.model import DiceBCELoss, build_model, select_device, tiled_predict
from filament_seg.postprocess import PostprocessParams, logits_to_instances
from filament_seg.rle import labels_to_rles

#: Validation tiling knobs are fixed rather than exposed as flags: they must
#: match what scripts/predict.py will actually submit with, or "best val PQ"
#: stops meaning anything.
VAL_TILE = 512
VAL_OVERLAP = 128

#: Fixed independently of --seed, so retraining with another seed is still
#: judged on the same observations. sweep_postprocess.py samples with the same
#: seed, so these are always a subset of the sweep's tuning observations and
#: never touch the held-out half used to judge the final result.
VAL_SAMPLE_SEED = 0


def run_validation(
    model: torch.nn.Module,
    views_by_stem: dict[str, list[str]],
    annotations: Annotations,
    disk_geometry: dict[str, Disk],
    flat_dir: Path,
    device: torch.device,
    postprocess_params: PostprocessParams,
) -> dict:
    """Predict each observation once and score it against every annotator's view."""
    model.eval()
    results = []
    for stem, image_ids in views_by_stem.items():
        disk = disk_geometry[stem]
        x = load_model_input(flat_dir, stem, disk)
        if x is None:
            print(f"  warning: no cached flat image for {stem}, skipping")
            continue

        logits = tiled_predict(model, x, tile=VAL_TILE, overlap=VAL_OVERLAP, device=device)
        labels = logits_to_instances(logits, disk.mask(logits.shape), postprocess_params)
        pred_rles = labels_to_rles(labels)
        for image_id in image_ids:
            results.append(evaluate_image(image_id, annotations.gt_rles(image_id), pred_rles))

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
    parser.add_argument("--limit-val", type=int, default=24,
                        help="validation observations scored after every epoch")
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target",
        choices=["annotator", "consensus"],
        default="annotator",
        help="'consensus' trains against the per-stem soft-agreement mask instead of "
        "one annotator's view -- see filament_seg.dataset.FilamentCrops",
    )
    args = parser.parse_args()

    device = select_device()
    print(f"device: {device}")

    annotations = load_annotations(args.annotations)
    split = load_split(args.split)
    train_ids = deduplicate_by_file(annotations, split["train"], seed=args.seed)
    val_stems = sample_stems(annotations, split["val"], args.limit_val, seed=VAL_SAMPLE_SEED)
    val_views = records_for_stems(annotations, split["val"], val_stems)
    views_by_stem = {
        stem: [i for i in val_views if annotations.images[i].stem == stem] for stem in val_stems
    }
    print(f"train observations: {len(train_ids)}  val observations (capped): "
          f"{len(val_stems)} ({len(val_views)} annotator views)")

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
        target=args.target,
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
        n_batches = len(loader)
        for step, (x, y) in enumerate(loader, start=1):
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

            # Without this an epoch is a silent multi-minute block, and a
            # stalled run looks exactly like a slow one.
            if step % 25 == 0 or step == n_batches:
                rate = n_seen / (time.perf_counter() - start)
                print(
                    f"  epoch {epoch} step {step}/{n_batches} "
                    f"loss={running_loss / n_seen:.4f} {rate:.1f} crops/s",
                    flush=True,
                )
        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        val_summary = run_validation(
            model, views_by_stem, annotations, disk_geometry, cache_dir / "flat", device,
            postprocess_params,
        )
        val_pq = val_summary.get("pq_pooled", 0.0)
        elapsed = time.perf_counter() - start

        payload = {
            "model": model.state_dict(),
            "encoder": args.encoder,
            "in_channels": 2,
            "crop_size": args.crop_size,
            "target": args.target,
            "epoch": epoch,
            "val_pq": val_pq,
        }

        # Validation runs on --limit-val observations (24 by default), where
        # per-image PQ carries a standard error near 0.02. Keeping only the
        # best checkpoint lets one lucky epoch permanently displace a
        # genuinely better one, so the latest epoch is kept alongside it and
        # both can be scored on the full validation split afterwards.
        torch.save(payload, out_path.with_name(out_path.stem + "_last.pt"))

        improved = val_pq > best_pq
        if improved:
            best_pq = val_pq
            torch.save(payload, out_path)

        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_pq={val_pq:.4f}  sq={val_summary.get('sq', 0.0):.4f}  "
            f"rq={val_summary.get('rq', 0.0):.4f}  ({elapsed:.1f}s)"
            + ("  * saved" if improved else "")
        )

    print(f"best val PQ: {best_pq:.4f}  -> {out_path}")


if __name__ == "__main__":
    main()
