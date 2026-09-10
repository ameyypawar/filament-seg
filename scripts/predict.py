"""Tiled full-resolution inference -> submission CSV.

Runs the trained network over every 2048x2048 image as overlapping tiles
(never downsampled -- filaments are a few pixels wide and resampling would
destroy them), blends the tiles with a Hann window
(``filament_seg.model.tiled_predict``) so seams don't fragment a filament
that crosses a tile boundary, masks to the solar disk, thresholds, forms
instances and writes the submission format.

    python scripts/predict.py --checkpoint outputs/model_best.pt --subset test
    python scripts/predict.py --checkpoint outputs/model_best.pt --subset val --out outputs/val_model.csv
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from filament_seg.config import REPO_ROOT, SPLIT_PATH, TRAIN_ANNOTATIONS, ensure_output_dir
from filament_seg.data import deduplicate_by_file, load_annotations
from filament_seg.dataset import load_disk_geometry
from filament_seg.model import build_model, select_device, tiled_predict
from filament_seg.postprocess import PostprocessParams, binary_to_instances
from filament_seg.rle import build_submission, labels_to_rles, write_submission

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"


def _logit(p: float) -> float:
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9
    return float(np.log(p / (1.0 - p)))


def resolve_stems(args: argparse.Namespace) -> list[str]:
    if args.subset == "test":
        # Every cached stem that isn't an annotated (train) observation is a
        # test one -- walking the flat cache avoids re-globbing raw JPEGs.
        annotations = load_annotations(args.annotations)
        train_stems = set(annotations.by_file_stem())
        flat_dir = Path(args.cache_dir) / "flat"
        return sorted(p.stem for p in flat_dir.glob("*.png") if p.stem not in train_stems)

    annotations = load_annotations(args.annotations)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    image_ids = deduplicate_by_file(annotations, split["val"])
    return sorted({annotations.images[i].stem for i in image_ids})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset", choices=["test", "val"], default="test")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--bridge-gap", type=int, default=12)
    parser.add_argument("--out", default=None)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    device = select_device()
    print(f"device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        encoder=checkpoint.get("encoder", "resnet34"),
        in_channels=checkpoint.get("in_channels", 2),
        weights=None,  # trained weights load next -- pretrained ones would just be overwritten
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cache_dir = Path(args.cache_dir)
    disk_geometry = load_disk_geometry(cache_dir / "disk.json")
    flat_dir = cache_dir / "flat"

    stems = resolve_stems(args)
    if not stems:
        raise SystemExit(
            "no observations found for this subset -- has scripts/preprocess.py run?"
        )
    print(f"predicting {len(stems)} {args.subset} observations")

    postprocess_params = PostprocessParams(min_area=args.min_area, bridge_gap=args.bridge_gap)
    threshold_logit = _logit(args.threshold)

    predictions: dict[str, list] = {}
    for n, stem in enumerate(stems, start=1):
        flat_u8 = cv2.imread(str(flat_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if flat_u8 is None:
            print(f"  warning: no cached flat image for {stem}, skipping")
            continue

        disk = disk_geometry[stem]
        shape = flat_u8.shape
        radius = disk.radius_map(shape).astype(np.float32)
        x = np.stack([flat_u8.astype(np.float32) / 255.0, radius], axis=0)

        logits = tiled_predict(model, x, tile=args.tile, overlap=args.overlap, device=device)
        disk_mask = disk.mask(shape)
        binary = ((logits > threshold_logit) & disk_mask).astype(np.uint8)
        labels = binary_to_instances(binary, postprocess_params, restrict_to=disk_mask)
        predictions[stem] = labels_to_rles(labels)

        if n % 25 == 0 or n == len(stems):
            print(f"  {n}/{len(stems)}", flush=True)

    frame = build_submission(predictions)
    out_path = Path(args.out or (ensure_output_dir() / f"submission_{args.subset}.csv"))
    write_submission(frame, out_path)

    n_instances = len(frame)
    empty = sum(1 for v in predictions.values() if not v)
    print(
        f"\n{n_instances} instances over {len(predictions)} images "
        f"({n_instances / max(len(predictions), 1):.2f} per image, {empty} empty)"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
