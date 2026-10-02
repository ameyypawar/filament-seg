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
from pathlib import Path

from filament_seg.config import (
    CACHE_DIR,
    SPLIT_PATH,
    TEST_IMAGE_DIR,
    TRAIN_ANNOTATIONS,
    ensure_output_dir,
)
from filament_seg.data import load_annotations, load_split, stems_of, test_image_paths
from filament_seg.dataset import load_disk_geometry, load_model_input
from filament_seg.model import TTA_MODES, ensemble_predict, load_trained, select_device
from filament_seg.postprocess import PostprocessParams, logits_to_instances
from filament_seg.rle import build_submission, labels_to_rles, write_submission


def resolve_stems(args: argparse.Namespace) -> list[str]:
    if args.subset == "test":
        # The test folder itself, not the preprocessing cache, defines what
        # must be predicted: an image missing from a partial cache would
        # otherwise drop out of the submission and cost all its filaments.
        return sorted(p.stem for p in test_image_paths(TEST_IMAGE_DIR))
    annotations = load_annotations(args.annotations)
    return stems_of(annotations, load_split(args.split)["val"])


def main() -> None:
    defaults = PostprocessParams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="one or more checkpoints; several are averaged as an ensemble")
    parser.add_argument("--subset", choices=["test", "val"], default="test")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    # Every post-processing knob sweep_postprocess.py tunes must be settable
    # here, or what it finds could not be applied to the submission.
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--min-area", type=int, default=defaults.min_area)
    parser.add_argument("--bridge-gap", type=int, default=defaults.bridge_gap)
    parser.add_argument("--open-radius", type=int, default=defaults.open_radius)
    parser.add_argument("--close-radius", type=int, default=defaults.close_radius)
    parser.add_argument("--min-confidence", type=float, default=defaults.min_confidence)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N observations")
    parser.add_argument("--tta", choices=list(TTA_MODES), default="none",
                        help="average logits over flipped/rotated copies of each image")
    parser.add_argument("--out", default=None)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = parser.parse_args()

    stems = resolve_stems(args)
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"no {args.subset} observations found")

    cache_dir = Path(args.cache_dir)
    disk_geometry = load_disk_geometry(cache_dir / "disk.json")
    flat_dir = cache_dir / "flat"
    uncached = [s for s in stems if s not in disk_geometry or not (flat_dir / f"{s}.png").exists()]
    if uncached:
        raise SystemExit(
            f"{len(uncached)}/{len(stems)} {args.subset} observations are missing from the "
            f"preprocessing cache (e.g. {uncached[:3]}) -- run scripts/preprocess.py"
        )

    device = select_device()
    print(f"device: {device}")
    models = [load_trained(path, device) for path in args.checkpoint]
    print(f"models: {len(models)} ({', '.join(args.checkpoint)})")
    print(f"predicting {len(stems)} {args.subset} observations")

    params = PostprocessParams(
        threshold=args.threshold,
        min_area=args.min_area,
        bridge_gap=args.bridge_gap,
        open_radius=args.open_radius,
        close_radius=args.close_radius,
        min_confidence=args.min_confidence,
    )
    print(f"post-processing: tta={args.tta} {params}")

    predictions: dict[str, list] = {}
    for n, stem in enumerate(stems, start=1):
        disk = disk_geometry[stem]
        x = load_model_input(flat_dir, stem, disk)
        logits = ensemble_predict(models, x, tta=args.tta, tile=args.tile,
                                  overlap=args.overlap, device=device)
        labels = logits_to_instances(logits, disk.mask(logits.shape), params)
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
        f"({n_instances / max(len(predictions), 1):.2f} per image, {empty} with none)"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
