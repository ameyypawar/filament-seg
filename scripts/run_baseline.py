"""Run the model-free baseline and write a submission CSV.

Phase 1 of the plan: prove the whole pipeline -- disk detection, thresholding,
instance formation, RLE encoding, CSV format -- against the real scorer before
any training happens.

    # leaderboard submission over the test set
    python scripts/run_baseline.py --out outputs/submission_baseline.csv

    # same detector over the validation split, for local scoring
    python scripts/run_baseline.py --subset val --out outputs/val_baseline.csv
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from filament_seg.baseline import BaselineParams, predict_rles
from filament_seg.config import (
    SPLIT_PATH,
    TEST_IMAGE_DIR,
    TRAIN_ANNOTATIONS,
    TRAIN_IMAGE_DIR,
    ensure_output_dir,
)
from filament_seg.data import load_annotations, test_image_paths
from filament_seg.postprocess import PostprocessParams
from filament_seg.rle import build_submission, write_submission

_PARAMS: BaselineParams | None = None


def _init(params: BaselineParams) -> None:
    global _PARAMS
    _PARAMS = params


def _predict(path_str: str) -> tuple[str, list]:
    path = Path(path_str)
    return path.stem, predict_rles(path, _PARAMS)


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    if args.subset == "test":
        return test_image_paths(args.images or TEST_IMAGE_DIR)

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    annotations = load_annotations(args.annotations)
    stems = {annotations.images[i].stem for i in split[args.subset]}
    directory = Path(args.images or TRAIN_IMAGE_DIR)
    return [p for p in test_image_paths(directory) if p.stem in stems]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["test", "train", "val"], default="test")
    parser.add_argument("--images", default=None, help="override the image directory")
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=0, help="debug: first N images")
    parser.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    # Detector knobs.
    parser.add_argument("--k", type=float, default=1.6)
    parser.add_argument("--disk-shrink", type=float, default=0.96)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--bridge-gap", type=int, default=12)
    parser.add_argument("--close-radius", type=int, default=5)
    parser.add_argument("--open-radius", type=int, default=2)
    args = parser.parse_args()

    params = BaselineParams(
        k=args.k,
        disk_shrink=args.disk_shrink,
        blur_sigma=args.blur_sigma,
        postprocess=PostprocessParams(
            open_radius=args.open_radius,
            close_radius=args.close_radius,
            bridge_gap=args.bridge_gap,
            min_area=args.min_area,
        ),
    )

    paths = resolve_paths(args)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no images found -- has the data been downloaded?")

    out_path = Path(args.out or (ensure_output_dir() / f"submission_{args.subset}.csv"))
    print(f"predicting {len(paths)} images -> {out_path}")

    start = time.perf_counter()
    predictions: dict[str, list] = {}
    workers = args.workers or None
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(params,)
    ) as pool:
        for n, (stem, rles) in enumerate(
            pool.map(_predict, [str(p) for p in paths], chunksize=4), start=1
        ):
            predictions[stem] = rles
            if n % 25 == 0 or n == len(paths):
                print(f"  {n}/{len(paths)}", flush=True)
    elapsed = time.perf_counter() - start

    frame = build_submission(predictions)
    write_submission(frame, out_path)

    n_instances = len(frame)
    empty = sum(1 for v in predictions.values() if not v)
    print(
        f"\n{n_instances} instances over {len(predictions)} images "
        f"({n_instances / max(len(predictions), 1):.2f} per image, {empty} empty)"
    )
    # End-to-end runtime is part of the judging rubric -- keep an eye on it.
    print(f"runtime {elapsed:.1f}s ({elapsed / len(paths):.2f}s per image)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
