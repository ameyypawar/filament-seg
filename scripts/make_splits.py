"""Build a leak-free train/validation split and save it to outputs/splits.json.

Grouping matters: the same observation is annotated by several people, and
filaments persist for days, so a random split trains and validates on nearly the
same pixels. Default grouping is by date; use --group-by month for a stricter,
more pessimistic estimate.

    python scripts/make_splits.py --group-by date --val-fraction 0.2
"""

from __future__ import annotations

import argparse
import json

from filament_seg.config import SPLIT_PATH, TRAIN_ANNOTATIONS, ensure_output_dir
from filament_seg.data import load_annotations, make_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--group-by", choices=["file", "date", "month"], default="date")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(SPLIT_PATH))
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    split = make_split(
        annotations,
        val_fraction=args.val_fraction,
        group_by=args.group_by,
        seed=args.seed,
    )

    ensure_output_dir()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    train_stems = {annotations.images[i].stem for i in split["train"]}
    val_stems = {annotations.images[i].stem for i in split["val"]}
    overlap = train_stems & val_stems

    print(f"group_by={split['group_by']}  groups={split['n_groups']}")
    print(f"train: {len(split['train'])} views / {len(train_stems)} observations")
    print(f"val  : {len(split['val'])} views / {len(val_stems)} observations")
    print(f"observation overlap between train and val: {len(overlap)} (must be 0)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
