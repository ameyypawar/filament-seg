"""Score a submission-format CSV against MAGFiLO ground truth, locally.

Reports everything the judging rubric asks for: Panoptic Quality under both
plausible aggregations, the SQ/RQ decomposition, the IoU and Dice distributions
over matched pairs, and the one-to-many / many-to-one fragmentation counts.

    python scripts/evaluate.py --submission outputs/val_baseline.csv --subset val
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

from filament_seg.config import (
    PQ_IOU_THRESHOLD,
    SPLIT_PATH,
    TRAIN_ANNOTATIONS,
    ensure_output_dir,
)
from filament_seg.data import deduplicate_by_file, load_annotations
from filament_seg.metrics import evaluate, format_report
from filament_seg.rle import read_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--subset", choices=["train", "val"], default="val")
    parser.add_argument(
        "--all-annotators",
        action="store_true",
        help="score against every annotator instead of one per observation "
        "(the default keeps heavily-annotated images from dominating)",
    )
    parser.add_argument("--iou-threshold", type=float, default=PQ_IOU_THRESHOLD)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    image_ids = split[args.subset]
    if not args.all_annotators:
        image_ids = deduplicate_by_file(annotations, image_ids)

    predictions_by_stem = read_submission(args.submission)

    # Ground truth is keyed by annotated view; predictions are keyed by
    # observation. Fan the predictions out so both dicts share keys.
    gt = annotations.gt_dict(image_ids)
    pred = {
        image_id: predictions_by_stem.get(annotations.images[image_id].stem, [])
        for image_id in image_ids
    }

    summary, _ = evaluate(gt, pred, iou_threshold=args.iou_threshold)
    print(f"subset={args.subset}  views={len(image_ids)}  "
          f"annotators={'all' if args.all_annotators else 'one per observation'}")
    print(format_report(summary))

    out = Path(args.out or (ensure_output_dir() / f"eval_{args.subset}.json"))
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
