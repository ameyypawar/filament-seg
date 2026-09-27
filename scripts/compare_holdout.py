"""Compare post-processing choices on validation observations the sweep never saw.

sweep_postprocess.py picks its settings on a random subset of the validation
observations. Scoring the winner on those same observations overstates it: with
144 combinations to choose from, the best one is partly just the luckiest one.
This reproduces exactly which observations the sweep used and scores each
submission on the complement, which is the honest comparison. It also reports
the sweep's own subset and the whole split, so the size of that optimism is
visible rather than assumed.

    python scripts/compare_holdout.py --sweep-images 72 \\
        --submission outputs/val_default.csv --submission outputs/val_tuned.csv
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from pathlib import Path

from filament_seg.config import SPLIT_PATH, TRAIN_ANNOTATIONS, ensure_output_dir
from filament_seg.data import deduplicate_by_file, load_annotations
from filament_seg.metrics import evaluate
from filament_seg.rle import read_submission


def sweep_subset(all_ids: list[str], n_images: int, seed: int) -> set[str]:
    """The validation views sweep_postprocess.py scored, reproduced exactly.

    Mirrors its selection: shuffle the deduplicated validation ids with
    random.Random(seed) and take the first n_images.
    """
    shuffled = list(all_ids)
    random.Random(seed).shuffle(shuffled)
    return set(shuffled[:n_images])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", action="append", required=True,
                        help="validation-set submission CSV; repeat to compare several")
    parser.add_argument("--sweep-images", type=int, required=True,
                        help="the --n-images the sweep was run with")
    parser.add_argument("--seed", type=int, default=0, help="the sweep's --seed")
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    all_ids = deduplicate_by_file(annotations, split["val"])
    used = sweep_subset(all_ids, args.sweep_images, args.seed)
    groups = {
        "held out from the sweep": [i for i in all_ids if i not in used],
        "used by the sweep": [i for i in all_ids if i in used],
        "all validation": all_ids,
    }
    print(f"validation observations: {len(all_ids)} "
          f"({len(groups['used by the sweep'])} used by the sweep, "
          f"{len(groups['held out from the sweep'])} held out)\n")

    results: dict[str, dict] = {}
    for path in args.submission:
        by_stem = read_submission(path)
        results[path] = {}
        for group, ids in groups.items():
            gt = annotations.gt_dict(ids)
            pred = {i: by_stem.get(annotations.images[i].stem, []) for i in ids}
            summary, _ = evaluate(gt, pred)
            results[path][group] = {
                key: summary[key]
                for key in ("pq_pooled", "pq_per_image_mean", "sq", "rq", "tp", "fp", "fn",
                            "one_to_many", "many_to_one", "n_images")
            }

    header = f"{'submission':<34} {'group':<25} {'PQ pooled':>9} {'per-image':>9} {'SQ':>6} {'RQ':>6} {'split up':>8}"
    print(header)
    print("-" * len(header))
    for path, by_group in results.items():
        for group, r in by_group.items():
            print(f"{Path(path).name:<34} {group:<25} {r['pq_pooled']:>9.4f} "
                  f"{r['pq_per_image_mean']:>9.4f} {r['sq']:>6.3f} {r['rq']:>6.3f} "
                  f"{r['one_to_many']:>8}")

    out = Path(args.out or (ensure_output_dir() / "holdout_comparison.json"))
    out.write_text(json.dumps({"sweep_images": args.sweep_images, "seed": args.seed,
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
