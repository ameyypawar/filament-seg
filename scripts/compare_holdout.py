"""Compare validation submissions on observations the post-processing sweep never saw.

sweep_postprocess.py picks its settings on a random subset of the validation
observations. Scoring the winner on those same observations overstates it:
with dozens of combinations to choose from, the best one is partly just the
luckiest one. This scores each submission on the complement, which is the
honest comparison, and also on the sweep's own subset and the whole split, so
the size of that optimism is visible rather than assumed.

The split is read from the sweep's ``<out>_subset.json`` rather than
re-derived, so it is exactly the one the sweep scored. Scoring mirrors the
organisers' self-evaluation notebook: each observation's predictions are
matched against every annotator's view of it, and TP/FP/FN are pooled over
all views.

    python scripts/compare_holdout.py --sweep-subset outputs/sweep_postprocess_subset.json \\
        --submission outputs/val_default.csv --submission outputs/val_tuned.csv
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

from filament_seg.config import OUTPUT_ROOT, SPLIT_PATH, TRAIN_ANNOTATIONS, ensure_output_dir
from filament_seg.data import load_annotations, load_split, records_for_stems, stems_of
from filament_seg.metrics import evaluate
from filament_seg.rle import read_submission

SUMMARY_KEYS = ("pq_pooled", "pq_per_image_mean", "sq", "rq", "tp", "fp", "fn",
                "one_to_many", "many_to_one", "n_images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", action="append", required=True,
                        help="validation-set submission CSV; repeat to compare several")
    parser.add_argument("--sweep-subset",
                        default=str(OUTPUT_ROOT / "sweep_postprocess_subset.json"),
                        help="the <out>_subset.json written by sweep_postprocess.py")
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    val_ids = load_split(args.split)["val"]
    subset = json.loads(Path(args.sweep_subset).read_text(encoding="utf-8"))
    tuned = set(subset["tune_stems"])
    all_stems = stems_of(annotations, val_ids)
    groups = {
        "held out from the sweep": [s for s in all_stems if s not in tuned],
        "used by the sweep": [s for s in all_stems if s in tuned],
        "all validation": all_stems,
    }
    print(f"validation observations: {len(all_stems)} "
          f"({len(groups['used by the sweep'])} used by the sweep, "
          f"{len(groups['held out from the sweep'])} held out)\n")

    results: dict[str, dict] = {}
    for path in args.submission:
        by_stem = read_submission(path)
        results[path] = {}
        for group, stems in groups.items():
            if not stems:
                continue
            views = records_for_stems(annotations, val_ids, stems)
            gt = annotations.gt_dict(views)
            pred = {i: by_stem.get(annotations.images[i].stem, []) for i in views}
            summary, _ = evaluate(gt, pred)
            results[path][group] = {key: summary[key] for key in SUMMARY_KEYS}

    header = (f"{'submission':<34} {'group':<25} {'PQ pooled':>9} {'per-view':>9} "
              f"{'SQ':>6} {'RQ':>6} {'split up':>8}")
    print(header)
    print("-" * len(header))
    for path, by_group in results.items():
        for group, r in by_group.items():
            print(f"{Path(path).name:<34} {group:<25} {r['pq_pooled']:>9.4f} "
                  f"{r['pq_per_image_mean']:>9.4f} {r['sq']:>6.3f} {r['rq']:>6.3f} "
                  f"{r['one_to_many']:>8}")

    out = Path(args.out or (ensure_output_dir() / "holdout_comparison.json"))
    out.write_text(json.dumps({"sweep_subset": args.sweep_subset, "results": results},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
