"""Measure inter-annotator Panoptic Quality -- the practical ceiling.

Where two people annotated the same observation, scoring one against the other
gives the PQ a *perfect* model would achieve against a single reference. If that
number is around 0.55, the top of the leaderboard is at human agreement and
further pixel-level effort buys nothing; the remaining points are in the 30%
qualitative half of the rubric.

    python scripts/annotator_agreement.py
"""

from __future__ import annotations

import argparse
import json
import statistics

from filament_seg.config import PQ_IOU_THRESHOLD, TRAIN_ANNOTATIONS, ensure_output_dir
from filament_seg.data import load_annotations
from filament_seg.metrics import evaluate_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--iou-threshold", type=float, default=PQ_IOU_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    grouped = {k: v for k, v in annotations.by_file_stem().items() if len(v) >= 2}
    stems = sorted(grouped)
    if args.limit:
        stems = stems[: args.limit]

    if not stems:
        print("no observation has more than one annotator; nothing to compare")
        return

    scores: list[float] = []
    per_stem: dict[str, float] = {}
    for n, stem in enumerate(stems, start=1):
        ids = grouped[stem]
        pairwise: list[float] = []
        for i, reference in enumerate(ids):
            for j, other in enumerate(ids):
                if i == j:
                    continue
                result = evaluate_image(
                    stem,
                    annotations.gt_rles(reference),
                    annotations.gt_rles(other),
                    iou_threshold=args.iou_threshold,
                    with_fragmentation=False,
                )
                pairwise.append(result.pq)
        if pairwise:
            per_stem[stem] = statistics.fmean(pairwise)
            scores.append(per_stem[stem])
        if n % 25 == 0 or n == len(stems):
            print(f"  {n}/{len(stems)}", flush=True)

    print(f"\nobservations with >=2 annotators : {len(scores)}")
    print(f"mean inter-annotator PQ          : {statistics.fmean(scores):.4f}")
    print(f"median                           : {statistics.median(scores):.4f}")
    print(f"min / max                        : {min(scores):.4f} / {max(scores):.4f}")

    out = ensure_output_dir() / "annotator_agreement.json"
    out.write_text(
        json.dumps(
            {
                "n_observations": len(scores),
                "mean_pq": statistics.fmean(scores),
                "median_pq": statistics.median(scores),
                "per_observation": per_stem,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
