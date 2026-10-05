"""How closely two test submissions agree, scoring one against the other as if it were ground truth.

There are no test labels, so this is the sanity check for a pipeline that
cannot be validated directly -- the final models trained on every labelled
observation. Their predictions should agree with the validated pipeline's
about as well as two validated pipelines that differ only by a random seed
agree with each other. Much less agreement means something went wrong (the
wrong checkpoint, data or settings), not that the new models are better.

    python scripts/agreement.py outputs/submission_all_data.csv outputs/submission_validated.csv
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse

import numpy as np

from filament_seg.metrics import evaluate_image, summarize
from filament_seg.rle import read_submission, rle_areas


def describe(predictions: dict) -> str:
    areas = np.concatenate([rle_areas(rles) for rles in predictions.values() if rles] or [[0]])
    return (f"{sum(len(r) for r in predictions.values())} instances over {len(predictions)} images, "
            f"median area {int(np.median(areas))} px")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate")
    parser.add_argument("reference", help="scored as if it were the ground truth")
    args = parser.parse_args()

    candidate, reference = read_submission(args.candidate), read_submission(args.reference)
    results = [evaluate_image(stem, reference.get(stem, []), candidate.get(stem, []),
                              with_fragmentation=False)
               for stem in sorted(set(candidate) | set(reference))]
    summary = summarize(results)
    print(f"candidate: {describe(candidate)}")
    print(f"reference: {describe(reference)}")
    print(f"agreement PQ {summary['pq_pooled']:.4f} (SQ {summary['sq']:.3f}, RQ {summary['rq']:.3f})")


if __name__ == "__main__":
    main()
