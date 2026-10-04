"""Compare two pipelines observation by observation.

Each input is the ``--save-totals`` file of a scripts/fuse_detections.py run
on the same validation observations. The difference in pooled PQ gets a
paired bootstrap over observations: a 95% interval, and the share of
resamples in which the first pipeline beats the second.

``--key crossfit`` (the default) compares on all observations, each scored
with the setting tuned on the other half; ``--key holdout`` compares on the
held-out half only, tuned on the other, as earlier runs were judged.

    python scripts/compare_runs.py outputs/step3/totals_det123.npz outputs/step3/totals_det12.npz
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse

import numpy as np

from filament_seg.scoring import bootstrap_deltas, pq_of


def load(path: str, key: str) -> tuple[list[str], np.ndarray]:
    data = np.load(path)
    stems = data["holdout_stems" if key == "holdout" else "stems"]
    return [str(stem) for stem in stems], data[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", help="--save-totals file of the candidate")
    parser.add_argument("b", help="--save-totals file of the reference")
    parser.add_argument("--key", choices=["crossfit", "holdout", "baseline"], default="crossfit")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    stems_a, a = load(args.a, args.key)
    stems_b, b = load(args.b, args.key)
    if stems_a != stems_b:
        raise SystemExit("the two runs scored different observations; compare like with like")
    deltas = bootstrap_deltas(a, b, n_boot=args.n_boot)
    delta = float(pq_of(a.sum(axis=0)) - pq_of(b.sum(axis=0)))
    low, high = np.percentile(deltas, [2.5, 97.5])
    print(f"{args.key}, {len(stems_a)} observations")
    print(f"  A  {float(pq_of(a.sum(axis=0))):.4f}  {args.a}")
    print(f"  B  {float(pq_of(b.sum(axis=0))):.4f}  {args.b}")
    print(f"  A - B  {delta:+.4f}  95% [{low:+.4f}, {high:+.4f}]  P(A > B) {np.mean(deltas > 0):.2f}")


if __name__ == "__main__":
    main()
