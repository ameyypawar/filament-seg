"""Combine several detectors' outputs into one detections file.

Each detector's instances for an observation are one "view" for
``filament_seg.fusion.merge_views``: instances that overlap with IoU above
``--iou`` are the same filament, keep the most confident member's mask, and
are scored by the average across detectors, counting a detector that missed
the filament as 0. Two detectors combined this way lifted held-out PQ from
0.413 (the better one alone) to 0.419.

    python scripts/merge_detections.py --out detections_val_ens.json det1/detections_val.json det2/detections_val.json
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

from filament_seg.fusion import Detection, merge_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="detections JSON files from scripts/yolo_predict.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    detectors = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    stems = sorted(set().union(*detectors))
    merged = {}
    for stem in stems:
        views = [[Detection(d["score"], d["counts"]) for d in found.get(stem, [])]
                 for found in detectors]
        merged[stem] = [{"score": round(d.score, 4), "counts": d.counts}
                        for d in merge_views(views, args.iou)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged), encoding="utf-8")
    print(f"wrote {out}: {sum(len(v) for v in merged.values())} detections over {len(stems)} "
          f"observations from {len(detectors)} detectors")


if __name__ == "__main__":
    main()
