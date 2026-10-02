"""Run the trained detector over cached frames and save its instances as JSON.

Detections are kept down to a low confidence (0.05 by default) so that the
fusion sweep, not this script, decides where to cut. Masks are stored at full
resolution as COCO RLE counts, keyed by observation:
``{stem: [{"score": 0.87, "counts": "..."}, ...]}``.

    python scripts/yolo_predict.py --weights detector_best.pt --subset val --out detections_val.json
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

from filament_seg.config import CACHE_DIR, SPLIT_PATH, TEST_IMAGE_DIR, TRAIN_ANNOTATIONS
from filament_seg.data import load_annotations, load_split, stems_of, test_image_paths
from filament_seg.rle import mask_to_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--subset", choices=["val", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    parser.add_argument("--max-det", type=int, default=150)
    parser.add_argument("--device", default=None)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--limit", type=int, default=0, help="debug: first N observations")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.subset == "test":
        stems = sorted(p.stem for p in test_image_paths(TEST_IMAGE_DIR))
    else:
        stems = stems_of(load_annotations(args.annotations), load_split(args.split)["val"])
    if args.limit:
        stems = stems[: args.limit]
    flat_dir = Path(args.cache_dir) / "flat"

    model = YOLO(args.weights)
    detections: dict[str, list[dict]] = {}
    for n, stem in enumerate(stems, start=1):
        result = model.predict(
            str(flat_dir / f"{stem}.png"), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=args.max_det, retina_masks=True, device=args.device, verbose=False,
        )[0]
        found = []
        if result.masks is not None and len(result.boxes):
            masks = result.masks.data.cpu().numpy() > 0.5
            scores = result.boxes.conf.cpu().numpy()
            for mask, score in zip(masks, scores):
                if mask.any():
                    found.append({"score": round(float(score), 4), "counts": mask_to_counts(mask)})
        detections[stem] = found
        if n % 25 == 0 or n == len(stems):
            print(f"  {n}/{len(stems)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detections), encoding="utf-8")
    n_found = sum(len(v) for v in detections.values())
    print(f"wrote {out}: {n_found} detections over {len(stems)} observations")


if __name__ == "__main__":
    main()
