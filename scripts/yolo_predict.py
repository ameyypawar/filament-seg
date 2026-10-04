"""Run the trained detector over cached frames and save its instances as JSON.

Detections are kept down to a low confidence (0.05 by default) so that the
fusion sweep, not this script, decides where to cut. Masks are stored at full
resolution as COCO RLE counts, keyed by observation:
``{stem: [{"score": 0.87, "counts": "..."}, ...]}``.

With ``--tta flips`` the detector also sees the image mirrored left-right,
top-bottom and both, and the four sets are merged by
``filament_seg.fusion.merge_views``: one detection per filament, scored by its
average confidence across the four.

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

import cv2
import numpy as np

from filament_seg.config import CACHE_DIR, SPLIT_PATH, TEST_IMAGE_DIR, TRAIN_ANNOTATIONS
from filament_seg.data import load_annotations, load_split, stems_of, test_image_paths
from filament_seg.fusion import Detection, merge_views
from filament_seg.rle import mask_to_counts

#: (flip left-right, flip top-bottom) for each copy the detector sees.
FLIPS = {"none": [(False, False)],
         "flips": [(False, False), (True, False), (False, True), (True, True)]}


def detect(model, image: np.ndarray, flip_lr: bool, flip_ud: bool, args) -> list[Detection]:
    """Detections on one flipped copy of ``image``, with masks flipped back."""
    view = image[:, ::-1] if flip_lr else image
    view = np.ascontiguousarray(view[::-1] if flip_ud else view)
    result = model.predict(view, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                           max_det=args.max_det, retina_masks=True, device=args.device,
                           verbose=False)[0]
    if result.masks is None or not len(result.boxes):
        return []
    masks = result.masks.data.cpu().numpy() > 0.5
    if flip_ud:
        masks = masks[:, ::-1]
    if flip_lr:
        masks = masks[:, :, ::-1]
    scores = result.boxes.conf.cpu().numpy()
    return [Detection(round(float(score), 4), mask_to_counts(mask))
            for mask, score in zip(masks, scores) if mask.any()]


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
    parser.add_argument("--tta", choices=list(FLIPS), default="none")
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
        image = cv2.imread(str(flat_dir / f"{stem}.png"))
        if image is None:
            raise SystemExit(f"no cached flat image for {stem} -- run scripts/preprocess.py")
        views = [detect(model, image, flip_lr, flip_ud, args) for flip_lr, flip_ud in FLIPS[args.tta]]
        found = views[0] if len(views) == 1 else merge_views(views)
        detections[stem] = [{"score": round(d.score, 4), "counts": d.counts} for d in found]
        if n % 25 == 0 or n == len(stems):
            print(f"  {n}/{len(stems)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detections), encoding="utf-8")
    n_found = sum(len(v) for v in detections.values())
    print(f"wrote {out}: {n_found} detections over {len(stems)} observations")


if __name__ == "__main__":
    main()
