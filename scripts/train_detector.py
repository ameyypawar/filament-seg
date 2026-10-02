"""Train the YOLO instance-segmentation model that groups filament pixels.

Its masks are too coarse to submit directly (an eighth of full resolution at
the default 1024 input); what fusion needs from it is which pixels belong to
one filament, and how confident it is that the filament exists. See
``filament_seg.fusion``.

Augmentation is limited to flips and mild brightness and scale jitter. Mosaic,
the YOLO default, is off: it would tile four Suns into one image, and a
filament's appearance depends on its place on the disk.

    python scripts/train_detector.py --data /tmp/yolo/data.yaml --out /tmp/detector
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="data.yaml from scripts/export_yolo.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=1.0, help="debug: train on a fraction")
    args = parser.parse_args()

    from ultralytics import YOLO

    out = Path(args.out).resolve()
    model = YOLO(args.model)
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, workers=args.workers, patience=args.patience,
        project=str(out), name="train", exist_ok=True, seed=0, deterministic=False,
        single_cls=True, cos_lr=True, plots=False, fraction=args.fraction,
        mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
        fliplr=0.5, flipud=0.5, degrees=0.0, shear=0.0, perspective=0.0,
        translate=0.05, scale=0.2, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        overlap_mask=True, mask_ratio=4,
    )
    run_dir = out / "train"
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training finished without {best}")
    shutil.copy2(best, out / "detector_best.pt")
    if (run_dir / "results.csv").exists():
        shutil.copy2(run_dir / "results.csv", out / "detector_results.csv")
    print(f"wrote {out / 'detector_best.pt'}")


if __name__ == "__main__":
    main()
