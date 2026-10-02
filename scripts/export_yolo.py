"""Write a YOLO instance-segmentation dataset from the preprocessing cache.

Images are the cached limb-darkening-flattened frames -- exactly what the U-Net
sees -- hard-linked rather than copied, so the dataset costs no extra disk.
Every annotator's view of an observation becomes its own sample with its own
labels: the competition scores each view separately, so a detector trained on
all of them learns the spread of what annotators call one filament rather than
one person's habits.

    python scripts/export_yolo.py --out /tmp/yolo
    python scripts/export_yolo.py --out /tmp/yolo_smoke --limit 8   # smoke test
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from filament_seg.config import CACHE_DIR, SPLIT_PATH, TRAIN_ANNOTATIONS
from filament_seg.data import Annotations, load_annotations, load_split


def polygon_area(polygon: list[float]) -> float:
    xs, ys = np.asarray(polygon[0::2]), np.asarray(polygon[1::2])
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))


def label_lines(annotations: Annotations, image_id: str) -> list[str]:
    """One ``0 x1 y1 x2 y2 ...`` line per filament, coordinates normalised to [0, 1].

    MAGFiLO stores one polygon per filament; the rare multi-part one keeps its
    largest part, since a YOLO segmentation label is a single polygon.
    """
    record = annotations.images[image_id]
    lines = []
    for annotation in annotations.annotations_by_image.get(image_id, []):
        polygons = [p for p in annotation.get("segmentation") or [] if len(p) >= 6]
        if not polygons:
            continue
        polygon = max(polygons, key=polygon_area)
        xs = np.clip(np.asarray(polygon[0::2], dtype=np.float64) / record.width, 0.0, 1.0)
        ys = np.clip(np.asarray(polygon[1::2], dtype=np.float64) / record.height, 0.0, 1.0)
        lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys)))
    return lines


def link(source: Path, target: Path) -> None:
    """Hard-link ``source`` to ``target`` (copy across filesystems)."""
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--limit", type=int, default=0, help="debug: first N views per subset")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    split = load_split(args.split)
    flat_dir = Path(args.cache_dir) / "flat"
    out = Path(args.out).resolve()

    for subset in ("train", "val"):
        image_dir, label_dir = out / "images" / subset, out / "labels" / subset
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        views = split[subset][: args.limit] if args.limit else split[subset]
        n_labels = 0
        for image_id in views:
            source = flat_dir / f"{annotations.images[image_id].stem}.png"
            if not source.exists():
                raise SystemExit(f"missing cached flat image {source} -- run scripts/preprocess.py")
            link(source, image_dir / f"{image_id}.png")
            lines = label_lines(annotations, image_id)
            (label_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_labels += len(lines)
        print(f"{subset}: {len(views)} views, {n_labels} filaments")

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
        encoding="utf-8",
    )
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
