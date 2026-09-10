"""Cache the expensive, parameter-independent preprocessing once.

Disk detection and limb-darkening flattening (``filament_seg.disk``) are the
same no matter what crop size, encoder or training run comes later -- they
depend only on the raw pixels. Paying for them again at the start of every
experiment would be wasted wall-clock, so this script runs them once per
observation and writes:

* the flattened image, quantised to uint8 as ``clip(flat, 0, 2.0) * 127.5`` so
  a network loader can recover it with a plain ``png / 255.0`` (channel 0 of
  the model input is exactly that value, in [0, 1])
* the disk geometry (cx, cy, radius) for every stem, so downstream code can
  rebuild the radius map without re-running disk detection
* for every annotated view, the union of its ground-truth filament polygons
  rendered to a binary mask, so training never touches COCO polygons

    python scripts/preprocess.py
    python scripts/preprocess.py --limit 40           # smoke test
    python scripts/preprocess.py --force               # rebuild everything
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from filament_seg.config import REPO_ROOT, TEST_IMAGE_DIR, TRAIN_ANNOTATIONS, TRAIN_IMAGE_DIR
from filament_seg.data import Annotations, load_annotations, test_image_paths
from filament_seg.disk import detect_disk, flatten, read_image
from filament_seg.rle import rle_to_mask

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"

#: `clip(flat, 0, FLAT_CLIP) * 255 / FLAT_CLIP` packs [0, FLAT_CLIP] into a
#: uint8 byte. Quiet Sun sits at 1.0 and filaments dip to roughly 0.5-0.8;
#: 2.0 leaves headroom for the rare bright plage pixel without wasting most
#: of the byte range on values that almost never occur.
FLAT_CLIP = 2.0

_ANNOTATIONS: Annotations | None = None
_FORCE = False


def _init(annotations: Annotations, force: bool) -> None:
    global _ANNOTATIONS, _FORCE
    _ANNOTATIONS = annotations
    _FORCE = force


def _process_observation(
    payload: tuple[str, str, list[str], Path, Path]
) -> tuple[str, dict | None, int]:
    """Flatten one observation (if not already cached) and render its masks.

    Returns ``(stem, geometry, n_masks_written)``. ``geometry`` is ``None``
    when the flat cache already existed and nothing needed recomputing -- the
    caller then keeps whatever geometry is already on record for that stem.
    """
    stem, image_path, image_ids, flat_dir, mask_dir = payload
    flat_path = flat_dir / f"{stem}.png"

    geometry = None
    if _FORCE or not flat_path.exists():
        image = read_image(image_path)
        disk = detect_disk(image)
        flat = flatten(image, disk)
        png = (np.clip(flat, 0.0, FLAT_CLIP) * (255.0 / FLAT_CLIP)).astype(np.uint8)
        cv2.imwrite(str(flat_path), png)
        geometry = {"cx": disk.cx, "cy": disk.cy, "radius": disk.radius}

    n_masks = 0
    for image_id in image_ids:
        mask_path = mask_dir / f"{image_id}.png"
        if not _FORCE and mask_path.exists():
            continue
        record = _ANNOTATIONS.images[image_id]
        mask = np.zeros((record.height, record.width), dtype=np.uint8)
        for rle in _ANNOTATIONS.gt_rles(image_id):
            mask |= rle_to_mask(rle).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask * 255)
        n_masks += 1

    return stem, geometry, n_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", default=str(TRAIN_IMAGE_DIR))
    parser.add_argument("--test-images", default=str(TEST_IMAGE_DIR))
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--limit", type=int, default=0, help="debug: first N observations")
    parser.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    parser.add_argument("--force", action="store_true", help="reprocess even if cached")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    flat_dir = cache_dir / "flat"
    mask_dir = cache_dir / "mask"
    disk_path = cache_dir / "disk.json"
    flat_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(args.annotations)
    views_by_stem = annotations.by_file_stem()

    train_paths = test_image_paths(args.train_images)
    test_paths = test_image_paths(args.test_images)
    payloads = [
        (p.stem, str(p), views_by_stem.get(p.stem, []), flat_dir, mask_dir) for p in train_paths
    ] + [(p.stem, str(p), [], flat_dir, mask_dir) for p in test_paths]

    if args.limit:
        payloads = payloads[: args.limit]
    if not payloads:
        raise SystemExit("no images found -- has the data been downloaded?")
    print(f"{len(payloads)} observations ({len(train_paths)} train, {len(test_paths)} test)")

    disk_geometry: dict[str, dict] = {}
    if disk_path.exists():
        disk_geometry = json.loads(disk_path.read_text(encoding="utf-8"))

    n_masks_total = 0
    with ProcessPoolExecutor(
        max_workers=args.workers or None, initializer=_init, initargs=(annotations, args.force)
    ) as pool:
        for n, (stem, geometry, n_masks) in enumerate(
            pool.map(_process_observation, payloads, chunksize=4), start=1
        ):
            if geometry is not None:
                disk_geometry[stem] = geometry
            n_masks_total += n_masks
            if n % 25 == 0 or n == len(payloads):
                print(f"  {n}/{len(payloads)}", flush=True)

    disk_path.write_text(json.dumps(disk_geometry, indent=2), encoding="utf-8")
    print(f"wrote {len(disk_geometry)} disk geometries -> {disk_path}")
    print(f"wrote {n_masks_total} new mask PNGs -> {mask_dir}")


if __name__ == "__main__":
    main()
