"""Inspect the downloaded dataset before writing any model code.

Answers the questions that shape every later decision: how many observations
are there really (as opposed to annotated views), how many annotators per
observation, how many filaments per image, how big are they in pixels, and how
the observations are spread over time and observatories.

The filament size distribution matters most: Panoptic Quality needs IoU > 0.5
per instance, and small thin objects almost never clear that bar. The fraction
of ground-truth filaments below a few hundred pixels is effectively an upper
bound on the score nobody can beat.

    python scripts/audit_data.py
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from filament_seg.config import (
    TEST_IMAGE_DIR,
    TRAIN_ANNOTATIONS,
    TRAIN_IMAGE_DIR,
    ensure_output_dir,
)
from filament_seg.data import load_annotations, test_image_paths
from filament_seg.rle import rle_areas


def percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    keys = [1, 5, 25, 50, 75, 95, 99]
    return {f"p{k:02d}": float(v) for k, v in zip(keys, np.percentile(values, keys))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="limit how many images are decoded for the area statistics (0 = all)",
    )
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    by_stem = annotations.by_file_stem()

    print(f"annotated views (image ids) : {len(annotations.images)}")
    print(f"distinct observations       : {len(by_stem)}")

    per_obs = collections.Counter(len(v) for v in by_stem.values())
    print("annotators per observation  : "
          + ", ".join(f"{k}x{v}" for k, v in sorted(per_obs.items())))

    counts = np.array(
        [len(annotations.annotations_by_image.get(i, [])) for i in annotations.image_ids]
    )
    print(
        f"filaments per annotated view: total={int(counts.sum())} "
        f"mean={counts.mean():.2f} median={np.median(counts):.0f} "
        f"min={counts.min()} max={counts.max()} empty={(counts == 0).sum()}"
    )

    dates = sorted({r.date for r in annotations.images.values()})
    sites = collections.Counter(r.observatory for r in annotations.images.values())
    print(f"date range                  : {dates[0]} .. {dates[-1]}  ({len(dates)} days)")
    print(f"observatories               : {dict(sites.most_common())}")

    n_train_images = len(test_image_paths(TRAIN_IMAGE_DIR))
    n_test_images = len(test_image_paths(TEST_IMAGE_DIR))
    print(f"image files on disk         : train={n_train_images}  test={n_test_images}")

    # --- Filament size distribution (the real ceiling on PQ) ---------------
    image_ids = annotations.image_ids
    if args.max_images:
        image_ids = image_ids[: args.max_images]

    areas: list[float] = []
    for image_id in image_ids:
        areas.extend(rle_areas(annotations.gt_rles(image_id)).tolist())
    areas_arr = np.asarray(areas, dtype=np.float64)

    print(f"\nfilament areas (px, over {len(image_ids)} views, n={areas_arr.size})")
    for key, value in percentiles(areas_arr).items():
        print(f"  {key}: {value:10.0f}")
    for cutoff in (100, 250, 500, 1000, 2000):
        share = float((areas_arr < cutoff).mean()) if areas_arr.size else 0.0
        print(f"  share below {cutoff:>5} px : {share:6.1%}")

    report = {
        "n_image_ids": len(annotations.images),
        "n_observations": len(by_stem),
        "annotators_per_observation": dict(per_obs),
        "filaments_total": int(counts.sum()),
        "filaments_per_view_mean": float(counts.mean()),
        "date_range": [dates[0], dates[-1]],
        "observatories": dict(sites),
        "n_train_image_files": n_train_images,
        "n_test_image_files": n_test_images,
        "area_percentiles": percentiles(areas_arr),
    }
    out = ensure_output_dir() / "data_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
