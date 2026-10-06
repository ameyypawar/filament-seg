"""Measurements behind statements in the report that no figure shows.

    .venv/bin/python reports/supporting_analyses.py   # writes reports/numbers_supporting.json

1. How far a thresholded solar disk sits outside the fitted limb, over every
   training and test frame -- the reason for the limb finder in disk.py.
2. Days from each validation and test observation to the nearest training
   observation: is the test set further from the training data in time?
3. The validated entry's cross-fitted PQ reweighted to the test set's mix of
   years and of GONG sites.
4. Whether annotated filaments that a detector found but scored below the
   fusion cut-off are ones that fewer of the other annotators drew, compared
   with the filaments the entry matched.
5. For near-misses (a prediction overlaps an annotated filament with IoU
   0.25-0.5), what U-Net probability the annotated pixels the prediction
   leaves out have: could a lower threshold have recovered them?

Validation scoring reuses make_figures.py, so the error categories are the
same ones Fig. 4 counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parent
REPO = REPORTS.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPORTS))

import datetime as dt
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import make_figures as mf
from filament_seg.config import TEST_IMAGE_DIR, TRAIN_IMAGE_DIR
from filament_seg.data import load_annotations, load_split, stems_of, test_image_paths
from filament_seg.dataset import load_disk_geometry
from filament_seg.disk import _threshold_disk, read_image
from filament_seg.rle import iou_matrix, rle_to_mask
from filament_seg.scoring import pq_of

#: The validated entry's per-observation cross-fitted totals.
TOTALS = REPO / "outputs" / "step3" / "totals_det124L_v5last.npz"
OUT = REPORTS / "numbers_supporting.json"


def quartiles(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"n": int(values.size), "median": round(float(np.median(values)), 3),
            "p25": round(float(np.percentile(values, 25)), 3),
            "p75": round(float(np.percentile(values, 75)), 3),
            "max": round(float(values.max()), 3)}


# --- 1. Thresholded disk versus fitted limb --------------------------------------------


def _limb_gap(item: tuple[str, float, float, float]) -> tuple[float, float]:
    """Radius excess of the thresholded disk, and the largest radial gap between the circles."""
    path, cx, cy, radius = item
    disk = _threshold_disk(read_image(path))
    centre_shift = float(np.hypot(disk.cx - cx, disk.cy - cy))
    return disk.radius - radius, centre_shift + abs(disk.radius - radius)


def limb_offsets(annotations) -> dict:
    fitted = load_disk_geometry(mf.CACHE_DIR / "disk.json")
    paths = {r.stem: TRAIN_IMAGE_DIR / r.file_name for r in annotations.images.values()}
    paths.update({p.stem: p for p in test_image_paths(TEST_IMAGE_DIR)})
    items = [(str(p), fitted[s].cx, fitted[s].cy, fitted[s].radius)
             for s, p in sorted(paths.items()) if s in fitted]
    with ProcessPoolExecutor() as pool:
        gaps = np.array(list(pool.map(_limb_gap, items, chunksize=8)))
    return {"frames": len(items),
            "threshold_radius_minus_limb_px": quartiles(gaps[:, 0]),
            "largest_radial_gap_px": quartiles(gaps[:, 1])}


# --- 2. Distance in time to the training data ------------------------------------------


def days_to_training(annotations, split: dict, test_stems: list[str]) -> dict:
    def seconds(stem: str) -> float:
        return dt.datetime.strptime(stem[:14], "%Y%m%d%H%M%S").timestamp()

    train = np.array(sorted(seconds(s) for s in stems_of(annotations, split["train"])))

    def nearest(stems) -> list[float]:
        out = []
        for stem in stems:
            t = seconds(stem)
            i = int(np.searchsorted(train, t))
            out.append(min(abs(train[j] - t) for j in (i - 1, i) if 0 <= j < train.size) / 86400)
        return out

    return {"validation": quartiles(nearest(stems_of(annotations, split["val"]))),
            "test": quartiles(nearest(test_stems))}


# --- 3. Reweighting validation to the test set's mix ------------------------------------


def reweighted_pq(test_stems: list[str]) -> dict:
    saved = np.load(TOTALS)
    stems = [str(s) for s in saved["stems"]]
    totals = saved["crossfit"]
    out = {"cross_fitted": round(float(pq_of(totals.sum(axis=0))), 4)}
    for name, key in (("year", lambda s: s[:4]), ("site", lambda s: s[14:16])):
        rows = defaultdict(list)
        for i, stem in enumerate(stems):
            rows[key(stem)].append(i)
        wanted = Counter(key(s) for s in test_stems)
        weights = np.zeros(len(stems))
        for group, idx in rows.items():
            weights[idx] = wanted.get(group, 0) / len(idx)
        out[f"reweighted_to_test_{name}s"] = round(float(pq_of((weights[:, None] * totals).sum(axis=0))), 4)
        out[f"test_observations_in_{name}s_without_validation"] = sum(
            n for group, n in wanted.items() if group not in rows)
    return out


# --- 4. Do other annotators draw the filaments scored below the cut-off? ------------------


def other_annotators(per_stem: list[dict], gt: dict) -> dict:
    """Share of the other annotators of an observation who drew the same filament
    (a polygon with IoU > 0.5 in their own view), by what happened to it."""
    shares = defaultdict(list)
    for entry in per_stem:
        views = entry["views"]
        causes = iter(cause for cause, _ in entry["fn"])
        for view in views:
            outcome = entry["outcome"][view]
            fn_cause = {a: next(causes) for a in outcome["fn"]}
            if len(views) < 2:
                continue
            matched = {g for g, _ in outcome["tp"]}
            for a, polygon in enumerate(gt[view]):
                label = "matched" if a in matched else fn_cause[a]
                drawn = [bool((iou_matrix([polygon], gt[o]) > 0.5).any()) if gt[o] else False
                         for o in views if o != view]
                shares[label].append(float(np.mean(drawn)))
    return {label: {"filament_views": len(v), "mean_share_of_other_annotators": round(float(np.mean(v)), 3)}
            for label, v in sorted(shares.items())}


# --- 5. Where the missing pixels of a near-miss are ----------------------------------------


def _near_miss_pixels(stem: str) -> list[float]:
    """For each near-missed annotated filament: the share of the annotated pixels
    outside the prediction that the U-Net scores below 0.3."""
    views, gt = mf._WORKER["views"][stem], mf._WORKER["gt"]
    preds, _, _ = mf.fuse_observation(stem, mf._WORKER["detections"][stem], mf._WORKER["disks"][stem])
    if not preds:
        return []
    probability = 1.0 / (1.0 + np.exp(-np.load(mf.LOGIT_DIR / f"{stem}.npy").astype(np.float32)))
    shares = []
    for view in views:
        if not gt[view]:
            continue
        ious = iou_matrix(gt[view], preds)
        for a, polygon in enumerate(gt[view]):
            b = int(ious[a].argmax())
            if not 0.25 < ious[a, b] <= 0.5:
                continue
            missing = rle_to_mask(polygon).astype(bool) & ~rle_to_mask(preds[b]).astype(bool)
            if missing.any():
                shares.append(float((probability[missing] < 0.3).mean()))
    return shares


def near_miss_pixels(inputs) -> dict:
    with ProcessPoolExecutor(initializer=mf._init_worker,
                             initargs=(inputs.views, inputs.gt, inputs.detections, inputs.disks)) as pool:
        shares = [s for stem in pool.map(_near_miss_pixels, inputs.stems) for s in stem]
    return {"near_missed_filament_views": len(shares),
            "mean_share_of_missing_pixels_below_0.3": round(float(np.mean(shares)), 3)}


def main() -> None:
    annotations = load_annotations()
    split = load_split(mf.SPLIT)
    test_stems = sorted(p.stem for p in test_image_paths(TEST_IMAGE_DIR))
    inputs = mf.load_inputs()
    per_stem = mf.analyse_validation(inputs, workers=0 or None)
    numbers = {
        "limb": limb_offsets(annotations),
        "days_to_nearest_training_observation": days_to_training(annotations, split, test_stems),
        "validated_entry_pq": reweighted_pq(test_stems),
        "drawn_by_other_annotators": other_annotators(per_stem, inputs.gt),
        "near_miss_pixels": near_miss_pixels(inputs),
    }
    OUT.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
