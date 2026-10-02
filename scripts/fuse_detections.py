"""Tune detector-guided fusion on half the validation set and judge it on the other half.

U-Net logits come from the same cache scripts/sweep_postprocess.py uses
(``filament_seg.logit_cache``), detections from scripts/yolo_predict.py.
Scoring mirrors the organisers' notebook: every annotator's view, pooled. The
tuning and held-out halves are the sweep's (``sample_stems``, seed 0), so the
numbers line up with earlier runs.

The held-out half also scores the U-Net-only pipeline (``--baseline``) on the
same logits, and every fusion candidate is reported as a paired-bootstrap
difference from it. That comparison, not the tuning score, decides whether
fusion is worth submitting.

With ``--submission``, the best setting on the tuning half is applied to the
test observations and written out as a submission CSV.

    python scripts/fuse_detections.py --checkpoint outputs/kaggle_v5/model_best.pt \\
        --cache-dir data/cache_v2 --logit-dir outputs/step3/logits_v5_dihedral \\
        --detections-val outputs/kaggle_det1/detections_val.json \\
        --baseline threshold=0.6,min_area=400,bridge_gap=24,close_radius=3,open_radius=0
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from filament_seg.config import CACHE_DIR, SPLIT_PATH, TEST_IMAGE_DIR, TRAIN_ANNOTATIONS
from filament_seg.data import (
    load_annotations,
    load_split,
    records_for_stems,
    sample_stems,
    stems_of,
    test_image_paths,
)
from filament_seg.dataset import load_disk_geometry
from filament_seg.disk import Disk
from filament_seg.fusion import Detection, FusionParams, assign, fuse, grow_regions, unet_binary
from filament_seg.logit_cache import cache_path, ensure_logits
from filament_seg.model import TTA_MODES, select_device
from filament_seg.postprocess import PostprocessParams, logits_to_instances, parse_settings
from filament_seg.rle import Rle, build_submission, labels_to_rles, write_submission
from filament_seg.scoring import COUNT_FIELDS, paired_bootstrap, pq_of, view_totals

SWEPT = ("threshold", "close_radius", "grow", "min_score", "min_area", "keep_unclaimed")

_VIEWS: dict[str, list[str]] = {}
_GT: dict[str, list[Rle]] = {}
_GRID: list = []
_DETECTIONS: dict[str, list[Detection]] = {}
_DISK: dict[str, Disk] = {}
_LOGIT_DIR: Path = Path()


def load_detections(path: str | Path) -> dict[str, list[Detection]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {stem: [Detection(d["score"], d["counts"]) for d in found] for stem, found in raw.items()}


def _init(views, gt, grid, detections, disk_geometry, logit_dir) -> None:
    global _VIEWS, _GT, _GRID, _DETECTIONS, _DISK, _LOGIT_DIR
    _VIEWS, _GT, _GRID, _DETECTIONS, _DISK, _LOGIT_DIR = (
        views, gt, grid, detections, disk_geometry, logit_dir)


def _score_stem(stem: str) -> np.ndarray:
    """Totals for every setting on one observation, pooled over its views.

    Settings are FusionParams, or PostprocessParams for the U-Net-only
    baseline. The thresholded U-Net mask and the grown detector regions are
    shared between the settings that use the same values.
    """
    logits = np.load(cache_path(_LOGIT_DIR, stem)).astype(np.float32)
    disk_mask = _DISK[stem].mask(logits.shape)
    detections = _DETECTIONS.get(stem, [])
    floor = min((p.min_score for p in _GRID if isinstance(p, FusionParams)), default=0.0)
    binaries: dict[tuple, np.ndarray] = {}
    regions: dict[int, list] = {}
    out = np.zeros((len(_GRID), len(COUNT_FIELDS)), dtype=np.float64)
    for index, params in enumerate(_GRID):
        if isinstance(params, FusionParams):
            key = (params.threshold, params.close_radius)
            if key not in binaries:
                binaries[key] = unet_binary(logits, disk_mask, *key)
            if params.grow not in regions:
                regions[params.grow] = grow_regions(detections, params.grow, logits.shape, floor)
            labels = assign(binaries[key], regions[params.grow], params.min_score,
                            params.min_area, params.keep_unclaimed)
        else:
            labels = logits_to_instances(logits, disk_mask, params)
        out[index] = view_totals(_VIEWS[stem], _GT, labels_to_rles(labels))
    return out


def score_stems(stems, grid, views, gt, detections, disk_geometry, logit_dir, workers) -> np.ndarray:
    """Per-observation totals, shape ``(len(stems), len(grid), len(COUNT_FIELDS))``."""
    per_stem = []
    with ProcessPoolExecutor(
        max_workers=workers or None, initializer=_init,
        initargs=(views, gt, grid, detections, disk_geometry, logit_dir),
    ) as pool:
        for n, totals in enumerate(pool.map(_score_stem, stems), start=1):
            per_stem.append(totals)
            if n % 10 == 0 or n == len(stems):
                print(f"  {n}/{len(stems)}", flush=True)
    return np.stack(per_stem)


def describe(params) -> str:
    if isinstance(params, FusionParams):
        return (f"fusion thr={params.threshold:.2f} close={params.close_radius} "
                f"grow={params.grow} score>={params.min_score:.2f} area={params.min_area} "
                f"keep={params.keep_unclaimed}")
    return (f"U-Net only thr={params.threshold:.2f} area={params.min_area} "
            f"gap={params.bridge_gap} close={params.close_radius} open={params.open_radius}")


def totals_row(totals: np.ndarray) -> dict:
    n_gt, n_pred, tp, fp, fn, iou_sum = totals
    return {"pq": float(pq_of(totals)), "sq": float(iou_sum / tp) if tp else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "n_pred": int(n_pred), "n_gt": int(n_gt)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", nargs="+", required=True, help="U-Net checkpoint(s)")
    parser.add_argument("--tta", choices=list(TTA_MODES), default="dihedral")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--logit-dir", required=True)
    parser.add_argument("--detections-val", required=True)
    parser.add_argument("--detections-test", default=None)
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--n-images", type=int, default=72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", required=True,
                        type=lambda text: parse_settings(text, PostprocessParams()),
                        help="U-Net-only settings to compare against")
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--close-radius", type=int, nargs="+", default=[3])
    parser.add_argument("--grow", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--min-score", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.5])
    parser.add_argument("--min-area", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--keep-unclaimed", type=int, nargs="+", default=[0, 2000])
    parser.add_argument("--holdout-top", type=int, default=5)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default=None, help="results JSON")
    parser.add_argument("--submission", default=None, help="write the best setting's test CSV here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    val_ids = load_split(args.split)["val"]
    cache_dir = Path(args.cache_dir)
    disk_geometry = load_disk_geometry(cache_dir / "disk.json")
    logit_dir = Path(args.logit_dir)

    tune = sample_stems(annotations, val_ids, args.n_images, seed=args.seed)
    holdout = [s for s in stems_of(annotations, val_ids) if s not in set(tune)]
    views: dict[str, list[str]] = {}
    for image_id in records_for_stems(annotations, val_ids, tune + holdout):
        views.setdefault(annotations.images[image_id].stem, []).append(image_id)
    gt = annotations.gt_dict(i for stem_views in views.values() for i in stem_views)
    detections = load_detections(args.detections_val)
    missing = [s for s in tune + holdout if s not in detections]
    if missing:
        raise SystemExit(f"{len(missing)} validation observations have no detections entry")

    device = select_device()
    ensure_logits(tune + holdout, args.checkpoint, disk_geometry, cache_dir / "flat", logit_dir,
                  args.tile, args.overlap, device, False, tta=args.tta)

    grid = [FusionParams(**dict(zip(SWEPT, combo))) for combo in itertools.product(
        args.threshold, args.close_radius, args.grow, args.min_score, args.min_area,
        args.keep_unclaimed)]
    print(f"tuning on {len(tune)} observations x {len(grid)} fusion settings")
    tune_totals = score_stems(tune, grid, views, gt, detections, disk_geometry, logit_dir,
                              args.workers).sum(axis=0)
    order = sorted(range(len(grid)), key=lambda i: -float(pq_of(tune_totals[i])))
    print(f"\n{'setting':<78} {'PQ':>7} {'SQ':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    for i in order[:15]:
        row = totals_row(tune_totals[i])
        print(f"{describe(grid[i]):<78} {row['pq']:>7.4f} {row['sq']:>6.3f} "
              f"{row['tp']:>5} {row['fp']:>5} {row['fn']:>5}")

    candidates = [grid[i] for i in order[: args.holdout_top]] + [args.baseline]
    per_stem = score_stems(holdout, candidates, views, gt, detections, disk_geometry,
                           logit_dir, args.workers)
    base = per_stem[:, -1]
    print(f"\nheld out from tuning: {len(holdout)} observations")
    report = []
    for k, params in enumerate(candidates):
        row = totals_row(per_stem[:, k].sum(axis=0))
        entry = {"setting": describe(params), **row}
        versus = "(baseline)"
        if k < len(candidates) - 1:
            delta, low, high = paired_bootstrap(per_stem[:, k], base, n_boot=5000)
            entry.update(delta_vs_baseline=delta, ci_low=low, ci_high=high)
            versus = f"{delta:+.4f} [{low:+.4f}, {high:+.4f}]"
        print(f"{describe(params):<78} {row['pq']:>7.4f}  {versus}")
        report.append(entry)

    best = grid[order[0]]
    results = {"tune_top": [{"setting": describe(grid[i]), **totals_row(tune_totals[i])}
                            for i in order[:30]],
               "holdout": report, "best": describe(best)}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.submission:
        if not args.detections_test:
            raise SystemExit("--submission needs --detections-test")
        test_stems = sorted(p.stem for p in test_image_paths(TEST_IMAGE_DIR))
        test_detections = load_detections(args.detections_test)
        ensure_logits(test_stems, args.checkpoint, disk_geometry, cache_dir / "flat", logit_dir,
                      args.tile, args.overlap, device, False, tta=args.tta)
        predictions = {}
        for stem in test_stems:
            logits = np.load(cache_path(logit_dir, stem)).astype(np.float32)
            labels = fuse(logits, disk_geometry[stem].mask(logits.shape),
                          test_detections.get(stem, []), best)
            predictions[stem] = labels_to_rles(labels)
        frame = build_submission(predictions)
        write_submission(frame, args.submission)
        print(f"wrote {args.submission}: {len(frame)} instances over {len(test_stems)} images "
              f"with {describe(best)}")


if __name__ == "__main__":
    main()
