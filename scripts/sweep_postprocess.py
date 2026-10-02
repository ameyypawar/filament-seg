"""Grid-search post-processing settings against the competition's own scoring.

Scoring mirrors the organisers' self-evaluation notebook: one set of
predictions per observation, matched against every annotator's view of that
observation separately, with TP/FP/FN pooled over all views. Scoring against
one random annotator per observation instead -- as this script used to --
optimises a slightly different objective and can reverse close calls.

The expensive half of the pipeline -- tiled full-resolution inference
(``filament_seg.model.ensemble_predict``) -- does not depend on any swept
setting, so it runs once per observation and the logit map is cached to disk
as float16 (2048x2048 float16 is 8 MB); the whole grid then reuses that cache.

With ``--holdout``, logits are also cached for the validation observations the
grid never saw, and the best few settings plus ``--baseline`` (the settings
currently in use) are scored there, each with a paired bootstrap over
observations for its difference from the baseline. That held-out comparison,
not the tuning score, decides whether a new setting is worth submitting.

The tuning observations, the held-out ones and the held-out results are saved
next to the results as ``<out>_subset.json``, so later comparisons
(scripts/compare_holdout.py) use exactly the same split instead of
re-deriving it.

    python scripts/sweep_postprocess.py --checkpoint outputs/model_best.pt --tta dihedral \\
        --n-images 72 --holdout \\
        --baseline threshold=0.7,min_area=400,bridge_gap=16,close_radius=3,open_radius=0
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import dataclasses
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from filament_seg.config import (
    CACHE_DIR,
    OUTPUT_ROOT,
    SPLIT_PATH,
    TRAIN_ANNOTATIONS,
    ensure_output_dir,
)
from filament_seg.data import (
    load_annotations,
    load_split,
    records_for_stems,
    sample_stems,
    stems_of,
)
from filament_seg.dataset import load_disk_geometry, load_model_input
from filament_seg.disk import Disk
from filament_seg.metrics import evaluate_image
from filament_seg.model import TTA_MODES, ensemble_predict, load_trained, select_device
from filament_seg.postprocess import PostprocessParams, logits_to_instances
from filament_seg.rle import Rle, labels_to_rles

#: Per-observation totals, in this order, summed over the observation's views.
COUNT_FIELDS = ("n_gt", "n_pred", "tp", "fp", "fn", "iou_sum")
#: The post-processing settings the grid varies (``fill_holes`` stays on).
SWEPT = ("threshold", "min_area", "bridge_gap", "close_radius", "open_radius", "min_confidence")

_VIEWS: dict[str, list[str]] = {}
_GT: dict[str, list[Rle]] = {}
_GRID: list[PostprocessParams] = []
_DISK: dict[str, Disk] = {}
_LOGIT_DIR: Path = Path()


def _cache_path(logit_dir: Path, stem: str) -> Path:
    return logit_dir / f"{stem}.npy"


def _check_cache_provenance(
    logit_dir: Path,
    checkpoint_paths: list[str],
    tta: str,
    tile: int,
    overlap: int,
    force: bool,
) -> None:
    """Make sure every cached logit map was produced by this model and setup.

    The cache is keyed by observation name alone, so without this a sweep run
    against a new checkpoint, TTA mode or tiling would silently reuse stale
    logits and tune against the wrong predictions. ``force`` deletes the old
    maps *before* recording the new provenance: otherwise a forced run that
    died part-way would leave the previous model's maps under the new record,
    and the next plain run would accept them.
    """
    checkpoints = []
    for path in checkpoint_paths:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        checkpoints.append({"path": str(resolved), "size": stat.st_size,
                            "mtime": int(stat.st_mtime)})
    wanted = {"checkpoints": checkpoints, "tta": tta, "tile": tile, "overlap": overlap}

    logit_dir.mkdir(parents=True, exist_ok=True)
    meta_path = logit_dir / "cache_meta.json"
    cached = sorted(logit_dir.glob("*.npy"))
    if force:
        for path in cached:
            path.unlink()
    elif meta_path.exists():
        found = json.loads(meta_path.read_text(encoding="utf-8"))
        if found != wanted:
            raise SystemExit(
                f"{logit_dir} holds logits from a different model or setup:\n"
                f"  cached: {found}\n  wanted: {wanted}\n"
                "use a different --out-dir, or --force to recompute"
            )
    elif cached:
        raise SystemExit(
            f"{logit_dir} holds cached logits with no record of which model made them; "
            "use a different --out-dir, or --force to recompute"
        )
    meta_path.write_text(json.dumps(wanted, indent=2), encoding="utf-8")


def ensure_logits(
    stems: list[str],
    checkpoint_paths: list[str],
    disk_geometry: dict[str, Disk],
    flat_dir: Path,
    logit_dir: Path,
    tile: int,
    overlap: int,
    device: torch.device,
    force: bool,
    tta: str = "none",
) -> None:
    """Cache the (ensemble-averaged, TTA-averaged) logits for every stem, once."""
    _check_cache_provenance(logit_dir, checkpoint_paths, tta, tile, overlap, force)
    pending = [s for s in stems if not _cache_path(logit_dir, s).exists()]
    if not pending:
        print(f"logits: {len(stems)}/{len(stems)} already cached, skipping model load")
        return

    print(f"device: {device}")
    models = [load_trained(path, device) for path in checkpoint_paths]
    print(f"models: {len(models)}; computing logits for {len(pending)}/{len(stems)} observations")
    for n, stem in enumerate(pending, start=1):
        x = load_model_input(flat_dir, stem, disk_geometry[stem])
        if x is None:
            raise SystemExit(f"no cached flat image for {stem} -- run scripts/preprocess.py")
        logits = ensemble_predict(models, x, tta=tta, tile=tile, overlap=overlap, device=device)
        # Written under a temporary name and renamed, so an interrupted run can
        # never leave a truncated map that a later run would take as complete.
        partial = logit_dir / f"{stem}.npy.partial"
        with open(partial, "wb") as handle:
            np.save(handle, logits.astype(np.float16))
        os.replace(partial, _cache_path(logit_dir, stem))
        if n % 5 == 0 or n == len(pending):
            print(f"  {n}/{len(pending)}", flush=True)


def _init(
    views: dict[str, list[str]],
    gt: dict[str, list[Rle]],
    grid: list[PostprocessParams],
    disk_geometry: dict[str, Disk],
    logit_dir: Path,
) -> None:
    global _VIEWS, _GT, _GRID, _DISK, _LOGIT_DIR
    _VIEWS, _GT, _GRID, _DISK, _LOGIT_DIR = views, gt, grid, disk_geometry, logit_dir


def _score_stem(stem: str) -> np.ndarray:
    """Totals for every grid point on one observation, pooled over its views."""
    logits = np.load(_cache_path(_LOGIT_DIR, stem)).astype(np.float32)
    disk_mask = _DISK[stem].mask(logits.shape)
    out = np.zeros((len(_GRID), len(COUNT_FIELDS)), dtype=np.float64)
    for index, params in enumerate(_GRID):
        pred_rles = labels_to_rles(logits_to_instances(logits, disk_mask, params))
        for image_id in _VIEWS[stem]:
            r = evaluate_image(image_id, _GT[image_id], pred_rles, with_fragmentation=False)
            out[index] += (r.n_gt, r.n_pred, r.tp, r.fp, r.fn, r.iou_sum)
    return out


def score_stems(
    stems: list[str],
    grid: list[PostprocessParams],
    views: dict[str, list[str]],
    gt: dict[str, list[Rle]],
    disk_geometry: dict[str, Disk],
    logit_dir: Path,
    workers: int,
) -> np.ndarray:
    """Per-observation totals, shape ``(len(stems), len(grid), len(COUNT_FIELDS))``."""
    per_stem = []
    with ProcessPoolExecutor(
        max_workers=workers or None,
        initializer=_init, initargs=(views, gt, grid, disk_geometry, logit_dir),
    ) as pool:
        for n, totals in enumerate(pool.map(_score_stem, stems), start=1):
            per_stem.append(totals)
            if n % 10 == 0 or n == len(stems):
                print(f"  {n}/{len(stems)}", flush=True)
    return np.stack(per_stem)


def pq_of(totals: np.ndarray) -> np.ndarray:
    """Pooled PQ from summed totals; works on any leading shape."""
    tp, fp, fn, iou_sum = (totals[..., COUNT_FIELDS.index(k)] for k in ("tp", "fp", "fn", "iou_sum"))
    denominator = tp + 0.5 * fp + 0.5 * fn
    return np.divide(iou_sum, denominator, out=np.zeros_like(iou_sum), where=denominator > 0)


def paired_bootstrap(
    per_stem_a: np.ndarray, per_stem_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """PQ(a) - PQ(b) on all observations, with a 95% interval from resampling them.

    Observations, not views, are resampled -- an observation's views share one
    set of predictions -- and both settings see the same resample each time.
    """
    n = per_stem_a.shape[0]
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot)
    # einsum rather than `@`: macOS's Accelerate BLAS raises spurious
    # floating-point warnings in matmul, and this is far too small to need BLAS.
    deltas = (pq_of(np.einsum("bs,sk->bk", weights, per_stem_a))
              - pq_of(np.einsum("bs,sk->bk", weights, per_stem_b)))
    full = float(pq_of(per_stem_a.sum(axis=0)) - pq_of(per_stem_b.sum(axis=0)))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return full, float(low), float(high)


def parse_params(text: str) -> PostprocessParams:
    """``"threshold=0.7,min_area=400"`` -> PostprocessParams, other fields at defaults."""
    defaults = PostprocessParams()
    values: dict = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, _, raw = item.partition("=")
        key = key.strip()
        if not hasattr(defaults, key):
            raise argparse.ArgumentTypeError(f"unknown post-processing setting {key!r}")
        kind = type(getattr(defaults, key))
        values[key] = raw.strip().lower() in ("1", "true", "yes") if kind is bool else kind(raw)
    return dataclasses.replace(defaults, **values)


def settings_of(params: PostprocessParams) -> dict:
    return {key: getattr(params, key) for key in SWEPT}


def summarize_grid(grid: list[PostprocessParams], totals: np.ndarray) -> list[dict]:
    """One result row per grid point, from totals of shape ``(len(grid), 6)``."""
    rows = []
    for params, (n_gt, n_pred, tp, fp, fn, iou_sum) in zip(grid, totals):
        denominator = tp + 0.5 * fp + 0.5 * fn
        rows.append({
            **settings_of(params),
            "pq": float(iou_sum / denominator) if denominator else 0.0,
            "sq": float(iou_sum / tp) if tp else 0.0,
            "rq": float(tp / denominator) if denominator else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "n_pred": int(n_pred), "n_gt": int(n_gt),
        })
    return rows


def print_rows(rows: list[dict]) -> None:
    header = (f"{'thr':>5} {'min_area':>9} {'gap':>4} {'close':>6} {'open':>5} {'conf':>5} "
              f"{'PQ':>7} {'SQ':>6} {'RQ':>6} {'TP':>5} {'FP':>6} {'FN':>5} {'pred/gt':>8}")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['threshold']:>5.2f} {row['min_area']:>9} {row['bridge_gap']:>4} "
              f"{row['close_radius']:>6} {row['open_radius']:>5} {row['min_confidence']:>5.2f} "
              f"{row['pq']:>7.4f} {row['sq']:>6.3f} {row['rq']:>6.3f} {row['tp']:>5} "
              f"{row['fp']:>6} {row['fn']:>5} {row['n_pred'] / max(row['n_gt'], 1):>8.2f}")


def holdout_report(
    candidates: list[PostprocessParams],
    per_stem: np.ndarray,
    tune_pq: dict[int, float],
    baseline_index: int | None,
) -> list[dict]:
    """Held-out PQ per candidate, and its paired difference from the baseline."""
    report = []
    for index, params in enumerate(candidates):
        entry = {"settings": settings_of(params),
                 "tune_pq": tune_pq[index],
                 "holdout_pq": float(pq_of(per_stem[:, index].sum(axis=0))),
                 "is_baseline": index == baseline_index}
        if baseline_index is not None and index != baseline_index:
            delta, low, high = paired_bootstrap(per_stem[:, index], per_stem[:, baseline_index])
            entry.update(delta_vs_baseline=delta, ci_low=low, ci_high=high)
        report.append(entry)
    return report


def print_holdout(report: list[dict], n_stems: int, n_views: int) -> None:
    print(f"\nheld out from tuning: {n_stems} observations ({n_views} annotator views)")
    print(f"{'settings':<58} {'tune PQ':>8} {'held-out':>9}  vs baseline [95% CI]")
    for entry in report:
        s = entry["settings"]
        label = (f"thr={s['threshold']:.2f} area={s['min_area']} gap={s['bridge_gap']} "
                 f"close={s['close_radius']} open={s['open_radius']} conf={s['min_confidence']:.2f}")
        if entry["is_baseline"]:
            versus = "(baseline)"
        elif "delta_vs_baseline" in entry:
            versus = (f"{entry['delta_vs_baseline']:+.4f} "
                      f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}]")
        else:
            versus = ""
        print(f"{label:<58} {entry['tune_pq']:>8.4f} {entry['holdout_pq']:>9.4f}  {versus}")


def build_grid(args: argparse.Namespace) -> list[PostprocessParams]:
    grid = [
        PostprocessParams(**dict(zip(SWEPT, combo)))
        for combo in itertools.product(
            args.threshold, args.min_area, args.bridge_gap, args.close_radius,
            args.open_radius, args.min_confidence,
        )
    ]
    if args.baseline is not None and args.baseline not in grid:
        grid.append(args.baseline)
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="one or more checkpoints; several are averaged as an ensemble")
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--out-dir", default=None,
                        help="logit cache; defaults to outputs/val_logits_<tta mode>")
    parser.add_argument("--out", default=None,
                        help="results JSON; defaults to outputs/sweep_postprocess.json")
    parser.add_argument("--tta", choices=list(TTA_MODES), default="none",
                        help="average logits over flipped/rotated copies of each image")
    parser.add_argument("--n-images", type=int, default=24,
                        help="validation observations to tune on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", action="store_true",
                        help="also score the best settings on the other validation observations")
    parser.add_argument("--holdout-top", type=int, default=5)
    parser.add_argument("--baseline", type=parse_params, default=None,
                        help="settings currently in use, e.g. threshold=0.7,min_area=400,"
                             "bridge_gap=16,close_radius=3,open_radius=0")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device", default=None, help="defaults to CUDA, then MPS, then CPU")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="delete and recompute the cached logits")
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--min-area", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--bridge-gap", type=int, nargs="+", default=[12, 16, 24])
    parser.add_argument("--close-radius", type=int, nargs="+", default=[3])
    # 0 disables opening entirely (filament_seg.postprocess.clean_binary skips
    # it) -- worth trying since opening is exactly what can erase thin barbs.
    parser.add_argument("--open-radius", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--min-confidence", type=float, nargs="+", default=[0.0])
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else select_device()

    annotations = load_annotations(args.annotations)
    val_ids = load_split(args.split)["val"]
    cache_dir = Path(args.cache_dir)
    disk_geometry = load_disk_geometry(cache_dir / "disk.json")

    tune = [s for s in sample_stems(annotations, val_ids, args.n_images, seed=args.seed)
            if s in disk_geometry]
    holdout = []
    if args.holdout:
        holdout = [s for s in stems_of(annotations, val_ids)
                   if s not in set(tune) and s in disk_geometry]
    views: dict[str, list[str]] = {}
    for image_id in records_for_stems(annotations, val_ids, tune + holdout):
        views.setdefault(annotations.images[image_id].stem, []).append(image_id)
    gt = annotations.gt_dict(i for stem_views in views.values() for i in stem_views)

    logit_dir = Path(args.out_dir or (OUTPUT_ROOT / f"val_logits_{args.tta}"))
    ensure_logits(tune + holdout, args.checkpoint, disk_geometry, cache_dir / "flat",
                  logit_dir, args.tile, args.overlap, device, args.force, tta=args.tta)

    grid = build_grid(args)
    print(f"tuning on {len(tune)} observations "
          f"({sum(len(views[s]) for s in tune)} annotator views) x {len(grid)} settings")
    tune_totals = score_stems(tune, grid, views, gt, disk_geometry, logit_dir, args.workers)
    rows = summarize_grid(grid, tune_totals.sum(axis=0))
    order = sorted(range(len(grid)), key=lambda i: -rows[i]["pq"])
    print_rows([rows[i] for i in order[: args.top]])

    out = Path(args.out or (ensure_output_dir() / "sweep_postprocess.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([rows[i] for i in order], indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    subset = {
        "seed": args.seed, "n_images": args.n_images,
        "checkpoints": [str(Path(p).resolve()) for p in args.checkpoint],
        "tta": args.tta, "tile": args.tile, "overlap": args.overlap,
        "tune_stems": tune, "holdout_stems": holdout,
    }
    if holdout:
        chosen = list(order[: args.holdout_top])
        baseline_index = None
        if args.baseline is not None:
            baseline_grid_index = grid.index(args.baseline)
            if baseline_grid_index not in chosen:
                chosen.append(baseline_grid_index)
            baseline_index = chosen.index(baseline_grid_index)
        candidates = [grid[i] for i in chosen]
        per_stem = score_stems(holdout, candidates, views, gt, disk_geometry, logit_dir,
                               args.workers)
        report = holdout_report(candidates, per_stem,
                                {k: rows[i]["pq"] for k, i in enumerate(chosen)}, baseline_index)
        print_holdout(report, len(holdout), sum(len(views[s]) for s in holdout))
        subset["holdout_results"] = report

    subset_path = out.with_name(out.stem + "_subset.json")
    subset_path.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    print(f"wrote {subset_path}")


if __name__ == "__main__":
    main()
