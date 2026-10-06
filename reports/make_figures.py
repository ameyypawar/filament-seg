"""Figures and diagnostic numbers for the technical report.

Regenerates every figure in ``reports/figures/`` and ``reports/numbers.json``
from local files only: cached U-Net logits, merged detector instances, the
annotations, the saved cross-fitted totals and the validated test submission.
Nothing is trained, downloaded or submitted, and nothing is written outside
``reports/``.

The pipeline reported is the validated entry, "V": run 3's last-epoch U-Net
(8-way dihedral TTA) fused with detectors 1+2+4 at the setting in ``PARAMS``.
Scoring follows the organisers, as scripts/fuse_detections.py does: each
annotator's view of an observation is scored on its own and TP, FP, FN and the
matched IoU are pooled over every view. That setting was chosen on these same
144 observations, so V's validation numbers here are in-sample for the fusion
knobs; the out-of-sample estimate is the cross-fitted one in ``fig_seeds``.

Besides the figures, numbers.json carries bootstrap intervals for the
pipelines in the report's progression table and a detectors-only ablation (the
same merged detections without the U-Net), cross-fitted like V.

    .venv/bin/python reports/make_figures.py [--workers 8]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Write nothing outside reports/, not even bytecode caches for imported modules.
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parent.parent
# Make `python reports/make_figures.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(REPO))

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from filament_seg.config import CACHE_DIR as CONFIG_CACHE_DIR, IMAGE_SIZE, TEST_IMAGE_DIR
from filament_seg.data import (
    load_annotations,
    load_split,
    records_for_stems,
    sample_stems,
    stems_of,
    test_image_paths,
)
from filament_seg.disk import Disk
from filament_seg.fusion import Detection, FusionParams, fuse, grow_regions, unet_binary
from filament_seg.metrics import ImageResult, evaluate_image, match_instances, summarize
from filament_seg.rle import (
    Rle,
    iou_matrix,
    labels_to_rles,
    mask_to_rle,
    read_submission,
    rle_areas,
    rle_from_counts,
    rle_to_mask,
)
from filament_seg.scoring import (
    COUNT_FIELDS,
    bootstrap_deltas,
    paired_bootstrap,
    pq_of,
    view_totals,
)

# --- Inputs and outputs ------------------------------------------------------

STEP3 = REPO / "outputs" / "step3"
#: The preprocessing cache that scripts/preprocess.py writes (data/cache).
CACHE_DIR = CONFIG_CACHE_DIR
LOGIT_DIR = STEP3 / "logits_v5last_dihedral"
DETECTIONS_VAL = STEP3 / "detections_val_det124L.json"
SUBMISSION_TEST = STEP3 / "submission_validated_v5last_det124L.csv"
SPLIT = REPO / "outputs" / "splits.json"

OUT_DIR = REPO / "reports"
FIG_DIR = OUT_DIR / "figures"
NUMBERS = OUT_DIR / "numbers.json"

#: V's fusion setting: the best on all 144 validation observations.
PARAMS = FusionParams(threshold=0.5, close_radius=3, grow=4, min_score=0.30, min_area=200,
                      keep_unclaimed=0)

#: scripts/fuse_detections.py's split: a seed-0 sample of 72 observations
#: tunes, the other 72 are held out, and cross-fitting swaps the roles.
N_TUNE = 72
#: Bootstrap intervals: resamples and seed, the draw of filament_seg.scoring.
N_BOOT = 2000
N_BOOT_PAIRED = 10000
BOOT_SEED = 0

#: Detectors-only ablation: no U-Net; the merged detections claim their own
#: masks. Grid swept, as (min_score, min_area) pairs.
DET_ONLY_SCORES = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
DET_ONLY_AREAS = (100, 200, 400, 600, 800, 1200)
DET_ONLY_GRID = list(product(DET_ONLY_SCORES, DET_ONLY_AREAS))

# --- Error causes ------------------------------------------------------------
# Each unmatched annotated filament (FN) and unmatched prediction (FP) of one
# view gets the first cause that applies, in this order. "IoU" is the best IoU
# with any prediction (FN) or any annotated filament of the view (FP); the
# detector causes look at the merged detections before fusion, at any score.

FN_CAUSES = {
    "near_miss": ("Near miss (IoU 0.25–0.5)", "best predicted IoU in (0.25, 0.5]"),
    "low_overlap": ("Low overlap (IoU ≤ 0.25)", "best predicted IoU in (0, 0.25]"),
    "dropped_small": ("Detected, < 200 U-Net px",
                      "no prediction overlaps it; the best detection has IoU > 0.5 and score "
                      ">= 0.30 but claimed fewer than 200 unclaimed U-Net pixels, so it was "
                      "dropped"),
    "below_score": ("Detected, score < 0.30",
                    "no prediction overlaps it; the best detection has IoU > 0.5 but score "
                    "< 0.30"),
    "poor_detection": ("Detector IoU 0.25–0.5 only",
                       "no prediction overlaps it; the best detection has IoU in (0.25, 0.5]"),
    "no_detection": ("No detection", "no prediction overlaps it; no detection has IoU > 0.25"),
}
#: Never expected: a confident detection matched the filament and was kept,
#: yet its instance does not touch the filament. Counted so it cannot hide.
FN_KEPT_DISJOINT = "kept_disjoint"
#: The FN panel's two groups: whether any prediction overlaps the filament.
FN_GROUPS = [
    ("Prediction overlaps it", ["near_miss", "low_overlap"]),
    ("No prediction overlaps it", ["below_score", "dropped_small", "poor_detection",
                                   "no_detection"]),
]
FP_CAUSES = {
    "near_miss": ("Near miss (IoU 0.25–0.5)", "best annotated IoU in (0.25, 0.5]"),
    "low_overlap": ("Low overlap (IoU ≤ 0.25)", "best annotated IoU in (0, 0.25]"),
    "no_overlap": ("No overlap", "touches no annotated filament of the view"),
}

# --- Public leaderboard history ------------------------------------------------
# Every submission in order, as the board shows it (two decimals).

PUBLIC_SUBMISSIONS = [
    ("Classical baseline", "classical baseline", 0.08),
    ("U-Net, 1 epoch", "U-Net 1 epoch", 0.22),
    ("U-Net, 30 epochs, tuned", "U-Net 30 epochs tuned", 0.30),
    ("+ 8-way TTA", "+ 8-way TTA", 0.32),
    ("Limb fix, all annotators", "corrected limb + all-annotator selection", 0.32),
    ("Fusion, det 1", "detector fusion (det1)", 0.36),
    ("Fusion, det 2", "det2", 0.36),
    ("Fusion, det 1+2", "det1+det2", 0.36),
    ("Last-epoch U-Net", "last-epoch U-Net", 0.37),
    ("All data: 2 U-Nets, 3 det.", "all-data (2 U-Nets, 3 detectors)", 0.37),
    ("All data: 4 U-Nets, 6 det.", "all-data (4 U-Nets, 6 detectors)", 0.37),
    ("V: validated det 1+2+4", "validated det1+2+4", 0.37),
]
#: The jumps annotated on the progress figure, as (name, from, to) positions
#: in PUBLIC_SUBMISSIONS. The U-Net jump spans its first two submissions.
PUBLIC_JUMPS = [("U-Net", 0, 2), ("TTA", 2, 3), ("Detector fusion", 4, 5)]

# --- Cross-fitted runs ---------------------------------------------------------
# (label, totals file, kind); kind "seed" marks a retrain of the row above's
# recipe with another seed. Files are scripts/fuse_detections.py --save-totals.

SEED_GROUPS = [
    ("U-Net variants, detectors 1+2", [
        ("Run 3 (used in both entries)", "totals_det12_v5last_wide.npz", "base"),
        ("Run 3 recipe, seed 1", "totals_det12_u2seed1last.npz", "seed"),
        ("All annotator views", "totals_det12_u1views_alllast.npz", "variant"),
        ("Union target", "totals_det12_u1unionlast.npz", "variant"),
        ("Tversky loss", "totals_det12_u2tverskylast.npz", "variant"),
    ]),
    ("Single detectors, run-3 U-Net", [
        ("Det 1: YOLO11s, 1024 px", "totals_det1_v5last.npz", "variant"),
        ("Det 2: YOLO11m, 1280 px", "totals_det2_v5last.npz", "base"),
        ("Det 5: det 2 recipe, seed 1", "totals_det5L_v5last.npz", "seed"),
        ("Det 3: YOLO11s, 1280 px", "totals_det3_v5last.npz", "variant"),
        ("Det 4: YOLO11m, 1536 px", "totals_det4L_v5last.npz", "variant"),
    ]),
    ("Detector ensembles, run-3 U-Net", [
        ("Det 1+2", "totals_det12_v5last.npz", "variant"),
        ("Det 1+2+4 (V)", "totals_det124L_v5last.npz", "final"),
        ("Det 1+2+3", "totals_det123_v5last.npz", "variant"),
        ("Det 1+2+4+5", "totals_det1245L_v5last.npz", "variant"),
    ]),
]
#: The U-Net-only pipeline rides along in every totals file; this one's is
#: run 3's last epoch, thresholded and grouped by connected components.
BASELINE_TOTALS = "totals_det12_v5last.npz"
#: The run whose cross-fitted totals are V's own, for the consistency check.
V_TOTALS = "totals_det124L_v5last.npz"
#: Run 3 with detectors 1+2 appears twice in the seed figure: tuned over the
#: wider fusion grid the U-Net variants used, and over the usual one.
RUN3_WIDE, RUN3_NARROW = "totals_det12_v5last_wide.npz", "totals_det12_v5last.npz"
#: The report's progression table: (key, label, totals file, array).
#: totals_det12.npz used the U-Net checkpoint selected on validation (epoch 23).
PROGRESSION = [
    ("unet_only_selected_epoch", "U-Net only, selected-epoch weights", "totals_det12.npz",
     "baseline"),
    ("unet_only_last_epoch", "U-Net only, last-epoch weights", "totals_det12_v5last.npz",
     "baseline"),
    ("fusion_det12_selected_epoch", "Fusion, detectors 1+2, selected-epoch U-Net",
     "totals_det12.npz", "crossfit"),
    ("fusion_det12_last_epoch", "Fusion, detectors 1+2, last-epoch U-Net",
     "totals_det12_v5last.npz", "crossfit"),
    ("v_fusion_det124", "V: fusion, detectors 1+2+4, last-epoch U-Net",
     "totals_det124L_v5last.npz", "crossfit"),
]

# --- Figure style (IEEE) ----------------------------------------------------------

SINGLE_COLUMN = 3.5  # inches
DOUBLE_COLUMN = 7.16
#: No text smaller than this, in points at print size.
MIN_TEXT = 7
#: Okabe & Ito's colour-blind-safe palette.
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
#: Outcome colours. Over the grey solar image: TP sky blue, FP vermillion, FN
#: yellow. On white the same hues, darkened where the light one would not stay
#: legible (a gold that still separates from vermillion under deuteranopia).
OUTCOME_ON_IMAGE = {"tp": OI["sky"], "fp": OI["vermillion"], "fn": OI["yellow"]}
OUTCOME_ON_WHITE = {"tp": OI["blue"], "fp": OI["vermillion"], "fn": "#CCAC00"}
FN_EDGE_ON_WHITE = "#8F7800"
OUTCOME_LABELS = {"tp": "TP: matched prediction", "fp": "FP: unmatched prediction",
                  "fn": "FN: missed filament (annotator 1)"}
#: Instance identity colours (purples, greens, pinks, browns, grey): none of
#: the outcome hues, so an instance's colour cannot be read as an outcome.
INSTANCE_COLORS = ["#AA4499", "#117733", "#DDDDDD", "#CC6677", "#882255", "#A6DBA0", "#8C613C"]
INK = "#1a1a1a"
MUTED = "#5f5f5f"
GREY = "#8c8c8c"
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": MIN_TEXT,
    "ytick.labelsize": MIN_TEXT,
    "legend.fontsize": MIN_TEXT,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    # TrueType embedding: IEEE PDF checks reject Type 3 fonts.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
}

# --- Fusion, replayed --------------------------------------------------------------


def replay_claims(
    logits: np.ndarray, disk_mask: np.ndarray, detections: list[Detection]
) -> tuple[np.ndarray, dict[int, str], list[int]]:
    """``fusion.assign``, step by step, keeping what became of every detection.

    Returns the label map, each detection's fate ("kept", "dropped" for a
    claim under ``min_area``, "below_score", "empty") and, for instance label
    ``k``, the detection it came from at position ``k - 1``.
    """
    binary = unet_binary(logits, disk_mask, PARAMS.threshold, PARAMS.close_radius)
    labels = np.zeros(binary.shape, dtype=np.int32)
    claimed = np.zeros(binary.shape, dtype=bool)
    fates: dict[int, str] = {}
    sources: list[int] = []
    # grow_regions' order: descending score, ties in input order.
    for index in sorted(range(len(detections)), key=lambda i: -detections[i].score):
        if detections[index].score < PARAMS.min_score:
            fates[index] = "below_score"
            continue
        regions = grow_regions([detections[index]], PARAMS.grow, binary.shape, PARAMS.min_score)
        if not regions:
            fates[index] = "empty"
            continue
        region = regions[0]
        window = (slice(region.y0, region.y1), slice(region.x0, region.x1))
        pixels = binary[window] & region.mask & ~claimed[window]
        if int(pixels.sum()) < PARAMS.min_area:
            fates[index] = "dropped"
            continue
        labels[window][pixels] = len(sources) + 1
        claimed[window] |= pixels
        sources.append(index)
        fates[index] = "kept"
    return labels, fates, sources


def fuse_observation(
    stem: str, detections: list[Detection], disk: Disk
) -> tuple[list[Rle], dict[int, str], float]:
    """V's instances for one observation, the fate of each detection, and the
    wall time of the fusion step itself (logits and detections in, RLEs out)."""
    logits = np.load(LOGIT_DIR / f"{stem}.npy").astype(np.float32)
    started = time.perf_counter()
    disk_mask = disk.mask(logits.shape)
    labels = fuse(logits, disk_mask, detections, PARAMS)
    preds = labels_to_rles(labels)
    seconds = time.perf_counter() - started
    replayed, fates, sources = replay_claims(logits, disk_mask, detections)
    if not np.array_equal(labels, replayed):
        raise RuntimeError(f"{stem}: the replayed claims disagree with fusion.fuse")
    if len(preds) != len(sources):
        raise RuntimeError(f"{stem}: {len(preds)} instances from {len(sources)} kept detections")
    return preds, fates, seconds


def detector_only_totals(views: list[str], gt: dict[str, list[Rle]],
                         detections: list[Detection], disk: Disk) -> np.ndarray:
    """Totals for every DET_ONLY_GRID setting: the merged detections alone, no U-Net.

    Each detection, in descending score, claims the not yet claimed pixels of
    its own full-resolution mask that lie on the disk; claims under
    ``min_area`` are dropped, so instances never overlap. A lower score
    cut-off only appends detections to the end of that order, so one pass
    per ``min_area`` serves every cut-off.
    """
    disk_mask = disk.mask(IMAGE_SIZE)
    # grow=0: each detection's own mask, cropped to its box, best first.
    regions = grow_regions(detections, 0, IMAGE_SIZE, min(DET_ONLY_SCORES))
    out = np.zeros((len(DET_ONLY_GRID), len(COUNT_FIELDS)), dtype=np.float64)
    for min_area in DET_ONLY_AREAS:
        claimed = np.zeros(IMAGE_SIZE, dtype=bool)
        kept: list[tuple[float, Rle]] = []
        for region in regions:
            window = (slice(region.y0, region.y1), slice(region.x0, region.x1))
            pixels = region.mask & disk_mask[window] & ~claimed[window]
            if int(pixels.sum()) < min_area:
                continue
            claimed[window] |= pixels
            instance = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            instance[window] = pixels
            kept.append((region.score, mask_to_rle(instance)))
        for min_score in DET_ONLY_SCORES:
            preds = [rle for score, rle in kept if score >= min_score]
            out[DET_ONLY_GRID.index((min_score, min_area))] = view_totals(views, gt, preds)
    return out


# --- Per-observation analysis (runs in worker processes) ------------------------------

_WORKER: dict = {}


def _init_worker(views, gt, detections, disks) -> None:
    _WORKER.update(views=views, gt=gt, detections=detections, disks=disks)


def fn_cause(pred_ious: np.ndarray, raw_ious: np.ndarray, raw_scores: np.ndarray,
             fates: dict[int, str]) -> str:
    best = float(pred_ious.max()) if pred_ious.size else 0.0
    if best > 0.25:
        return "near_miss"
    if best > 0:
        return "low_overlap"
    if raw_ious.size == 0:
        return "no_detection"
    k = int(raw_ious.argmax())
    raw_best = float(raw_ious[k])
    if raw_best > 0.5:
        if raw_scores[k] >= PARAMS.min_score:
            return "dropped_small" if fates[k] == "dropped" else FN_KEPT_DISJOINT
        return "below_score"
    if raw_best > 0.25:
        return "poor_detection"
    return "no_detection"


def fp_cause(gt_ious: np.ndarray) -> str:
    best = float(gt_ious.max()) if gt_ious.size else 0.0
    if best > 0.25:
        return "near_miss"
    return "low_overlap" if best > 0 else "no_overlap"


def _boxes(rles: list[Rle]) -> list[list[float]]:
    return mask_utils.toBbox(list(rles)).tolist() if rles else []


def analyse_observation(stem: str) -> dict:
    """Score V on every view of one observation and classify each error."""
    views: list[str] = _WORKER["views"][stem]
    gt: dict[str, list[Rle]] = _WORKER["gt"]
    detections: list[Detection] = _WORKER["detections"][stem]
    disk: Disk = _WORKER["disks"][stem]
    preds, fates, fuse_seconds = fuse_observation(stem, detections, disk)
    pred_area = rle_areas(preds)
    raw = [rle_from_counts(d.counts) for d in detections]
    raw_scores = np.array([d.score for d in detections])

    out = {"stem": stem, "views": views, "results": [], "n_pred": len(preds),
           "fuse_seconds": fuse_seconds,
           "pred_area": pred_area.tolist(), "fates": dict(Counter(fates.values())),
           "tp_area": [], "fn": [], "fp": [], "fp_matched_elsewhere": [], "gt_best_iou": [],
           "outcome": {}, "boxes": {"pred": _boxes(preds), "gt": {}},
           "det_only": detector_only_totals(views, gt, detections, disk)}
    matched_preds: dict[str, set[int]] = {}
    for view in views:
        g = gt[view]
        result: ImageResult = evaluate_image(view, g, preds, with_fragmentation=True)
        matches, ious = match_instances(g, preds)
        if len(matches) != result.tp:
            raise RuntimeError(f"{view}: matching disagrees with evaluate_image")
        out["results"].append(result)
        gt_hit = {gi for gi, _, _ in matches}
        matched_preds[view] = {pj for _, pj, _ in matches}
        out["outcome"][view] = {
            "tp": [[gi, pj] for gi, pj, _ in matches],
            "fn": [a for a in range(len(g)) if a not in gt_hit],
            "fp": [b for b in range(len(preds)) if b not in matched_preds[view]],
        }
        out["boxes"]["gt"][view] = _boxes(g)
        out["gt_best_iou"].extend(ious.max(axis=1).tolist() if ious.shape[1] else [0.0] * len(g))
        g_area = rle_areas(g)
        raw_ious = iou_matrix(g, raw)
        for a in range(len(g)):
            if a in gt_hit:
                out["tp_area"].append(float(g_area[a]))
            else:
                cause = fn_cause(ious[a], raw_ious[a], raw_scores, fates)
                out["fn"].append((cause, float(g_area[a])))
        for b in out["outcome"][view]["fp"]:
            out["fp"].append((fp_cause(ious[:, b]), float(pred_area[b])))
    # On an observation with several annotators, is an FP in one view a match in another?
    if len(views) > 1:
        for view in views:
            others = [v for v in views if v != view]
            for b in out["outcome"][view]["fp"]:
                out["fp_matched_elsewhere"].append(any(b in matched_preds[o] for o in others))
    return out


# --- Loading ------------------------------------------------------------------------


@dataclass
class Inputs:
    stems: list[str]
    #: fuse_detections.py's halves: the seed-0 sample that tunes, and the rest.
    tune: list[str]
    holdout: list[str]
    views: dict[str, list[str]]
    gt: dict[str, list[Rle]]
    detections: dict[str, list[Detection]]
    disks: dict[str, Disk]


def load_inputs() -> Inputs:
    # Imported here so worker processes never pay for torch/albumentations.
    from filament_seg.dataset import load_disk_geometry

    annotations = load_annotations()
    val_ids = load_split(SPLIT)["val"]
    stems = stems_of(annotations, val_ids)
    tune = sample_stems(annotations, val_ids, N_TUNE, seed=0)
    holdout = [s for s in stems if s not in set(tune)]
    views: dict[str, list[str]] = {}
    for image_id in records_for_stems(annotations, val_ids, stems):
        views.setdefault(annotations.images[image_id].stem, []).append(image_id)
    views = {stem: sorted(ids) for stem, ids in views.items()}
    gt = annotations.gt_dict(i for ids in views.values() for i in ids)
    raw = json.loads(DETECTIONS_VAL.read_text(encoding="utf-8"))
    missing = [s for s in stems if s not in raw]
    if missing:
        raise SystemExit(f"{len(missing)} validation observations have no detections entry")
    detections = {s: [Detection(d["score"], d["counts"]) for d in raw[s]] for s in stems}
    geometry = load_disk_geometry(CACHE_DIR / "disk.json")
    disks = {s: geometry[s] for s in stems}
    return Inputs(stems, tune, holdout, views, gt, detections, disks)


def analyse_validation(inputs: Inputs, workers: int) -> list[dict]:
    per_stem = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(inputs.views, inputs.gt, inputs.detections, inputs.disks),
    ) as pool:
        for n, result in enumerate(pool.map(analyse_observation, inputs.stems), start=1):
            per_stem.append(result)
            if n % 24 == 0 or n == len(inputs.stems):
                print(f"  scored {n}/{len(inputs.stems)} observations", flush=True)
    return per_stem


# --- Numbers --------------------------------------------------------------------------


def _round(value, digits: int = 6):
    """JSON-ready copy with numpy scalars converted and floats rounded."""
    if isinstance(value, dict):
        return {str(k): _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, np.ndarray):
        return _round(value.tolist(), digits)
    return value


def _quartiles(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    return {"n": int(values.size), "mean": float(values.mean()), "p25": float(p25),
            "median": float(p50), "p75": float(p75), "min": float(values.min()),
            "max": float(values.max())}


def view_totals_of(result: ImageResult) -> np.ndarray:
    """One view's six COUNT_FIELDS totals."""
    return np.array([result.n_gt, result.n_pred, result.tp, result.fp, result.fn,
                     result.iou_sum], dtype=np.float64)


def observation_totals(entry: dict) -> np.ndarray:
    """The six COUNT_FIELDS totals of one observation, summed over its views."""
    return np.sum([view_totals_of(r) for r in entry["results"]], axis=0)


def _field(totals: np.ndarray, name: str) -> np.ndarray:
    return totals[..., COUNT_FIELDS.index(name)]


def sq_of(totals: np.ndarray) -> np.ndarray:
    """Pooled segmentation quality, sum IoU / TP, on any leading shape."""
    tp, iou_sum = _field(totals, "tp"), _field(totals, "iou_sum")
    return np.divide(iou_sum, tp, out=np.zeros_like(iou_sum), where=tp > 0)


def rq_of(totals: np.ndarray) -> np.ndarray:
    """Pooled recognition quality, TP / (TP + FP/2 + FN/2), on any leading shape."""
    tp, fp, fn = (_field(totals, k) for k in ("tp", "fp", "fn"))
    denominator = tp + 0.5 * fp + 0.5 * fn
    return np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def bootstrap_ci(per_unit: np.ndarray, statistic=pq_of) -> list[float]:
    """95% percentile interval of a pooled statistic, resampling the rows of ``per_unit``.

    Rows are units (observations, or views) of six COUNT_FIELDS totals. The
    resampling counts are drawn exactly as filament_seg.scoring.bootstrap_deltas
    draws them, so every pipeline scored on the same observations sees the
    same resamples.
    """
    n = per_unit.shape[0]
    weights = np.random.default_rng(BOOT_SEED).multinomial(n, np.full(n, 1.0 / n), size=N_BOOT)
    # einsum rather than `@`, as in scoring.py: Accelerate BLAS warns spuriously in matmul.
    values = statistic(np.einsum("bs,sk->bk", weights, per_unit))
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def pooled_summary(rows: np.ndarray, with_ci: bool = True) -> dict:
    """PQ, SQ, RQ and counts of per-observation totals, with PQ's interval."""
    totals = rows.sum(axis=0)
    out = {"n_observations": int(rows.shape[0]), "pq": float(pq_of(totals)),
           "sq": float(sq_of(totals)), "rq": float(rq_of(totals)),
           **{k: int(_field(totals, k)) for k in ("tp", "fp", "fn")}}
    if with_ci:
        out["pq_ci95"] = bootstrap_ci(rows)
    return out


def _pq_sq_rq(tp: float, fp: float, fn: float, iou_sum: float) -> dict:
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {"pq": iou_sum / denominator if denominator else float("nan"),
            "sq": iou_sum / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 0.0}


def validation_numbers(per_stem: list[dict]) -> dict:
    results = [r for entry in per_stem for r in entry["results"]]
    summary = summarize(results)
    per_obs = np.stack([observation_totals(e) for e in per_stem])
    obs_pq = pq_of(per_obs)
    tp, fp, fn = summary["tp"], summary["fp"], summary["fn"]

    # Annotator groups: the first two digits of the view id. Two intervals per
    # group: resampling its views, and resampling its observations with their
    # views kept together (views of one observation share one set of
    # predictions, so they are not independent).
    views_of_group: dict[str, list[tuple[str, ImageResult]]] = {}
    for e in per_stem:
        for r in e["results"]:
            views_of_group.setdefault(r.image_id[:2], []).append((e["stem"], r))
    by_group = {}
    for group, members in sorted(views_of_group.items()):
        s = summarize([r for _, r in members])
        per_view = np.stack([view_totals_of(r) for _, r in members])
        clusters: dict[str, np.ndarray] = {}
        for stem, r in members:
            clusters[stem] = clusters.get(stem, 0) + view_totals_of(r)
        by_group[group] = {
            "n_views": len(members),
            "n_observations": len(clusters),
            "pq": s["pq_pooled"],
            "pq_ci95_views": bootstrap_ci(per_view),
            "pq_ci95_observations": bootstrap_ci(np.stack(list(clusters.values()))),
            "sq": s["sq"], "rq": s["rq"],
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "n_gt": s["n_gt"],
        }
    group_pqs = [g["pq"] for g in by_group.values()]
    pred_area = np.concatenate([e["pred_area"] for e in per_stem])
    n_pred_obs = np.array([e["n_pred"] for e in per_stem])
    fuse_seconds = np.array([e["fuse_seconds"] for e in per_stem])
    fates = Counter()
    for e in per_stem:
        fates.update(e["fates"])
    return {
        "n_observations": len(per_stem),
        "n_views": len(results),
        "views_per_observation": dict(sorted(Counter(str(len(e["views"]))
                                                     for e in per_stem).items())),
        "pq": summary["pq_pooled"], "sq": summary["sq"], "rq": summary["rq"],
        "pq_ci95": bootstrap_ci(per_obs, pq_of),
        "sq_ci95": bootstrap_ci(per_obs, sq_of),
        "rq_ci95": bootstrap_ci(per_obs, rq_of),
        "bootstrap": {"unit": "observation (its views kept together)", "n_resamples": N_BOOT,
                      "seed": BOOT_SEED, "interval": "2.5th-97.5th percentile"},
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "hit_rate": tp / (tp + fn),
        "miss_rate": fn / (tp + fn),
        "dice_mean": summary["dice_distribution"]["mean"],
        "dice_median": summary["dice_distribution"]["median"],
        "iou_mean": summary["iou_distribution"]["mean"],
        "iou_median": summary["iou_distribution"]["median"],
        "n_gt_instances": summary["n_gt"],
        "n_pred_instance_views": summary["n_pred"],
        "n_pred_instances": int(n_pred_obs.sum()),
        "pred_instances_per_observation": _quartiles(n_pred_obs),
        "pred_area_px": _quartiles(pred_area),
        "matched_iou": summary["iou_distribution"],
        "matched_dice": summary["dice_distribution"],
        "one_to_many": summary["one_to_many"],
        "many_to_one": summary["many_to_one"],
        "pq_per_view_mean": summary["pq_per_image_mean"],
        "pq_per_view_distribution": summary["pq_per_image_distribution"],
        "pq_per_observation": _quartiles(obs_pq),
        "by_annotator_group": by_group,
        "pq_range_over_annotator_groups": [min(group_pqs), max(group_pqs)],
        "detections_scored": dict(sorted(fates.items())),
        "fusion_runtime_seconds_per_observation": {
            **_quartiles(fuse_seconds),
            "note": "wall time of the fusion step alone (fuse() and RLE encoding, from cached "
                    "logits and detections) on this Mac's CPU, measured while the other "
                    "workers ran in parallel; U-Net and detector inference ran on Kaggle GPUs "
                    "and are not timed here",
        },
        "definitions": {
            "pq_sq_rq": "pooled over all 248 views: PQ = sum IoU / (TP + FP/2 + FN/2), "
                        "SQ = sum IoU / TP, RQ = TP / (TP + FP/2 + FN/2)",
            "precision_recall": "pooled over views: precision = TP / (TP + FP), recall = hit "
                                "rate = TP / (TP + FN), miss rate = FN / (TP + FN)",
            "ci95": "95% bootstrap interval, 2000 resamples of the 144 observations (seed 0)",
            "by_annotator_group_ci": "pq_ci95_views resamples the group's views; "
                                     "pq_ci95_observations resamples its observations with "
                                     "their views of that group kept together",
            "n_pred_instance_views": "predicted instances counted once per view they are scored "
                                     "against (summarize's n_pred)",
            "one_to_many": "annotated filaments covered >10% each by >= 2 predictions, per view",
            "many_to_one": "predictions covering >10% each of >= 2 annotated filaments, per view",
            "pq_per_view_mean": "mean of per-view PQ (summarize's pq_per_image_mean)",
            "pq_per_observation": "PQ pooled over each observation's views, then summarised",
            "by_annotator_group": "views grouped by the first two digits of the view id",
            "detections_scored": "fate of every merged detection on validation: kept, dropped "
                                 "(claim < 200 px), below_score (< 0.30)",
        },
    }


def consistency_check(per_stem: list[dict]) -> dict:
    """V's totals must equal the half of its cross-fitted run scored with V's setting.

    scripts/fuse_detections.py scored the tuning half with the setting chosen
    on the held-out half, which for this run is V's own; those rows are the
    first 72 of the saved totals.
    """
    saved = np.load(STEP3 / V_TOTALS)
    mine = {e["stem"]: observation_totals(e) for e in per_stem}
    n_half = len(saved["stems"]) - len(saved["holdout_stems"])
    rows = [str(s) for s in saved["stems"][:n_half]]
    diff = max(float(np.abs(mine[s] - saved["crossfit"][i]).max()) for i, s in enumerate(rows))
    return {"reference": f"outputs/step3/{V_TOTALS}, crossfit rows of the tuning half",
            "n_observations_compared": n_half, "max_abs_difference": diff,
            "matches": diff < 1e-6}


def test_numbers(per_stem: list[dict]) -> dict:
    predictions = read_submission(SUBMISSION_TEST)
    test_stems = sorted(p.stem for p in test_image_paths(TEST_IMAGE_DIR))
    unknown = sorted(set(predictions) - set(test_stems))
    if unknown:
        raise SystemExit(f"submission has rows for {len(unknown)} non-test images, "
                         f"e.g. {unknown[0]}")
    counts = np.array([len(predictions.get(s, [])) for s in test_stems])
    areas = rle_areas([r for s in test_stems for r in predictions.get(s, [])])
    return {
        "submission": str(SUBMISSION_TEST.relative_to(REPO)),
        "n_images": len(test_stems),
        "n_images_with_predictions": int((counts > 0).sum()),
        "n_instances": int(counts.sum()),
        "instances_per_image": _quartiles(counts),
        "pred_area_px": _quartiles(areas),
        "validation_comparison": {
            "instances_per_observation_mean": float(np.mean([e["n_pred"] for e in per_stem])),
            "pred_area_px_median": float(np.median(np.concatenate(
                [e["pred_area"] for e in per_stem]))),
        },
    }


def progression_intervals() -> dict:
    """PQ with 95% intervals for the progression table's pipelines.

    Each on all 144 observations (cross-fitted for fusion; the U-Net-only
    baseline has one fixed setting) and on the 72 held-out from tuning.
    """
    out = {}
    for key, label, name, array in PROGRESSION:
        saved = np.load(STEP3 / name)
        stems = [str(s) for s in saved["stems"]]
        held_stems = [str(s) for s in saved["holdout_stems"]]
        rows = saved[array]
        if array == "baseline":
            held = rows[[stems.index(s) for s in held_stems]]
        else:
            held = saved["holdout"]
        out[key] = {"label": label, "file": f"outputs/step3/{name}", "array": array,
                    "all_144": pooled_summary(rows), "held_out_72": pooled_summary(held)}
    return {"interval": f"95% bootstrap percentile interval, {N_BOOT} resamples of observations "
                        f"with replacement, seed {BOOT_SEED}",
            "note": "all_144 is cross-fitted for fusion rows (each half scored with the setting "
                    "the other half chose); the U-Net-only baseline uses one fixed setting (thr "
                    "0.6, area 400, gap 24, close 3, open 0). held_out_72 is the half never used "
                    "for tuning.",
            "pipelines": out}


def crossfit_settings(per_setting: np.ndarray, n_tune: int) -> tuple[np.ndarray, int, int]:
    """scripts/fuse_detections.py's cross-fitting.

    ``per_setting`` is (observations, settings, 6), tuning half first. Each
    half is scored with the setting ranked first on the other half; returns
    those per-observation totals and both chosen settings.
    """
    half_a = np.arange(n_tune)
    half_b = np.arange(n_tune, per_setting.shape[0])

    def ranked(rows: np.ndarray) -> list[int]:
        totals = per_setting[rows].sum(axis=0)
        return sorted(range(per_setting.shape[1]), key=lambda i: -float(pq_of(totals[i])))

    best_a, best_b = ranked(half_a)[0], ranked(half_b)[0]
    crossfit = np.empty((per_setting.shape[0], per_setting.shape[2]))
    crossfit[half_b], crossfit[half_a] = per_setting[half_b, best_a], per_setting[half_a, best_b]
    return crossfit, best_a, best_b


def detectors_only_numbers(per_stem: list[dict], inputs: Inputs) -> dict:
    """The validated entry's detectors without the U-Net, cross-fitted like V."""
    order = inputs.tune + inputs.holdout
    saved = np.load(STEP3 / V_TOTALS)
    if [str(s) for s in saved["stems"]] != order:
        raise SystemExit(f"{V_TOTALS} was scored on another split; cannot pair with it")
    by_stem = {e["stem"]: e["det_only"] for e in per_stem}
    per_setting = np.stack([by_stem[s] for s in order])
    crossfit, best_a, best_b = crossfit_settings(per_setting, len(inputs.tune))
    held = per_setting[len(inputs.tune):, best_a]
    all_totals = per_setting.sum(axis=0)
    best_all = max(range(len(DET_ONLY_GRID)), key=lambda i: (float(pq_of(all_totals[i])), -i))

    def setting(i: int) -> dict:
        return {"min_score": DET_ONLY_GRID[i][0], "min_area": DET_ONLY_GRID[i][1]}

    def paired(a: np.ndarray, b: np.ndarray) -> dict:
        delta, low, high = paired_bootstrap(a, b, n_boot=N_BOOT_PAIRED)
        wins = float(np.mean(bootstrap_deltas(a, b, n_boot=N_BOOT_PAIRED) > 0))
        return {"delta": delta, "ci95": [low, high], "p_v_better": wins,
                "v_pq": float(pq_of(a.sum(axis=0))), "detectors_only_pq": float(pq_of(b.sum(axis=0)))}

    return {
        "description": "no U-Net: every merged detection of detections_val_det124L.json, in "
                       "descending score, claims the not yet claimed pixels of its own "
                       "full-resolution mask on the disk (0.98 R); claims under min_area are "
                       "dropped; scored on every view, pooled",
        "grid": {"min_score": list(DET_ONLY_SCORES), "min_area": list(DET_ONLY_AREAS)},
        "split": "fuse_detections.py's: sample_stems(seed 0, 72) tunes, the other 72 are held "
                 "out; cross-fitting scores each half with the setting the other half chose",
        "crossfit_144": pooled_summary(crossfit),
        "held_out_72": pooled_summary(held),
        "settings": {
            "tuned_on_tuning_half_applied_to_held_out": setting(best_a),
            "tuned_on_held_out_half_applied_to_tuning_half": setting(best_b),
            "best_on_all_144_in_sample": {**setting(best_all),
                                          "pq": float(pq_of(all_totals[best_all]))},
        },
        "v_minus_detectors_only": {
            "method": f"paired bootstrap over observations, {N_BOOT_PAIRED} resamples, seed 0 "
                      "(filament_seg.scoring.paired_bootstrap)",
            "crossfit_144": paired(saved["crossfit"], crossfit),
            "held_out_72": paired(saved["holdout"], held),
        },
    }


# --- Plotting helpers ---------------------------------------------------------------


def _pyplot():
    """pyplot with the report style; imported lazily to keep workers light."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)
    return plt


def _save(fig, name: str) -> str:
    """Save at exactly the figure's size, refusing any text under MIN_TEXT pt."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    # Tick labels get their text only when the figure is laid out.
    fig.draw_without_rendering()
    small = sorted({(round(t.get_fontsize(), 1), t.get_text()[:30])
                    for t in fig.findobj(Text)
                    if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_TEXT})
    if small:
        raise RuntimeError(f"{name}: text below {MIN_TEXT} pt: {small[:5]}")
    path = FIG_DIR / name
    # No bbox_inches="tight": the page width must stay exactly the column width.
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


# --- Image panels: crops, outlines, outcomes -------------------------------------------

#: Crops must lie wholly inside the pipeline's disk mask (98% of the radius),
#: except the limb example, which must cross the limb.
DISK_MASK_FRACTION = 0.98
#: Display range of the corrected image, in units of the quiet Sun (1.0):
#: symmetric, so the quiet Sun is mid-grey and filaments (0.5-0.8) are dark.
DISPLAY_RANGE = (0.65, 1.35)
#: The flat cache stores clip(flat, 0, 2) * 127.5 (scripts/preprocess.py).
FLAT_SCALE = 127.5


#: A limb crop shows sky beyond the limb over this share of its area, so the
#: disk edge is plainly visible while most of the crop stays on the disk.
LIMB_SKY_SHARE = (0.10, 0.40)


def _sky_share(disk: Disk, x0: np.ndarray, y0: np.ndarray, size: int,
               samples: int = 24) -> np.ndarray:
    """Share of each crop beyond the fitted limb, from a grid of sample points."""
    offsets = (np.arange(samples) + 0.5) * size / samples
    xs = x0[:, None, None] + offsets[None, None, :]
    ys = y0[:, None, None] + offsets[None, :, None]
    return (np.hypot(xs - disk.cx, ys - disk.cy) > disk.radius).mean(axis=(1, 2))


def _crop_grid(disk: Disk, size: int, step: int, where: str):
    """Top-left corners of candidate crops, and each crop centre's distance
    from the disk centre. ``where="disk"`` keeps crops wholly on the disk
    mask; ``"limb"`` keeps crops with LIMB_SKY_SHARE of their area beyond
    the limb."""
    grid = np.arange(0, IMAGE_SIZE[0] - size + 1, step)
    x0, y0 = (a.ravel() for a in np.meshgrid(grid, grid))
    centre = np.hypot(x0 + size / 2 - disk.cx, y0 + size / 2 - disk.cy)
    if where == "disk":
        far = np.max([np.hypot(x0 + dx - disk.cx, y0 + dy - disk.cy)
                      for dx in (0, size - 1) for dy in (0, size - 1)], axis=0)
        keep = far <= DISK_MASK_FRACTION * disk.radius
    else:
        sky = _sky_share(disk, x0, y0, size)
        keep = (sky >= LIMB_SKY_SHARE[0]) & (sky <= LIMB_SKY_SHARE[1])
    return x0[keep], y0[keep], centre[keep]


def _inside(boxes: list[list[float]], x0: np.ndarray, y0: np.ndarray, size: int) -> np.ndarray:
    """``(n_windows, n_boxes)``: is each box wholly inside each window?"""
    if not boxes:
        return np.zeros((x0.size, 0), dtype=bool)
    b = np.asarray(boxes)
    return ((b[None, :, 0] >= x0[:, None]) & (b[None, :, 1] >= y0[:, None])
            & (b[None, :, 0] + b[None, :, 2] <= x0[:, None] + size)
            & (b[None, :, 1] + b[None, :, 3] <= y0[:, None] + size))


def _show_crop(ax, stem: str, disk: Disk, x0: int, y0: int, size: int) -> None:
    """The corrected image, cropped; beyond the fitted limb (sky) shown black."""
    flat = cv2.imread(str(CACHE_DIR / "flat" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    image = flat[y0:y0 + size, x0:x0 + size].astype(np.float64) / FLAT_SCALE
    low, high = DISPLAY_RANGE
    shown = np.clip((image - low) / (high - low), 0.0, 1.0)
    ys, xs = np.ogrid[y0:y0 + size, x0:x0 + size]
    shown[np.hypot(xs - disk.cx, ys - disk.cy) > disk.radius] = 0.0
    # interpolation="none" embeds the crop at its native resolution.
    ax.imshow(shown, cmap="gray", vmin=0, vmax=1, interpolation="none")
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def _contours(rle: Rle, x0: int, y0: int, size: int, margin: int = 6) -> list[np.ndarray]:
    """Outer outlines of a mask near the crop, in crop pixel coordinates."""
    ys0, xs0 = max(y0 - margin, 0), max(x0 - margin, 0)
    window = rle_to_mask(rle)[ys0:y0 + size + margin, xs0:x0 + size + margin]
    if not window.any():
        return []
    found, _ = cv2.findContours(window.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :].astype(np.float64) + (xs0 - x0, ys0 - y0) for c in found if len(c) >= 3]


def _draw(ax, outlines: list[np.ndarray], color: str, fill_alpha: float = 0.0,
          linestyle="-", linewidth: float = 0.8, halo_alpha: float = 0.5) -> None:
    """Fill and outline polygons; a thin dark halo keeps outlines visible on any grey."""
    from matplotlib import patheffects
    from matplotlib.patches import Polygon

    halo = [patheffects.Stroke(linewidth=linewidth + 0.8, foreground="black", alpha=halo_alpha),
            patheffects.Normal()] if halo_alpha > 0 else None
    for points in outlines:
        if fill_alpha > 0:
            ax.add_patch(Polygon(points, closed=True, facecolor=color, alpha=fill_alpha,
                                 edgecolor="none", linewidth=0))
        ax.add_patch(Polygon(points, closed=True, fill=False, edgecolor=color,
                             linewidth=linewidth, linestyle=linestyle, path_effects=halo))


def _draw_outcomes(ax, outcome: dict, pred_outlines: list, gt_outlines: list) -> dict:
    """TP/FP/FN against one annotator, as in the qualitative figure's panel (d).

    Returns how many of each are at least partly in the crop.
    """
    for g in outcome["fn"]:
        _draw(ax, gt_outlines[g], OUTCOME_ON_IMAGE["fn"], fill_alpha=0.45)
    for p in outcome["fp"]:
        _draw(ax, pred_outlines[p], OUTCOME_ON_IMAGE["fp"], fill_alpha=0.45)
    for _, p in outcome["tp"]:
        _draw(ax, pred_outlines[p], OUTCOME_ON_IMAGE["tp"], fill_alpha=0.45)
    return {"tp": sum(1 for _, p in outcome["tp"] if pred_outlines[p]),
            "fp": sum(1 for p in outcome["fp"] if pred_outlines[p]),
            "fn": sum(1 for g in outcome["fn"] if gt_outlines[g])}


def _outcome_handles() -> list:
    from matplotlib.patches import Patch

    return [Patch(facecolor=OUTCOME_ON_IMAGE[k], edgecolor=OUTCOME_ON_IMAGE[k], alpha=0.8,
                  label=OUTCOME_LABELS[k]) for k in ("tp", "fp", "fn")]


def _scale_bar(ax, size: int, bar: int = 100) -> None:
    from matplotlib import patheffects

    shadow = [patheffects.withStroke(linewidth=1.6, foreground="black", alpha=0.6)]
    ax.plot([30, 30 + bar], [size - 34, size - 34], color="white", linewidth=1.6,
            solid_capstyle="butt", path_effects=shadow)
    ax.text(30 + bar / 2, size - 44, f"{bar} px", color="white", ha="center", va="bottom",
            fontsize=MIN_TEXT, path_effects=shadow)


def _observation_summary(entry: dict) -> dict:
    totals = observation_totals(entry)
    pooled = _pq_sq_rq(*(totals[COUNT_FIELDS.index(k)] for k in ("tp", "fp", "fn", "iou_sum")))
    return {"pq": pooled["pq"], "sq": pooled["sq"], "rq": pooled["rq"],
            **{k: int(totals[COUNT_FIELDS.index(k)]) for k in ("n_gt", "n_pred", "tp", "fp", "fn")},
            "iou_sum": float(totals[COUNT_FIELDS.index("iou_sum")]),
            "n_pred_instances": entry["n_pred"], "n_views": len(entry["views"])}


# --- Figure 1: qualitative example ---------------------------------------------------

QUAL_SIZE = 700
QUAL_STEP = 10
#: "Roughly median": observations whose PQ is within this of the median of all 144.
QUAL_PQ_BAND = 0.02
#: A crop must hold at least this many matched filaments (and one FP and one
#: FN) of the first annotator wholly inside, to illustrate every outcome.
QUAL_MIN_TP = 2


def best_crop(entry: dict, disk: Disk) -> dict | None:
    """The on-disk crop with the most outcomes against the first annotator wholly
    inside it (matched filaments, FPs and FNs), if it shows all three."""
    first = entry["views"][0]
    x0, y0, centre = _crop_grid(disk, QUAL_SIZE, QUAL_STEP, "disk")
    gt_in = _inside(entry["boxes"]["gt"][first], x0, y0, QUAL_SIZE)
    pred_in = _inside(entry["boxes"]["pred"], x0, y0, QUAL_SIZE)
    outcome = entry["outcome"][first]

    def count(inside: np.ndarray, members: list[int]) -> np.ndarray:
        return inside[:, members].sum(axis=1) if members else np.zeros(len(x0), dtype=int)

    tp = count(gt_in, [g for g, _ in outcome["tp"]])
    fn = count(gt_in, outcome["fn"])
    fp = count(pred_in, outcome["fp"])
    ok = (tp >= QUAL_MIN_TP) & (fp >= 1) & (fn >= 1)
    if not ok.any():
        return None
    score = np.where(ok, tp + fp + fn, -1)
    # Most outcomes; among equals, the crop nearest the disk centre.
    best = int(np.lexsort((centre, -score))[0])
    return {"x0": int(x0[best]), "y0": int(y0[best]), "size": QUAL_SIZE,
            "outcomes_inside": int(score[best]), "tp_inside": int(tp[best]),
            "fp_inside": int(fp[best]), "fn_inside": int(fn[best])}


def choose_qualitative(per_stem: list[dict], disks: dict[str, Disk]) -> dict:
    """A multi-annotator observation with about the median per-observation PQ.

    Among observations with two or three annotators whose PQ is within
    ``QUAL_PQ_BAND`` of the median over all 144, the one whose best crop shows
    the most outcomes is chosen; ties go to the PQ nearer the median.
    """
    pq = {e["stem"]: float(pq_of(observation_totals(e))) for e in per_stem}
    median = float(np.median(list(pq.values())))
    band = sorted((e for e in per_stem if len(e["views"]) in (2, 3)
                   and abs(pq[e["stem"]] - median) <= QUAL_PQ_BAND),
                  key=lambda e: (abs(pq[e["stem"]] - median), e["stem"]))
    best = None
    for entry in band:
        crop = best_crop(entry, disks[entry["stem"]])
        if crop is not None and (best is None
                                 or crop["outcomes_inside"] > best[1]["outcomes_inside"]):
            best = (entry, crop)
    if best is None:
        raise SystemExit("no observation near the median PQ has a crop showing every outcome")
    entry, crop = best
    values = np.array(list(pq.values()))
    return {"entry": entry, "crop": crop, "median_observation_pq": median,
            "observation_pq": pq[entry["stem"]], "candidates_in_band": len(band),
            "pq_percentile": 100.0 * float(np.mean(values <= pq[entry["stem"]]))}


def fig_qualitative(choice: dict, inputs: Inputs) -> dict:
    plt = _pyplot()
    from matplotlib.lines import Line2D

    entry, crop = choice["entry"], choice["crop"]
    stem, views = entry["stem"], entry["views"]
    x0, y0, size = crop["x0"], crop["y0"], crop["size"]
    preds, _, _ = fuse_observation(stem, inputs.detections[stem], inputs.disks[stem])
    pred_outlines = [_contours(r, x0, y0, size) for r in preds]
    gt_outlines = {v: [_contours(r, x0, y0, size) for r in inputs.gt[v]] for v in views}

    # Four square panels in a row, legends underneath.
    gap, side_margin, legend_h = 0.06, 0.02, 0.42
    side = (DOUBLE_COLUMN - 2 * side_margin - 3 * gap) / 4
    height = side + legend_h + 0.03
    fig = plt.figure(figsize=(DOUBLE_COLUMN, height))
    axes = []
    for i in range(4):
        left = side_margin + i * (side + gap)
        ax = fig.add_axes((left / DOUBLE_COLUMN, (height - 0.03 - side) / height,
                           side / DOUBLE_COLUMN, side / height))
        _show_crop(ax, stem, inputs.disks[stem], x0, y0, size)
        ax.text(0.025, 0.975, f"({'abcd'[i]})", transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none",
                                  "alpha": 0.85})
        axes.append(ax)
    legend = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.02), "frameon": False,
              "handlelength": 1.8, "handletextpad": 0.5, "borderaxespad": 0.0,
              "labelspacing": 0.25, "fontsize": MIN_TEXT}

    # (a) the image, with a scale bar.
    _scale_bar(axes[0], size)
    axes[0].text(0.5, -0.03, "Hα, limb darkening removed", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=MIN_TEXT)

    # (b) every annotator's filaments. Annotators 2 and 3 are dashed out of
    # phase, so where all three agree the outline alternates their colours.
    annotator_style = [(OI["sky"], "-"), (OI["yellow"], (0, (2.4, 2.4))),
                       (OI["purple"], (2.4, (2.4, 2.4)))]
    handles = []
    for k, view in enumerate(views):
        color, style = annotator_style[k]
        for outlines in gt_outlines[view]:
            _draw(axes[1], outlines, color, linestyle=style, linewidth=0.9,
                  halo_alpha=0.45 if k == 0 else 0.0)
        handles.append(Line2D([], [], color=color, linestyle=style, linewidth=1.2,
                              label=f"Annotator {k + 1} ({view.split('-')[0]})"))
    axes[1].legend(handles=handles, **legend)

    # (c) V's instances, one colour each; colour only tells neighbours apart,
    # and none of the outcome colours of (d) is used.
    visible = [outlines for outlines in pred_outlines if outlines]
    for k, outlines in enumerate(visible):
        _draw(axes[2], outlines, INSTANCE_COLORS[k % len(INSTANCE_COLORS)], fill_alpha=0.45)
    in_crop = len(visible)
    axes[2].text(0.5, -0.03, f"Predicted instances, one colour each ({in_crop} in crop)",
                 transform=axes[2].transAxes, ha="center", va="top", fontsize=MIN_TEXT)

    # (d) outcome against the first annotator.
    shown_counts = _draw_outcomes(axes[3], entry["outcome"][views[0]], pred_outlines,
                                  gt_outlines[views[0]])
    axes[3].legend(handles=_outcome_handles(), **legend)
    path = _save(fig, "fig_qualitative.pdf")

    per_view = []
    for view, r in zip(views, entry["results"]):
        per_view.append({"view": view, "annotator_prefix": view.split("-")[0], "n_gt": r.n_gt,
                         "n_pred": r.n_pred, "tp": r.tp, "fp": r.fp, "fn": r.fn, "pq": r.pq,
                         "sq": r.iou_sum / r.tp if r.tp else 0.0,
                         "one_to_many": r.one_to_many, "many_to_one": r.many_to_one})
    return {
        "file": path,
        "stem": stem,
        "views": views,
        "selection": {
            "rule": f"observations with 2-3 annotators and PQ within {QUAL_PQ_BAND} of the "
                    f"median per-observation PQ of all 144; for each, the {QUAL_SIZE} px crop "
                    f"wholly inside {DISK_MASK_FRACTION} R of the disk with the most outcomes "
                    "against annotator 1 (matched filaments, FPs, FNs) wholly inside, needing "
                    f">= {QUAL_MIN_TP} TP, >= 1 FP and >= 1 FN; the observation whose crop "
                    "shows the most outcomes is chosen",
            "median_observation_pq": choice["median_observation_pq"],
            "candidates_in_band": choice["candidates_in_band"],
            "observation_pq_percentile": choice["pq_percentile"],
        },
        "crop": {**crop, "display_range_quiet_sun_units": list(DISPLAY_RANGE),
                 "scale_bar_px": 100, "embedded_ppi": round(size / side)},
        "observation": _observation_summary(entry),
        "per_view": per_view,
        "panel_c_instances_in_crop": in_crop,
        "panel_c_instance_colors": INSTANCE_COLORS,
        "panel_d_in_crop_vs_annotator_1": shown_counts,
    }


# --- Figure: three more examples (good, failure, limb) ---------------------------------

EXAMPLE_SIZE = 800
EXAMPLE_STEP = 20
#: Good and failure crops hold at least this many of the first annotator's
#: filaments wholly inside, so neither is an empty frame.
EXAMPLE_MIN_GT = 5
#: A filament is near the limb when its box centre lies beyond this fraction
#: of the radius; the limb crop must hold at least LIMB_MIN_NEAR of them.
LIMB_NEAR = 0.85
LIMB_MIN_NEAR = 2


def _densest_crop(entry: dict, disk: Disk, where: str) -> dict | None:
    """The crop holding the most of the first annotator's filaments wholly
    inside it (for the limb, the most near-limb ones first)."""
    boxes = entry["boxes"]["gt"][entry["views"][0]]
    x0, y0, centre = _crop_grid(disk, EXAMPLE_SIZE, EXAMPLE_STEP, where)
    if not boxes or not len(x0):
        return None
    inside = _inside(boxes, x0, y0, EXAMPLE_SIZE)
    n = inside.sum(axis=1)
    crop: dict = {}
    if where == "limb":
        b = np.asarray(boxes)
        near = (np.hypot(b[:, 0] + b[:, 2] / 2 - disk.cx, b[:, 1] + b[:, 3] / 2 - disk.cy)
                >= LIMB_NEAR * disk.radius)
        n_near = (inside & near[None, :]).sum(axis=1)
        k = int(np.lexsort((centre, -n, -n_near))[0])
        crop["near_limb_inside"] = int(n_near[k])
    else:
        k = int(np.lexsort((centre, -n))[0])
    return {"x0": int(x0[k]), "y0": int(y0[k]), "size": EXAMPLE_SIZE,
            "annotator_1_inside": int(n[k]), **crop}


def choose_examples(per_stem: list[dict], disks: dict[str, Disk], exclude: set[str]) -> list:
    """(i) a top-quartile observation with >= 2 annotators, (ii) a bottom-quartile
    one, each with a well-populated on-disk crop, and (iii) a crop across the
    limb; three different observations, none of them the qualitative figure's."""
    pq = {e["stem"]: float(pq_of(observation_totals(e))) for e in per_stem}
    q25, q75 = (float(q) for q in np.percentile(list(pq.values()), [25, 75]))
    used = set(exclude)
    picks = []

    def pick(case: str, rule: str, candidates, where: str, accept, key) -> None:
        best = None
        for entry in candidates:
            if entry["stem"] in used:
                continue
            crop = _densest_crop(entry, disks[entry["stem"]], where)
            if crop is None or not accept(crop):
                continue
            score = key(entry, crop)
            if best is None or score > best[0]:
                best = (score, entry, crop)
        if best is None:
            raise SystemExit(f"no observation qualifies for the {case} example")
        used.add(best[1]["stem"])
        picks.append({"case": case, "rule": rule, "entry": best[1], "crop": best[2],
                      "pq": pq[best[1]["stem"]], "where": where})

    pick("good", f"per-observation PQ >= the 75th percentile ({q75:.3f}), >= 2 annotators, an "
                 f"{EXAMPLE_SIZE} px crop on the disk with >= {EXAMPLE_MIN_GT} of annotator 1's "
                 "filaments wholly inside; the most such filaments, then the higher PQ",
         [e for e in per_stem if pq[e["stem"]] >= q75 and len(e["views"]) >= 2], "disk",
         lambda c: c["annotator_1_inside"] >= EXAMPLE_MIN_GT,
         lambda e, c: (c["annotator_1_inside"], pq[e["stem"]]))
    pick("failure", f"per-observation PQ <= the 25th percentile ({q25:.3f}), an {EXAMPLE_SIZE} "
                    f"px crop on the disk with >= {EXAMPLE_MIN_GT} of annotator 1's filaments "
                    "wholly inside; the most such filaments, then the lower PQ",
         [e for e in per_stem if pq[e["stem"]] <= q25], "disk",
         lambda c: c["annotator_1_inside"] >= EXAMPLE_MIN_GT,
         lambda e, c: (c["annotator_1_inside"], -pq[e["stem"]]))
    pick("limb", f"an {EXAMPLE_SIZE} px crop with {LIMB_SKY_SHARE[0]:.0%}-{LIMB_SKY_SHARE[1]:.0%} "
                 f"of its area beyond the limb, holding >= {LIMB_MIN_NEAR} of annotator 1's "
                 f"filaments wholly inside with box centres beyond {LIMB_NEAR} R; the most "
                 "near-limb filaments, then the most filaments, any PQ",
         per_stem, "limb", lambda c: c["near_limb_inside"] >= LIMB_MIN_NEAR,
         lambda e, c: (c["near_limb_inside"], c["annotator_1_inside"]))
    for p in picks:
        p["quartiles"] = {"p25": q25, "p75": q75}
    return picks


def fig_examples(picks: list[dict], inputs: Inputs) -> dict:
    plt = _pyplot()
    from matplotlib import patheffects
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    gap, side_margin, header_h, legend_h = 0.10, 0.02, 0.20, 0.26
    side = (DOUBLE_COLUMN - 2 * side_margin - 2 * gap) / 3
    height = header_h + side + legend_h
    fig = plt.figure(figsize=(DOUBLE_COLUMN, height))
    names = {"good": "Top quartile", "failure": "Bottom quartile", "limb": "At the limb"}
    edge_style = {"color": "#DDDDDD", "linestyle": (0, (3, 2)), "linewidth": 0.9,
                  "path_effects": [patheffects.Stroke(linewidth=1.8, foreground="black",
                                                      alpha=0.6), patheffects.Normal()]}
    panels = []
    for i, p in enumerate(picks):
        entry, crop = p["entry"], p["crop"]
        stem, disk = entry["stem"], inputs.disks[entry["stem"]]
        x0, y0, size = crop["x0"], crop["y0"], crop["size"]
        left = side_margin + i * (side + gap)
        ax = fig.add_axes((left / DOUBLE_COLUMN, legend_h / height, side / DOUBLE_COLUMN,
                           side / height))
        _show_crop(ax, stem, disk, x0, y0, size)
        preds, _, _ = fuse_observation(stem, inputs.detections[stem], disk)
        first = entry["views"][0]
        counts = _draw_outcomes(ax, entry["outcome"][first],
                                [_contours(r, x0, y0, size) for r in preds],
                                [_contours(r, x0, y0, size) for r in inputs.gt[first]])
        # Where the pipeline's disk mask ends: nothing beyond it can be predicted.
        ax.add_patch(Circle((disk.cx - x0, disk.cy - y0), DISK_MASK_FRACTION * disk.radius,
                            fill=False, **edge_style))
        ax.text(0.0, 1.015, f"({'i' * (i + 1) if i < 3 else i + 1}) {names[p['case']]} · {stem} "
                            f"· PQ {p['pq']:.2f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=MIN_TEXT)
        if i == 0:
            _scale_bar(ax, size)
        ys, xs = np.ogrid[y0:y0 + size, x0:x0 + size]
        off_disk = float(np.mean(np.hypot(xs - disk.cx, ys - disk.cy) > disk.radius))
        panels.append({
            "panel": "i" * (i + 1), "case": p["case"], "rule": p["rule"], "stem": stem,
            "views": entry["views"], "observation": _observation_summary(entry),
            "crop": {**crop, "fraction_beyond_limb": off_disk},
            "in_crop_vs_annotator_1": counts,
        })
    fig.legend(handles=_outcome_handles() + [Line2D([], [], label="Disk mask edge (0.98 R)",
                                                    **edge_style)],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
               fontsize=MIN_TEXT, handlelength=1.8, columnspacing=1.6)
    path = _save(fig, "fig_examples.pdf")
    return {"file": path, "crop_size_px": EXAMPLE_SIZE,
            "embedded_ppi": round(EXAMPLE_SIZE / side),
            "quartiles_of_per_observation_pq": picks[0]["quartiles"],
            "display": f"corrected image, {DISPLAY_RANGE[0]}-{DISPLAY_RANGE[1]} quiet-Sun units "
                       "to black-white; beyond the fitted limb shown black; outcomes against "
                       "each observation's first annotator as in fig_qualitative (d)",
            "panels": panels}


# --- Figure 2: IoU and Dice ------------------------------------------------------------


def fig_iou_dice(per_stem: list[dict]) -> dict:
    plt = _pyplot()
    best = np.array([v for e in per_stem for v in e["gt_best_iou"]])
    ious = np.array([i for e in per_stem for r in e["results"] for i in r.matched_ious])
    dices = 2.0 * ious / (1.0 + ious)
    regions = {"missed": best <= 0.25, "near_miss": (best > 0.25) & (best <= 0.5),
               "matched": best > 0.5}
    if int(regions["matched"].sum()) != ious.size:
        raise RuntimeError("filaments with best IoU > 0.5 are not exactly the matched ones")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, 2.25), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.7, 1]})
    # (a) every annotated filament view, by its best IoU with any prediction.
    # Bins are right-closed, (a, b], like the categories: an IoU of exactly
    # 0.5 is a near miss, since matching needs more than 0.5.
    bins_a = np.linspace(0.0, 1.0, 21)
    index = np.clip(np.searchsorted(bins_a, best, side="left") - 1, 0, len(bins_a) - 2)
    counts_a = np.bincount(index, minlength=len(bins_a) - 1)
    centres = (bins_a[:-1] + bins_a[1:]) / 2
    matched_bin = centres > 0.5
    ax_a.bar(centres, counts_a, width=bins_a[1] - bins_a[0],
             color=[OUTCOME_ON_WHITE["tp"] if m else OUTCOME_ON_WHITE["fn"] for m in matched_bin],
             edgecolor=[OUTCOME_ON_WHITE["tp"] if m else FN_EDGE_ON_WHITE for m in matched_bin],
             linewidth=0.5)
    ax_a.axvline(0.5, color=INK, linestyle=(0, (3, 2)), linewidth=0.8)
    ax_a.axvline(0.25, color=INK, linestyle=(0, (1, 1.5)), linewidth=0.7)
    top = counts_a.max() * 1.62
    ax_a.set_ylim(0, top)
    # Labels sit over their regions, the two narrow ones nudged apart.
    for key, name, x in (("missed", "Missed\n≤ 0.25", 0.112),
                         ("near_miss", "Near miss\n0.25–0.5", 0.39),
                         ("matched", "Matched\n> 0.5", 0.75)):
        n = int(regions[key].sum())
        ax_a.text(x, top * 0.99, f"{name}\n{n}\n({n / best.size:.0%})",
                  ha="center", va="top", fontsize=MIN_TEXT, linespacing=1.05,
                  bbox={"boxstyle": "square,pad=0.04", "fc": "white", "ec": "none"})
    zero = int((best == 0).sum())
    ax_a.text(0.065, counts_a[0] * 0.86, f"{zero} with\nIoU = 0", ha="left", va="center",
              fontsize=MIN_TEXT, linespacing=1.0)
    ax_a.set_xlim(0, 1)
    ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax_a.set_xlabel("(a) Best IoU of an annotated\nfilament with any prediction",
                    linespacing=1.1)
    ax_a.set_ylabel("Annotated filaments (view-level)")

    # (b) Dice of the matched pairs.
    bins_b = np.arange(0.66, 0.9601, 0.02)
    counts_b, _ = np.histogram(dices, bins_b)
    ax_b.hist(dices, bins_b, color=OUTCOME_ON_WHITE["tp"], edgecolor="white", linewidth=0.4)
    q = np.percentile(dices, [25, 50, 75])
    ax_b.axvline(q[1], color=INK, linestyle=(0, (3, 2)), linewidth=0.8)
    top_b = counts_b.max() * 1.22
    ax_b.set_ylim(0, top_b)
    ax_b.text(q[1] - 0.008, top_b * 0.99, f"median\n{q[1]:.2f}", ha="right", va="top",
              fontsize=MIN_TEXT, linespacing=1.1)
    ax_b.set_xlim(0.66, 0.96)
    ax_b.set_xticks([0.7, 0.8, 0.9])
    ax_b.set_xlabel("(b) Dice of the\nmatched pairs", linespacing=1.1)
    ax_b.set_ylabel("Matched pairs")
    path = _save(fig, "fig_iou_dice.pdf")

    # The matched pairs' IoU and Dice on one 0.5-1.0 grid, as earlier versions
    # of this figure reported them.
    pair_bins = np.linspace(0.5, 1.0, 26)

    def pair_stats(values: np.ndarray) -> dict:
        p25, p50, p75 = np.percentile(values, [25, 50, 75])
        return {"n": int(values.size), "mean": float(values.mean()), "p25": float(p25),
                "median": float(p50), "p75": float(p75), "min": float(values.min()),
                "max": float(values.max()),
                "histogram_counts": np.histogram(values, pair_bins)[0].tolist()}

    return {
        "file": path,
        "panel_a_best_match_iou": {
            "unit": "every annotated filament of every view (n_gt), with its best IoU against "
                    "any of V's predictions for that observation",
            "n": int(best.size),
            "missed_le_0.25": int(regions["missed"].sum()),
            "of_which_iou_0": zero,
            "near_miss_0.25_to_0.5": int(regions["near_miss"].sum()),
            "matched_gt_0.5": int(regions["matched"].sum()),
            "shares": {k: float(v.mean()) for k, v in regions.items()},
            "matched_equals_tp": True,
            "bin_edges": bins_a.tolist(), "histogram_counts": counts_a.tolist(),
            **{k: v for k, v in _quartiles(best).items() if k != "n"},
        },
        "panel_b_dice": {"n": int(dices.size), "mean": float(dices.mean()), "p25": float(q[0]),
                         "median": float(q[1]), "p75": float(q[2]),
                         "min": float(dices.min()), "max": float(dices.max()),
                         "bin_edges": bins_b.tolist(), "histogram_counts": counts_b.tolist()},
        "bin_edges": pair_bins.tolist(),
        "iou": pair_stats(ious),
        "dice": pair_stats(dices),
        "note": "pooled over all 248 views; Dice = 2 IoU / (1 + IoU); dashed lines: the 0.5 "
                "matching threshold in (a), the median in (b); iou and dice are the matched "
                "pairs on bin_edges",
    }


# --- Figure 3: causes of errors ----------------------------------------------------------


def fig_errors(per_stem: list[dict]) -> dict:
    plt = _pyplot()
    fn = Counter(cause for e in per_stem for cause, _ in e["fn"])
    fp = Counter(cause for e in per_stem for cause, _ in e["fp"])
    n_fn, n_fp = sum(fn.values()), sum(fp.values())
    fn_labels = {**{k: v[0] for k, v in FN_CAUSES.items()},
                 FN_KEPT_DISJOINT: "Detected, kept, disjoint"}
    groups = [(name, list(keys)) for name, keys in FN_GROUPS]
    if fn[FN_KEPT_DISJOINT]:
        groups[1][1].append(FN_KEPT_DISJOINT)
    # FN rows: a header per group, then its causes.
    fn_rows: list[tuple[str, str]] = []
    for name, keys in groups:
        fn_rows.append(("header", name))
        fn_rows.extend(("item", k) for k in keys)

    fig, (ax_fn, ax_fp) = plt.subplots(
        2, 1, figsize=(SINGLE_COLUMN, 2.9), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [len(fn_rows), len(FP_CAUSES)]})
    largest = max(max(fn.values()), max(fp.values()))
    bar = {"height": 0.68, "linewidth": 0.5}

    def value_labels(ax, ys, values, total) -> None:
        for yi, v in zip(ys, values):
            ax.text(v + largest * 0.012, yi, f"{v} ({v / total:.0%})", va="center", ha="left",
                    fontsize=MIN_TEXT)

    y_fn = np.arange(len(fn_rows))
    items = [(y, key) for y, (kind, key) in zip(y_fn, fn_rows) if kind == "item"]
    values = [fn[key] for _, key in items]
    ax_fn.barh([y for y, _ in items], values, color=OUTCOME_ON_WHITE["fn"],
               edgecolor=FN_EDGE_ON_WHITE, **bar)
    value_labels(ax_fn, [y for y, _ in items], values, n_fn)
    ax_fn.set_yticks(y_fn, [fn_labels[key] if kind == "item" else key for kind, key in fn_rows])
    for tick, label, (kind, _) in zip(ax_fn.yaxis.get_major_ticks(), ax_fn.get_yticklabels(),
                                      fn_rows):
        if kind == "header":
            label.set_fontstyle("italic")
            tick.tick1line.set_visible(False)
    # A thin divider above the second group's header.
    second = [y for y, (kind, _) in zip(y_fn, fn_rows) if kind == "header"][1]
    ax_fn.axhline(second - 0.5, color=GREY, linewidth=0.5)
    ax_fn.set_ylabel(f"False negatives\n(n = {n_fn})")

    y_fp = np.arange(len(FP_CAUSES))
    values = [fp[k] for k in FP_CAUSES]
    ax_fp.barh(y_fp, values, color=OUTCOME_ON_WHITE["fp"], edgecolor=OUTCOME_ON_WHITE["fp"],
               **bar)
    value_labels(ax_fp, y_fp, values, n_fp)
    ax_fp.set_yticks(y_fp, [v[0] for v in FP_CAUSES.values()])
    ax_fp.set_ylabel(f"False positives\n(n = {n_fp})")
    for ax in (ax_fn, ax_fp):
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
    ax_fp.set_xlim(0, largest * 1.3)
    ax_fp.set_xlabel("Instance-views (each annotator view counted)")
    path = _save(fig, "fig_errors.pdf")

    tp_area = np.array([a for e in per_stem for a in e["tp_area"]])
    fn_area = np.array([a for e in per_stem for _, a in e["fn"]])
    fp_area = np.array([a for e in per_stem for _, a in e["fp"]])
    elsewhere = [m for e in per_stem for m in e["fp_matched_elsewhere"]]
    n_fp_multi = len(elsewhere)

    def table(counts: Counter, keys, total: int, causes: dict) -> dict:
        return {k: {"label": causes[k][0] if k in causes else fn_labels[k],
                    "definition": causes[k][1] if k in causes else
                    "a detection with IoU > 0.5 and score >= 0.30 was kept, but its instance "
                    "does not touch the filament",
                    "count": counts[k], "share": counts[k] / total if total else 0.0}
                for k in keys}

    return {
        "file": path,
        "unit": "instance-views: an unmatched filament or prediction counted once per view",
        "fn_total": n_fn,
        "fp_total": n_fp,
        "fn_causes": table(fn, list(FN_CAUSES) + [FN_KEPT_DISJOINT], n_fn, FN_CAUSES),
        "fn_groups": {name: {"causes": keys, "count": sum(fn[k] for k in keys),
                             "share": sum(fn[k] for k in keys) / n_fn if n_fn else 0.0}
                      for name, keys in groups},
        "fp_causes": table(fp, list(FP_CAUSES), n_fp, FP_CAUSES),
        "fp_on_multi_view_observations": {
            "n_fp": n_fp_multi,
            "share_of_all_fp": n_fp_multi / n_fp if n_fp else 0.0,
            "matched_in_another_view": int(sum(elsewhere)),
            "share_matched_in_another_view": sum(elsewhere) / n_fp_multi if n_fp_multi else 0.0,
        },
        "gt_area_px": {"tp": _quartiles(tp_area), "fn": _quartiles(fn_area)},
        "fp_pred_area_px": _quartiles(fp_area),
    }


# --- Figure 4: cross-fitted PQ next to seed variation ---------------------------------


def fig_seeds() -> dict:
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    group_colors = [OI["blue"], OI["vermillion"], OI["green"]]
    # Rows top to bottom: ("header", text) or ("item", label, pq, [lo, hi], colour, kind).
    rows: list[tuple] = []
    missing, out_groups = [], []
    stems_ref = None
    for (group, members), color in zip(SEED_GROUPS, group_colors):
        rows.append(("header", group))
        out_members = []
        for label, name, kind in members:
            path = STEP3 / name
            if not path.exists():
                missing.append(name)
                continue
            saved = np.load(path)
            stems = [str(s) for s in saved["stems"]]
            if stems_ref is None:
                stems_ref = stems
            elif stems != stems_ref:
                raise SystemExit(f"{name} scored different observations")
            crossfit = float(pq_of(saved["crossfit"].sum(axis=0)))
            ci = bootstrap_ci(saved["crossfit"])
            rows.append(("item", label, crossfit, ci, color, kind))
            out_members.append({"label": label, "file": f"outputs/step3/{name}", "kind": kind,
                                "crossfit_pq": crossfit, "crossfit_pq_ci95": ci,
                                "holdout_pq": float(pq_of(saved["holdout"].sum(axis=0))),
                                "n_observations": len(stems)})
        out_groups.append({"group": group, "members": out_members})
    baseline = None
    if (STEP3 / BASELINE_TOTALS).exists():
        saved = np.load(STEP3 / BASELINE_TOTALS)
        baseline = {"pq": float(pq_of(saved["baseline"].sum(axis=0))),
                    "pq_ci95": bootstrap_ci(saved["baseline"])}
        rows.append(("header", "No detector"))
        rows.append(("item", "U-Net only, components", baseline["pq"], baseline["pq_ci95"],
                     GREY, "baseline"))
    else:
        missing.append(BASELINE_TOTALS)
    run3 = {name: float(pq_of(np.load(STEP3 / name)["crossfit"].sum(axis=0)))
            for name in (RUN3_WIDE, RUN3_NARROW) if (STEP3 / name).exists()}

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 3.55), layout="constrained")
    y = np.arange(len(rows), dtype=float)
    items = {r[1]: (yi, r) for yi, r in zip(y, rows) if r[0] == "item"}
    # Seed bands: from each recipe's first run to its retrain with another
    # seed, drawn across the rows of that group.
    seed_pairs = [("Run 3 (used in both entries)", "Run 3 recipe, seed 1"),
                  ("Det 2: YOLO11m, 1280 px", "Det 5: det 2 recipe, seed 1")]
    seed_out = []
    for a, b in seed_pairs:
        if a in items and b in items:
            (_, ra), (_, rb) = items[a], items[b]
            group_rows = [yi for yi, r in zip(y, rows) if r[0] == "item" and r[4] == ra[4]]
            ax.fill_betweenx([min(group_rows) - 0.45, max(group_rows) + 0.45],
                             min(ra[2], rb[2]), max(ra[2], rb[2]), color=ra[4], alpha=0.16,
                             linewidth=0, zorder=0)
            seed_out.append({"original": a, "retrain": b, "original_pq": ra[2],
                             "retrain_pq": rb[2], "delta": rb[2] - ra[2]})
    bar_style = {"fmt": "none", "elinewidth": 0.8, "capsize": 1.8, "capthick": 0.8}
    for yi, row in zip(y, rows):
        if row[0] != "item":
            continue
        _, label, value, (low, high), color, kind = row
        ax.errorbar([value], [yi], xerr=[[value - low], [high - value]], ecolor=color, zorder=2,
                    **bar_style)
        marker, size = ("D", 5.0) if kind == "final" else ("o", 4.4)
        ax.plot([value], [yi], marker=marker, markersize=size, color=color,
                markerfacecolor="white" if kind == "seed" else color, markeredgewidth=1.0,
                markeredgecolor=INK if kind == "final" else color, linestyle="none", zorder=3)
        ax.text(high + 0.0015, yi, f"{value:.3f}", va="center", ha="left", fontsize=MIN_TEXT)
    ax.set_yticks(y, [r[1] for r in rows])
    for tick, label, row in zip(ax.yaxis.get_major_ticks(), ax.get_yticklabels(), rows):
        if row[0] == "header":
            label.set_fontweight("bold")
            label.set_fontstyle("italic")
            tick.tick1line.set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(0.34, 0.464)
    ax.set_xticks(np.arange(0.34, 0.461, 0.02))
    ax.grid(axis="x", color="#e2e2e2", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Cross-fitted validation PQ (144 observations)")
    note = None
    if len(run3) == 2:
        note = (f"U-Net variants were tuned over a wider fusion grid, so run 3 with\n"
                f"detectors 1+2 reads {run3[RUN3_WIDE]:.3f} there and {run3[RUN3_NARROW]:.3f} "
                "in the Det 1+2 row.")
        fig.supxlabel(note, fontsize=MIN_TEXT, color=MUTED)
    handles = [
        Line2D([], [], marker="o", color=MUTED, markerfacecolor=MUTED, linestyle="none",
               markersize=4.0, label="First run of a recipe"),
        Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", linestyle="none",
               markersize=4.0, label="Same recipe, another seed"),
        Patch(facecolor=MUTED, alpha=0.22, edgecolor="none", label="Seed-to-seed range"),
        # A neutral, undrawn error bar, only for the legend.
        ax.errorbar([np.nan], [np.nan], xerr=[[0.01], [0.01]], ecolor=MUTED,
                    label="95% bootstrap interval", **bar_style),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=MIN_TEXT,
               handlelength=1.4, columnspacing=1.0, handletextpad=0.5)
    path = _save(fig, "fig_seeds.pdf")
    return {"file": path,
            "metric": "pooled PQ of each run's cross-fitted totals (pq_of(crossfit.sum(0))); "
                      "holdout_pq is the 72 held-out observations only",
            "interval": f"95% bootstrap percentile interval, {N_BOOT} resamples of the 144 "
                        f"observations (rows of crossfit) with replacement, seed {BOOT_SEED}; "
                        "the same resamples for every run",
            "groups": out_groups,
            "baseline_unet_only": {"file": f"outputs/step3/{BASELINE_TOTALS}",
                                   "array": "baseline", **(baseline or {"pq": None}),
                                   "note": "U-Net alone, fixed setting thr 0.6, area 400, "
                                           "gap 24, close 3, open 0; not tuned per half"},
            "run3_det12": {"wide_grid": run3.get(RUN3_WIDE), "usual_grid": run3.get(RUN3_NARROW),
                           "note": note},
            "seed_pairs": seed_out, "missing_files": missing}


# --- Figure 5: public leaderboard progress ---------------------------------------------


def fig_progress() -> dict:
    plt = _pyplot()
    labels = [s[0] for s in PUBLIC_SUBMISSIONS]
    scores = np.array([s[2] for s in PUBLIC_SUBMISSIONS])
    x = np.arange(1, len(scores) + 1)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.85), layout="constrained")
    ax.plot(x, scores, color=GREY, linewidth=0.9, zorder=1)
    jumps = []
    for name, a, b in PUBLIC_JUMPS:
        ax.plot(x[a:b + 1], scores[a:b + 1], color=OI["vermillion"], linewidth=1.8, zorder=2,
                solid_capstyle="round")
        jumps.append({"name": name, "from": PUBLIC_SUBMISSIONS[a][1],
                      "to": PUBLIC_SUBMISSIONS[b][1], "from_score": scores[a],
                      "to_score": scores[b], "delta": round(float(scores[b] - scores[a]), 2)})
    ax.plot(x, scores, marker="o", markersize=3.8, color=OI["blue"], linestyle="none", zorder=3)
    for i, (xi, s) in enumerate(zip(x, scores)):
        rising = i + 1 < len(scores) and scores[i + 1] - s > 0.05
        if rising and i == 0:  # below: the steep segment leaves up and to the right
            ax.text(xi, s - 0.014, f"{s:.2f}", ha="center", va="top", fontsize=MIN_TEXT)
        elif rising:  # beside the point, clear of the segments on either side
            ax.text(xi - 0.18, s, f"{s:.2f}", ha="right", va="center", fontsize=MIN_TEXT)
        else:
            ax.text(xi, s + 0.012, f"{s:.2f}", ha="center", va="bottom", fontsize=MIN_TEXT)
    # Each jump's label sits just below its highlighted segment, with a short leader.
    placements = {"U-Net": ((1.75, 0.115), (1.5, 0.15)),
                  "TTA": ((3.3, 0.262), (3.5, 0.31)),
                  "Detector fusion": ((5.7, 0.292), (5.5, 0.34))}
    for jump in jumps:
        text_xy, target = placements[jump["name"]]
        ax.annotate(f"{jump['name']} +{jump['delta']:.2f}", xy=target, xytext=text_xy,
                    fontsize=MIN_TEXT, ha="left", va="center", color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6,
                                "shrinkA": 0.5, "shrinkB": 1.5})
    ax.set_xticks(x, labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=MIN_TEXT)
    ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_ylim(0, 0.42)
    ax.set_yticks(np.arange(0, 0.41, 0.1))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax.set_ylabel("Public leaderboard score")
    ax.set_xlabel("Submission, in order")
    ax.grid(axis="y", color="#e2e2e2", linewidth=0.5)
    ax.set_axisbelow(True)
    path = _save(fig, "fig_progress.pdf")
    return {"file": path, "note": "public leaderboard, two decimals as the board shows them",
            "submissions": [{"order": i + 1, "label": s[1], "axis_label": s[0], "public": s[2]}
                            for i, s in enumerate(PUBLIC_SUBMISSIONS)],
            "jumps": jumps}


# --- Main -------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="processes for the per-observation scoring")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    print(f"scoring V on {len(inputs.stems)} validation observations "
          f"({sum(len(v) for v in inputs.views.values())} views), {args.workers} workers")
    per_stem = analyse_validation(inputs, args.workers)

    numbers: dict = {"meta": {
        "generated_by": "reports/make_figures.py",
        "pipeline": "V (validated entry): run-3 ResNet34 U-Net, last epoch, 8-way dihedral TTA, "
                    "fused with detectors 1+2+4 (merged with fusion.merge_views)",
        "fusion_params": asdict(PARAMS),
        "inputs": {"logits": str(LOGIT_DIR.relative_to(REPO)),
                   "detections_val": str(DETECTIONS_VAL.relative_to(REPO)),
                   "disk_geometry": str((CACHE_DIR / "disk.json").relative_to(REPO)),
                   "flat_images": str((CACHE_DIR / "flat").relative_to(REPO)),
                   "split": str(SPLIT.relative_to(REPO)),
                   "test_submission": str(SUBMISSION_TEST.relative_to(REPO))},
        "scoring": "every annotator view scored separately, TP/FP/FN/IoU pooled over views "
                   "(filament_seg.metrics.evaluate_image with fragmentation, summarize)",
        "in_sample_note": "V's fusion setting was chosen on all 144 validation observations, so "
                          "these validation numbers are in-sample for the fusion knobs; V's "
                          "cross-fitted PQ is in fig_seeds (Det 1+2+4)",
    }}
    numbers["validation"] = validation_numbers(per_stem)
    numbers["validation"]["consistency_check"] = consistency_check(per_stem)
    numbers["test"] = test_numbers(per_stem)
    numbers["progression_intervals"] = progression_intervals()
    numbers["detectors_only"] = detectors_only_numbers(per_stem, inputs)

    print("drawing figures")
    choice = choose_qualitative(per_stem, inputs.disks)
    numbers["fig_qualitative"] = fig_qualitative(choice, inputs)
    picks = choose_examples(per_stem, inputs.disks, exclude={choice["entry"]["stem"]})
    numbers["fig_examples"] = fig_examples(picks, inputs)
    numbers["fig_iou_dice"] = fig_iou_dice(per_stem)
    numbers["fig_errors"] = fig_errors(per_stem)
    numbers["fig_seeds"] = fig_seeds()
    numbers["fig_progress"] = fig_progress()

    numbers["meta"]["runtime_seconds"] = round(time.perf_counter() - started, 1)
    NUMBERS.write_text(json.dumps(_round(numbers), indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")

    v = numbers["validation"]
    print(f"V on validation: PQ {v['pq']:.4f}  SQ {v['sq']:.4f}  RQ {v['rq']:.4f}  "
          f"TP {v['tp']}  FP {v['fp']}  FN {v['fn']}  "
          f"one-to-many {v['one_to_many']}  many-to-one {v['many_to_one']}")
    d = numbers["detectors_only"]
    print(f"detectors only, cross-fitted: PQ {d['crossfit_144']['pq']:.4f} "
          f"{d['crossfit_144']['pq_ci95']}; V minus detectors only "
          f"{d['v_minus_detectors_only']['crossfit_144']['delta']:+.4f} "
          f"{d['v_minus_detectors_only']['crossfit_144']['ci95']}")
    check = v["consistency_check"]
    print(f"consistency with {V_TOTALS}: {'ok' if check['matches'] else 'MISMATCH'} "
          f"(max |diff| {check['max_abs_difference']:.2e})")
    if numbers["fig_seeds"]["missing_files"]:
        print(f"missing totals files, skipped: {numbers['fig_seeds']['missing_files']}")
    print(f"wrote {NUMBERS.relative_to(REPO)} and {len(list(FIG_DIR.glob('*.pdf')))} figures "
          f"in {numbers['meta']['runtime_seconds']:.0f} s")


if __name__ == "__main__":
    main()
