"""Turning a binary probability/threshold map into scored filament instances.

This is where Panoptic Quality is won or lost. The organisers name structural
continuity as challenge #3: algorithms tend to shatter one filament into a
cluster of dark "islands". Under PQ a shattered filament is one false negative
*plus* several false positives, so fragmentation is punished roughly three times
over while pixel accuracy barely moves.

The pipeline here is deliberately ordered:

``threshold`` (probability map -> binary) -> ``open`` (kill speckle) ->
``close`` (rejoin thin necks) -> ``bridge`` (join components separated by a
small gap) -> ``area filter`` (drop instances too small to plausibly clear IoU
0.5) -> ``confidence filter`` (drop instances the model is unsure of).

Every step is a tunable knob; tune them against local PQ, never against Dice.
:func:`logits_to_instances` is the single entry point used by training
validation, the post-processing sweep and the submission, so the three cannot
quietly disagree about what a setting means.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TypeVar

import cv2
import numpy as np


@dataclass
class PostprocessParams:
    """Knobs for logits -> instances. Defaults are a starting point, not tuned."""

    #: Probability above which a pixel counts as filament.
    threshold: float = 0.5
    open_radius: int = 2
    close_radius: int = 5
    #: Components whose gap is below this many pixels are merged into one
    #: instance. Directly targets the fragmentation penalty.
    bridge_gap: int = 12
    #: Instances smaller than this (in pixels, at 2048x2048) are dropped.
    #: A doubtful instance costs 1.0 in the PQ denominator; silence costs 0.5.
    min_area: int = 400
    #: Fill interior holes so a filament is one solid blob. Not swept: an
    #: unfilled ring can only lower IoU against the solid annotated polygons.
    fill_holes: bool = True
    #: Instances whose mean predicted probability is below this are dropped;
    #: 0 keeps everything. A prediction that misses costs 0.5 in the PQ
    #: denominator for nothing, so a faint, uncertain component is often worth
    #: more unsubmitted.
    min_confidence: float = 0.0


_Settings = TypeVar("_Settings")


def parse_settings(text: str, defaults: _Settings) -> _Settings:
    """``"threshold=0.7,min_area=400"`` -> a copy of ``defaults`` with those fields set.

    Values are converted to each field's type; an unknown name raises
    ``ValueError`` rather than being silently ignored.
    """
    values: dict = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, _, raw = item.partition("=")
        key = key.strip()
        if not hasattr(defaults, key):
            raise ValueError(f"unknown setting {key!r} for {type(defaults).__name__}")
        kind = type(getattr(defaults, key))
        values[key] = raw.strip().lower() in ("1", "true", "yes") if kind is bool else kind(raw)
    return dataclasses.replace(defaults, **values)


def probability_to_logit(p: float) -> float:
    """The logit threshold equivalent to probability ``p`` (clamped at 0 and 1)."""
    if p <= 0.0:
        return -1e9
    if p >= 1.0:
        return 1e9
    return float(np.log(p / (1.0 - p)))


def _ellipse(radius: int) -> np.ndarray:
    size = max(1, 2 * radius + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def clean_binary(binary: np.ndarray, params: PostprocessParams) -> np.ndarray:
    """Morphological cleanup of a binary mask (uint8, 0/1)."""
    out = (binary > 0).astype(np.uint8)
    if params.open_radius > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _ellipse(params.open_radius))
    if params.close_radius > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _ellipse(params.close_radius))
    return out


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions, leaving the outer background intact."""
    mask = (binary > 0).astype(np.uint8)
    flood = mask.copy()
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 1)
    return (mask | (1 - flood)).astype(np.uint8)


def bridge_and_label(binary: np.ndarray, gap: int) -> np.ndarray:
    """Label connected components, merging any that lie within ``gap`` pixels.

    Dilating by ``gap/2`` and labelling *that* lets nearby fragments share a
    label; the labels are then mapped back onto the original (undilated) pixels,
    so instance shapes are unchanged -- only their grouping is.
    """
    mask = (binary > 0).astype(np.uint8)
    if gap <= 0:
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        return labels

    dilated = cv2.dilate(mask, _ellipse(max(1, gap // 2)))
    _, group_labels = cv2.connectedComponents(dilated, connectivity=8)
    return np.where(mask > 0, group_labels, 0).astype(np.int32)


def filter_by_area(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Drop instances below ``min_area`` and relabel consecutively from 1.

    One counting pass and a lookup table, rather than a full-frame comparison
    per instance: the post-processing sweep calls this thousands of times.
    """
    counts = np.bincount(labels.ravel())
    keep = counts >= min_area
    keep[0] = False
    lookup = np.zeros(counts.size, dtype=np.int32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lookup[labels]


def binary_to_instances(
    binary: np.ndarray,
    params: PostprocessParams | None = None,
    restrict_to: np.ndarray | None = None,
) -> np.ndarray:
    """Full binary -> integer label map pipeline.

    ``restrict_to`` is typically the solar disk mask: anything outside it cannot
    be a filament and is discarded before instances are formed.
    """
    params = params or PostprocessParams()
    mask = (binary > 0).astype(np.uint8)
    if restrict_to is not None:
        mask = (mask & restrict_to.astype(np.uint8)).astype(np.uint8)

    mask = clean_binary(mask, params)
    if params.fill_holes:
        mask = fill_holes(mask)
    labels = bridge_and_label(mask, params.bridge_gap)
    return filter_by_area(labels, params.min_area)


def drop_low_confidence(
    labels: np.ndarray, logits: np.ndarray, min_confidence: float
) -> np.ndarray:
    """Drop instances whose mean probability is below ``min_confidence``.

    Survivors are relabelled consecutively from 1, as ``filter_by_area`` does.
    Only instance pixels are passed through the sigmoid, so this stays cheap
    on a 2048x2048 map.
    """
    if min_confidence <= 0.0:
        return labels
    inside = labels > 0
    if not inside.any():
        return labels
    ids = labels[inside]
    probs = 1.0 / (1.0 + np.exp(-logits[inside].astype(np.float64)))
    n = int(labels.max()) + 1
    sums = np.bincount(ids, weights=probs, minlength=n)
    counts = np.bincount(ids, minlength=n)
    keep = counts > 0
    keep[keep] = sums[keep] / counts[keep] >= min_confidence
    keep[0] = False
    lookup = np.zeros(n, dtype=np.int32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lookup[labels]


def logits_to_instances(
    logits: np.ndarray,
    disk_mask: np.ndarray,
    params: PostprocessParams | None = None,
) -> np.ndarray:
    """Model logits -> integer instance label map, restricted to the disk."""
    params = params or PostprocessParams()
    binary = ((logits > probability_to_logit(params.threshold)) & disk_mask).astype(np.uint8)
    labels = binary_to_instances(binary, params, restrict_to=disk_mask)
    return drop_low_confidence(labels, logits, params.min_confidence)
