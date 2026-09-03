"""Turning a binary probability/threshold map into scored filament instances.

This is where Panoptic Quality is won or lost. The organisers name structural
continuity as challenge #3: algorithms tend to shatter one filament into a
cluster of dark "islands". Under PQ a shattered filament is one false negative
*plus* several false positives, so fragmentation is punished roughly three times
over while pixel accuracy barely moves.

The pipeline here is deliberately ordered:

``open`` (kill speckle) -> ``close`` (rejoin thin necks) -> ``bridge`` (join
components separated by a small gap) -> ``area filter`` (drop instances too
small to plausibly clear IoU 0.5).

Every step is a tunable knob; tune them against local PQ, never against Dice.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PostprocessParams:
    """Knobs for binary -> instances. Defaults are a starting point, not tuned."""

    open_radius: int = 2
    close_radius: int = 5
    #: Components whose gap is below this many pixels are merged into one
    #: instance. Directly targets the fragmentation penalty.
    bridge_gap: int = 12
    #: Instances smaller than this (in pixels, at 2048x2048) are dropped.
    #: A doubtful instance costs 1.0 in the PQ denominator; silence costs 0.5.
    min_area: int = 400
    #: Fill interior holes so a filament is one solid blob.
    fill_holes: bool = True


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
    """Drop instances below ``min_area`` and relabel consecutively from 1."""
    out = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for value in np.unique(labels):
        if value == 0:
            continue
        component = labels == value
        if int(component.sum()) < min_area:
            continue
        out[component] = next_label
        next_label += 1
    return out


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
