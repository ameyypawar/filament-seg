"""Detector-guided instance formation.

The U-Net answers "is this pixel a filament?" well -- its matched masks reach
an IoU of about 0.67 -- but it has no notion of which pixels belong to the
*same* filament. Connected components and gap bridging guess that from
geometry alone, and that guess is where most of the Panoptic Quality goes:
the U-Net pipeline makes about as many false positives as true positives. An
instance detector answers the grouping question directly and scores every
instance, but its masks are coarse: predicted at a quarter of its input
resolution, which is itself half the image's.

Fusion takes the best of both. Each detection, in descending confidence,
claims the full-resolution U-Net filament pixels inside a slightly grown copy
of its coarse mask. A pixel belongs to the first detection that claims it, so
instances never overlap. U-Net pixels that no detection claims are dropped --
or, with ``keep_unclaimed``, grouped the way the U-Net-only pipeline groups
them (gap bridging, then an area cut), for filaments the detector missed. At
``keep_unclaimed`` equal to the U-Net pipeline's own ``min_area``, fusion only
changes the grouping where a detection is confident enough to claim pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .postprocess import (
    PostprocessParams,
    _ellipse,
    bridge_and_label,
    clean_binary,
    fill_holes,
    probability_to_logit,
)
from .rle import counts_to_mask


@dataclass(frozen=True)
class Detection:
    """One detector instance: its confidence and its mask as COCO RLE counts."""

    score: float
    counts: str


@dataclass(frozen=True)
class Region:
    """A detection's grown mask, cropped to its bounding box."""

    score: float
    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray


@dataclass
class FusionParams:
    """Knobs for detector-guided instance formation."""

    #: U-Net probability above which a pixel counts as filament.
    threshold: float = 0.5
    #: Morphological closing applied to the U-Net mask before assignment.
    close_radius: int = 3
    #: Pixels the detector's coarse mask is grown by before claiming U-Net
    #: pixels; it is predicted at an eighth of full resolution.
    grow: int = 8
    #: Detections below this confidence claim nothing.
    min_score: float = 0.25
    #: Instances smaller than this, after claiming, are dropped.
    min_area: int = 400
    #: Unclaimed U-Net pixels are grouped by bridging gaps of this many
    #: pixels, and groups at least ``keep_unclaimed`` large are kept as
    #: instances of their own; 0 drops every unclaimed pixel.
    keep_unclaimed: int = 0
    unclaimed_gap: int = 24


def unet_binary(
    logits: np.ndarray, disk_mask: np.ndarray, threshold: float, close_radius: int
) -> np.ndarray:
    """The U-Net's filament pixels: thresholded, closed, holes filled, on the disk."""
    binary = ((logits > probability_to_logit(threshold)) & disk_mask).astype(np.uint8)
    binary = clean_binary(binary, PostprocessParams(open_radius=0, close_radius=close_radius))
    return fill_holes(binary).astype(bool) & disk_mask


def grow_regions(
    detections: list[Detection],
    grow: int,
    shape: tuple[int, int],
    min_score: float = 0.0,
) -> list[Region]:
    """Decode, crop and grow each detection's mask, highest confidence first."""
    height, width = shape
    regions = []
    kernel = _ellipse(grow) if grow > 0 else None
    for detection in sorted(detections, key=lambda d: -d.score):
        if detection.score < min_score:
            continue
        mask = counts_to_mask(detection.counts, height, width)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        y0, y1 = max(int(ys.min()) - grow, 0), min(int(ys.max()) + grow + 1, height)
        x0, x1 = max(int(xs.min()) - grow, 0), min(int(xs.max()) + grow + 1, width)
        crop = mask[y0:y1, x0:x1].astype(np.uint8)
        if kernel is not None:
            crop = cv2.dilate(crop, kernel)
        regions.append(Region(detection.score, y0, y1, x0, x1, crop.astype(bool)))
    return regions


def assign(
    binary: np.ndarray,
    regions: list[Region],
    min_score: float,
    min_area: int,
    keep_unclaimed: int = 0,
    unclaimed_gap: int = 24,
) -> np.ndarray:
    """Integer instance labels: each region, best first, claims its U-Net pixels."""
    labels = np.zeros(binary.shape, dtype=np.int32)
    claimed = np.zeros(binary.shape, dtype=bool)
    next_label = 1
    for region in regions:
        if region.score < min_score:
            break
        window = (slice(region.y0, region.y1), slice(region.x0, region.x1))
        pixels = binary[window] & region.mask & ~claimed[window]
        if int(pixels.sum()) < min_area:
            continue
        labels[window][pixels] = next_label
        claimed[window] |= pixels
        next_label += 1

    if keep_unclaimed > 0:
        components = bridge_and_label((binary & ~claimed).astype(np.uint8), unclaimed_gap)
        counts = np.bincount(components.ravel())
        keep = counts >= keep_unclaimed
        keep[0] = False
        lookup = np.zeros(counts.size, dtype=np.int32)
        lookup[keep] = np.arange(next_label, next_label + int(keep.sum()), dtype=np.int32)
        extra = lookup[components]
        labels = np.where(extra > 0, extra, labels)
    return labels


def fuse(
    logits: np.ndarray,
    disk_mask: np.ndarray,
    detections: list[Detection],
    params: FusionParams | None = None,
) -> np.ndarray:
    """U-Net logits + detector instances -> integer instance label map."""
    params = params or FusionParams()
    binary = unet_binary(logits, disk_mask, params.threshold, params.close_radius)
    regions = grow_regions(detections, params.grow, logits.shape, params.min_score)
    return assign(binary, regions, params.min_score, params.min_area, params.keep_unclaimed,
                  params.unclaimed_gap)
