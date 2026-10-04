"""Detector-guided instance formation: who claims which U-Net pixels."""

from __future__ import annotations

import numpy as np

from filament_seg.fusion import (
    Detection,
    FusionParams,
    assign,
    fuse,
    grow_regions,
    merge_views,
)
from filament_seg.rle import mask_to_counts

SHAPE = (64, 64)


def _detection(score: float, rows: slice, cols: slice) -> Detection:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[rows, cols] = 1
    return Detection(score, mask_to_counts(mask))


def _binary(*boxes: tuple[slice, slice]) -> np.ndarray:
    binary = np.zeros(SHAPE, dtype=bool)
    for rows, cols in boxes:
        binary[rows, cols] = True
    return binary


def test_two_detections_split_one_unet_component():
    binary = _binary((slice(10, 14), slice(5, 60)))       # one long U-Net streak
    detections = [_detection(0.9, slice(8, 16), slice(3, 30)),
                  _detection(0.8, slice(8, 16), slice(30, 62))]
    labels = assign(binary, grow_regions(detections, 0, SHAPE), min_score=0.5, min_area=1)
    assert labels.max() == 2
    assert set(np.unique(labels[10:14, 5:30])) == {1}
    assert set(np.unique(labels[10:14, 30:60])) == {2}


def test_fragments_inside_one_detection_become_one_instance():
    binary = _binary((slice(10, 14), slice(5, 20)), (slice(10, 14), slice(26, 40)))
    detections = [_detection(0.9, slice(8, 16), slice(3, 42))]
    labels = assign(binary, grow_regions(detections, 0, SHAPE), min_score=0.5, min_area=1)
    assert labels.max() == 1


def test_low_confidence_detections_claim_nothing():
    binary = _binary((slice(10, 14), slice(5, 20)))
    detections = [_detection(0.2, slice(8, 16), slice(3, 22))]
    assert assign(binary, grow_regions(detections, 0, SHAPE), 0.5, 1).max() == 0


def test_higher_confidence_claims_shared_pixels_first():
    binary = _binary((slice(10, 14), slice(5, 40)))
    detections = [_detection(0.6, slice(8, 16), slice(3, 40)),
                  _detection(0.9, slice(8, 16), slice(20, 42))]
    labels = assign(binary, grow_regions(detections, 0, SHAPE), 0.5, 1)
    # The 0.9 detection is processed first and keeps the overlap (columns 20-39).
    assert (labels[10:14, 20:40] == 1).all()
    assert (labels[10:14, 5:20] == 2).all()


def test_small_claims_are_dropped_and_large_orphans_kept_on_request():
    binary = _binary((slice(10, 12), slice(5, 8)),          # 6 px, claimed
                     (slice(40, 50), slice(40, 60)))        # 200 px, unclaimed
    detections = [_detection(0.9, slice(8, 14), slice(3, 10))]
    regions = grow_regions(detections, 0, SHAPE)
    assert assign(binary, regions, 0.5, min_area=10).max() == 0
    kept = assign(binary, regions, 0.5, min_area=1, keep_unclaimed=100)
    assert kept.max() == 2 and (kept[40:50, 40:60] == 2).all()


def test_unclaimed_fragments_are_grouped_like_the_unet_pipeline():
    # Two 100 px fragments 6 px apart, unclaimed: bridged into one 200 px
    # instance they clear keep_unclaimed=150; left apart, neither does.
    binary = _binary((slice(40, 50), slice(10, 20)), (slice(40, 50), slice(26, 36)))
    assert assign(binary, [], 0.5, 1, keep_unclaimed=150, unclaimed_gap=0).max() == 0
    joined = assign(binary, [], 0.5, 1, keep_unclaimed=150, unclaimed_gap=12)
    assert joined.max() == 1 and (joined[binary] == 1).all()


def test_growing_reaches_pixels_just_outside_the_coarse_mask():
    binary = _binary((slice(10, 14), slice(5, 30)))
    detections = [_detection(0.9, slice(11, 13), slice(8, 27))]   # thinner, shorter
    tight = assign(binary, grow_regions(detections, 0, SHAPE), 0.5, 1)
    grown = assign(binary, grow_regions(detections, 4, SHAPE), 0.5, 1)
    assert (tight > 0).sum() < (grown > 0).sum() == binary.sum()


def test_fuse_applies_the_unet_threshold_and_disk():
    logits = np.full(SHAPE, -5.0, dtype=np.float32)
    logits[10:14, 5:30] = 3.0
    logits[40:44, 5:30] = 0.2                      # p ~ 0.55
    disk = np.ones(SHAPE, dtype=bool)
    detections = [_detection(0.9, slice(8, 16), slice(3, 32)),
                  _detection(0.9, slice(38, 46), slice(3, 32))]
    params = FusionParams(threshold=0.5, close_radius=0, grow=0, min_score=0.5, min_area=10)
    assert fuse(logits, disk, detections, params).max() == 2
    params.threshold = 0.7
    assert fuse(logits, disk, detections, params).max() == 1
    disk[:, :] = False
    assert fuse(logits, disk, detections, params).max() == 0


def test_flipped_views_merge_into_one_detection_with_averaged_confidence():
    same = [_detection(score, slice(10, 14), slice(5, 40)) for score in (0.8, 0.6, 0.7, 0.5)]
    once = _detection(0.8, slice(40, 44), slice(5, 40))       # seen in one view only
    merged = merge_views([[same[0], once], [same[1]], [same[2]], [same[3]]])
    assert len(merged) == 2
    scores = sorted(round(d.score, 3) for d in merged)
    assert scores == [0.2, 0.65]                               # 0.8/4 and 2.6/4


def test_two_detections_from_the_same_view_never_merge():
    a = _detection(0.9, slice(10, 14), slice(5, 40))
    b = _detection(0.8, slice(10, 14), slice(6, 40))           # near-duplicate, same view
    assert len(merge_views([[a, b]])) == 2
