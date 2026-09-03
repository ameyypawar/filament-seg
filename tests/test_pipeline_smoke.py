"""End-to-end smoke test on a synthetic Sun -- no competition data required.

Builds a limb-darkened disk with a few dark elongated blobs on it, then checks
that disk detection, limb-darkening correction, thresholding, instance
formation and RLE encoding all agree with each other. Cheap insurance against
the pipeline breaking silently between real data runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament_seg.baseline import BaselineParams, segment
from filament_seg.disk import detect_disk, flatten
from filament_seg.postprocess import PostprocessParams, binary_to_instances
from filament_seg.rle import counts_of, iou_matrix, labels_to_rles

SIZE = 512
CENTER = SIZE / 2.0
RADIUS = 220.0

# Filaments as (row slice, column slice) boxes, in pixels.
FILAMENTS = [
    (slice(200, 215), slice(160, 260)),
    (slice(300, 312), slice(240, 330)),
    (slice(150, 190), slice(300, 316)),
]


@pytest.fixture(scope="module")
def synthetic_sun() -> np.ndarray:
    ys, xs = np.ogrid[:SIZE, :SIZE]
    r = np.sqrt((xs - CENTER) ** 2 + (ys - CENTER) ** 2) / RADIUS

    # Classic limb-darkening falloff, bright centre to dim edge.
    image = np.zeros((SIZE, SIZE), dtype=np.float32)
    inside = r <= 1.0
    image[inside] = 200.0 * (0.4 + 0.6 * np.sqrt(np.clip(1 - r[inside] ** 2, 0, 1)))

    for rows, cols in FILAMENTS:
        image[rows, cols] *= 0.55

    rng = np.random.default_rng(0)
    image += rng.normal(0, 1.5, image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_disk_detection_recovers_geometry(synthetic_sun):
    disk = detect_disk(synthetic_sun)
    assert disk.cx == pytest.approx(CENTER, abs=6)
    assert disk.cy == pytest.approx(CENTER, abs=6)
    assert disk.radius == pytest.approx(RADIUS, rel=0.05)


def test_flatten_removes_the_radial_gradient(synthetic_sun):
    disk = detect_disk(synthetic_sun)
    flat = flatten(synthetic_sun, disk)
    r = disk.radius_map(synthetic_sun.shape)

    # Compare quiet-Sun level near the centre against near the limb. Before
    # flattening these differ by roughly a factor of two.
    inner = flat[(r < 0.3)]
    outer = flat[(r > 0.7) & (r < 0.9)]
    assert np.median(inner) == pytest.approx(np.median(outer), rel=0.05)


def test_baseline_finds_the_planted_filaments(synthetic_sun):
    params = BaselineParams(
        k=2.0,
        postprocess=PostprocessParams(min_area=100, bridge_gap=4, close_radius=3),
    )
    labels, disk = segment(synthetic_sun, params)

    rles = labels_to_rles(labels)
    assert len(rles) == len(FILAMENTS)

    truth = []
    for rows, cols in FILAMENTS:
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        mask[rows, cols] = 1
        truth.append(mask)

    from filament_seg.rle import mask_to_rle

    ious = iou_matrix([mask_to_rle(m) for m in truth], rles)
    # Every planted filament should be recovered well past the PQ threshold.
    assert (ious.max(axis=1) > 0.5).all()

    for rle in rles:
        assert counts_of(rle)


def test_nothing_is_detected_outside_the_disk(synthetic_sun):
    labels, disk = segment(synthetic_sun, BaselineParams(k=2.0))
    outside = ~disk.mask(synthetic_sun.shape, shrink=1.0)
    assert labels[outside].max() == 0


def test_bridging_merges_a_split_filament():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[60:70, 20:55] = 1
    mask[60:70, 63:100] = 1  # same filament, 8 px gap

    split = binary_to_instances(
        mask, PostprocessParams(open_radius=0, close_radius=0, bridge_gap=0, min_area=10)
    )
    joined = binary_to_instances(
        mask, PostprocessParams(open_radius=0, close_radius=0, bridge_gap=12, min_area=10)
    )
    assert split.max() == 2
    assert joined.max() == 1
