"""Training-side pieces: the Tversky option of the loss and the crop dataset's targets."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from filament_seg.data import Annotations, ImageRecord
from filament_seg.dataset import FilamentCrops
from filament_seg.disk import Disk
from filament_seg.model import DiceBCELoss

SIZE = 64
STEM = "20140101000000Bh"
VIEWS = [f"01010{n}-{STEM}" for n in (1, 2)]


def test_default_loss_is_plain_dice_plus_bce():
    torch.manual_seed(0)
    logits = torch.randn(3, 1, 16, 16)
    target = (torch.rand(3, 1, 16, 16) > 0.7).float()
    probs = torch.sigmoid(logits).flatten(1)
    flat = target.flatten(1)
    dice = (2 * (probs * flat).sum(1) + 1) / (probs.sum(1) + flat.sum(1) + 1)
    expected = 1 - dice.mean() + torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert DiceBCELoss()(logits, target).item() == pytest.approx(expected.item(), abs=1e-6)


def test_recall_weighting_punishes_misses_more_than_extras():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    shrunk = torch.full_like(target, -10.0)
    shrunk[..., 3:5, 3:5] = 10.0  # finds a quarter of the filament
    grown = torch.full_like(target, -10.0)
    grown[..., 0:8, 0:8] = 10.0  # finds all of it and four times as much sky
    dice_only = dict(bce_weight=0.0)
    plain, recall = DiceBCELoss(**dice_only), DiceBCELoss(fn_weight=0.7, **dice_only)
    assert recall(shrunk, target) > plain(shrunk, target)
    assert recall(grown, target) < plain(grown, target)


def test_radius_window_is_the_same_slice_of_the_radius_map():
    disk = Disk(cx=900.3, cy=1010.7, radius=899.0)
    full = disk.radius_map((2048, 2048))
    np.testing.assert_allclose(disk.radius_window(300, 1200, 512, 256), full[300:812, 1200:1456])


@pytest.fixture()
def cache(tmp_path):
    """One observation seen by two annotators, cached the way preprocess.py writes it."""
    for sub in ("flat", "mask", "consensus"):
        (tmp_path / sub).mkdir()
    rng = np.random.default_rng(0)
    cv2.imwrite(str(tmp_path / "flat" / f"{STEM}.png"), rng.integers(0, 256, (SIZE, SIZE), np.uint8))
    first, second = np.zeros((SIZE, SIZE), np.uint8), np.zeros((SIZE, SIZE), np.uint8)
    first[10:20, 10:50] = 255
    second[30:40, 10:50] = 255
    cv2.imwrite(str(tmp_path / "mask" / f"{VIEWS[0]}.png"), first)
    cv2.imwrite(str(tmp_path / "mask" / f"{VIEWS[1]}.png"), second)
    agreement = ((first > 0).astype(np.uint8) + (second > 0)) * 127
    cv2.imwrite(str(tmp_path / "consensus" / f"{STEM}.png"), agreement)
    (tmp_path / "disk.json").write_text(json.dumps({STEM: {"cx": 32.0, "cy": 32.0, "radius": 30.0}}))
    images = {v: ImageRecord(v, f"{STEM}.jpg", SIZE, SIZE) for v in VIEWS}
    return tmp_path, Annotations(images=images, annotations_by_image={}, categories={})


def _crops(cache_dir, annotations, **kwargs):
    dataset = FilamentCrops(VIEWS, annotations, cache_dir, crop_size=SIZE, crops_per_image=1,
                            augment=False, **kwargs)
    return dataset, [dataset[i] for i in range(len(dataset))]


def test_union_target_is_every_pixel_any_annotator_drew(cache):
    cache_dir, annotations = cache
    dataset, crops = _crops(cache_dir, annotations, target="union")
    assert len(dataset) == 1  # one observation, whichever view was picked
    y = crops[0][1][0].numpy()
    expected = np.zeros((SIZE, SIZE), np.float32)
    expected[10:20, 10:50] = expected[30:40, 10:50] = 1.0
    np.testing.assert_array_equal(y, expected)


def test_annotator_target_keeps_each_view_and_its_own_mask(cache):
    cache_dir, annotations = cache
    dataset, crops = _crops(cache_dir, annotations, target="annotator")
    assert len(dataset) == 2
    assert [float(y.sum()) for _, y in crops] == [400.0, 400.0]
    assert not torch.equal(crops[0][1], crops[1][1])


def test_preloading_changes_nothing_but_speed(cache):
    cache_dir, annotations = cache
    _, lazy = _crops(cache_dir, annotations, target="annotator")
    _, preloaded = _crops(cache_dir, annotations, target="annotator", preload=True)
    for (x_lazy, y_lazy), (x_pre, y_pre) in zip(lazy, preloaded):
        assert torch.equal(x_lazy, x_pre) and torch.equal(y_lazy, y_pre)
