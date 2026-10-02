"""Validation sampling, post-processing filters and the sweep's logit cache.

Each test pins down a bug the code review found: checkpoint selection that saw
only one annotator batch, scoring that ignored all but one annotator, a cache
that could hand back a previous model's logits, and a missing mask that was
silently read as "no filaments".
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from filament_seg.data import (
    Annotations,
    ImageRecord,
    records_for_stems,
    sample_stems,
    stems_of,
)
from filament_seg.postprocess import (
    PostprocessParams,
    drop_low_confidence,
    filter_by_area,
    logits_to_instances,
)

REPO = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_script("sweep_postprocess")


@pytest.fixture()
def annotations() -> Annotations:
    """30 observations, viewed by one to three of six annotator batches."""
    images = {}
    for n in range(30):
        stem = f"201{n % 10}0101{n:06d}Bh"
        for batch in range(1 + n % 3):
            image_id = f"0{(n + batch) % 6 + 1}0101-{stem}"
            images[image_id] = ImageRecord(image_id, f"{stem}.jpg", 2048, 2048)
    return Annotations(images=images, annotations_by_image={}, categories={})


# --- validation sampling -----------------------------------------------------


def test_sample_spans_annotator_batches(annotations):
    # Taking the first n sorted view ids would draw every view from batch 01.
    ids = sorted(annotations.images)
    stems = sample_stems(annotations, ids, 10, seed=0)
    batches = {annotations.images[i].annotator[:2]
               for i in records_for_stems(annotations, ids, stems)}
    assert len(stems) == 10
    assert len(batches) > 1


def test_sample_is_deterministic_and_nested(annotations):
    ids = sorted(annotations.images)
    assert sample_stems(annotations, ids, 10, seed=0) == sample_stems(annotations, ids, 10, seed=0)
    # Checkpoint selection (fewer observations) must stay inside the sweep's
    # tuning sample, so neither ever touches the held-out half.
    assert set(sample_stems(annotations, ids, 8)) <= set(sample_stems(annotations, ids, 15))


def test_sample_of_zero_means_all(annotations):
    ids = sorted(annotations.images)
    assert sample_stems(annotations, ids, 0) == stems_of(annotations, ids)


def test_records_for_stems_returns_every_view(annotations):
    ids = sorted(annotations.images)
    views_per_stem = {s: sum(annotations.images[i].stem == s for i in ids)
                      for s in stems_of(annotations, ids)}
    stem = max(views_per_stem, key=views_per_stem.get)
    views = records_for_stems(annotations, ids, [stem])
    assert len(views) == 3
    assert {annotations.images[i].stem for i in views} == {stem}


# --- post-processing ---------------------------------------------------------


def _reference_filter_by_area(labels: np.ndarray, min_area: int) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    for value in np.unique(labels):
        if value == 0 or int((labels == value).sum()) < min_area:
            continue
        out[labels == value] = next_label
        next_label += 1
    return out


def test_fast_area_filter_matches_the_reference():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 40, size=(64, 64)).astype(np.int32)
    labels[labels > 30] = 0
    for min_area in (0, 90, 110, 130, 10_000):
        np.testing.assert_array_equal(filter_by_area(labels, min_area),
                                      _reference_filter_by_area(labels, min_area))


def test_low_confidence_instances_are_dropped_and_survivors_relabelled():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0:3, 0:3], labels[5:8, 5:8], labels[0:2, 7:9] = 1, 2, 3
    logits = np.full((10, 10), -5.0, dtype=np.float32)
    logits[labels == 1] = 3.0     # p ~ 0.95
    logits[labels == 2] = 0.0     # p = 0.5
    logits[labels == 3] = 2.0     # p ~ 0.88
    out = drop_low_confidence(labels, logits, 0.8)
    assert sorted(np.unique(out)) == [0, 1, 2]
    assert (out[labels == 2] == 0).all()
    assert (out[labels == 1] == 1).all() and (out[labels == 3] == 2).all()
    np.testing.assert_array_equal(drop_low_confidence(labels, logits, 0.0), labels)


def test_logits_to_instances_applies_threshold_disk_and_confidence():
    logits = np.full((64, 64), -6.0, dtype=np.float32)
    logits[10:20, 10:40] = 4.0          # confident, inside the disk
    logits[40:50, 10:40] = 0.4          # p ~ 0.6: above 0.5, below 0.7
    logits[30:34, 60:64] = 4.0          # outside the disk
    disk = np.ones((64, 64), dtype=bool)
    disk[:, 56:] = False
    params = PostprocessParams(threshold=0.5, open_radius=0, close_radius=0,
                               bridge_gap=0, min_area=10)
    assert logits_to_instances(logits, disk, params).max() == 2
    params.min_confidence = 0.7
    labels = logits_to_instances(logits, disk, params)
    assert labels.max() == 1 and (labels[10:20, 10:40] == 1).all()
    params.threshold = 0.99
    assert logits_to_instances(logits, disk, params).max() == 0


# --- the sweep's logit cache -------------------------------------------------


def _fake_checkpoint(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return str(path)


def test_forced_rerun_deletes_the_previous_models_logits(tmp_path):
    model_a = _fake_checkpoint(tmp_path / "a.pt", b"a")
    model_b = _fake_checkpoint(tmp_path / "b.pt", b"bb")
    cache = tmp_path / "logits"
    sweep._check_cache_provenance(cache, [model_a], "none", 512, 128, force=False)
    for stem in ("s1", "s2", "s3"):
        np.save(cache / f"{stem}.npy", np.zeros(2))

    # A forced run for model B that dies before writing anything...
    sweep._check_cache_provenance(cache, [model_b], "none", 512, 128, force=True)
    # ...must not leave model A's maps behind for the next plain run to accept.
    assert not list(cache.glob("*.npy"))
    sweep._check_cache_provenance(cache, [model_b], "none", 512, 128, force=False)


def test_cache_refuses_another_model_tta_or_tiling(tmp_path):
    model = _fake_checkpoint(tmp_path / "a.pt", b"a")
    other = _fake_checkpoint(tmp_path / "b.pt", b"bb")
    cache = tmp_path / "logits"
    sweep._check_cache_provenance(cache, [model], "none", 512, 128, force=False)
    for args in (([other], "none", 512, 128), ([model], "dihedral", 512, 128),
                 ([model], "none", 1024, 128)):
        with pytest.raises(SystemExit):
            sweep._check_cache_provenance(cache, *args, force=False)
    meta = json.loads((cache / "cache_meta.json").read_text())
    assert meta["tile"] == 512 and meta["overlap"] == 128


def test_unlabelled_cache_is_refused(tmp_path):
    cache = tmp_path / "logits"
    cache.mkdir()
    np.save(cache / "s1.npy", np.zeros(2))
    with pytest.raises(SystemExit):
        sweep._check_cache_provenance(cache, [_fake_checkpoint(tmp_path / "a.pt", b"a")],
                                      "none", 512, 128, force=False)


# --- the sweep's statistics --------------------------------------------------


def test_pq_of_matches_the_formula():
    # n_gt, n_pred, tp, fp, fn, iou_sum
    totals = np.array([10, 8, 6, 2, 4, 4.2])
    assert sweep.pq_of(totals) == pytest.approx(4.2 / (6 + 1 + 2))
    assert sweep.pq_of(np.zeros(6)) == 0.0


def test_identical_settings_differ_by_exactly_zero():
    rng = np.random.default_rng(1)
    per_stem = rng.integers(0, 10, size=(40, 6)).astype(np.float64)
    delta, low, high = sweep.paired_bootstrap(per_stem, per_stem)
    assert delta == low == high == 0.0


def test_a_uniformly_better_setting_has_a_positive_interval():
    per_stem_b = np.tile([5, 5, 3, 2, 2, 2.0], (40, 1))
    per_stem_a = np.tile([5, 5, 4, 1, 1, 2.8], (40, 1))
    delta, low, high = sweep.paired_bootstrap(per_stem_a, per_stem_b)
    assert delta > 0 and low > 0


def test_baseline_settings_parse_with_defaults_for_the_rest():
    params = sweep.parse_params("threshold=0.7, min_area=400,bridge_gap=16")
    assert (params.threshold, params.min_area, params.bridge_gap) == (0.7, 400, 16)
    assert params.min_confidence == PostprocessParams().min_confidence
    with pytest.raises(Exception):
        sweep.parse_params("nonsense=1")
