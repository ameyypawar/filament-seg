"""Self-contained checks that run without the competition data.

These cover the two things most likely to silently destroy a score: a
transposed RLE encoding, and a Panoptic Quality implementation that disagrees
with the competition's formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from filament_seg.metrics import evaluate, evaluate_image
from filament_seg.rle import (
    build_submission,
    counts_of,
    counts_to_mask,
    iou_matrix,
    mask_to_rle,
    read_submission,
    rle_to_mask,
    validate_counts,
    write_submission,
)

H = W = 64


def box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def rle(y0: int, y1: int, x0: int, x1: int):
    return mask_to_rle(box(y0, y1, x0, x1))


# --- RLE round-trips --------------------------------------------------------


def test_rle_roundtrip_preserves_orientation():
    # Deliberately asymmetric: a transposed encoding would survive a square,
    # symmetric mask unnoticed.
    mask = box(0, 10, 0, 5)
    restored = rle_to_mask(mask_to_rle(mask))
    assert restored.shape == mask.shape
    assert np.array_equal(restored, mask.astype(bool))


def test_counts_roundtrip_via_string():
    mask = box(3, 20, 7, 9)
    counts = counts_of(mask_to_rle(mask))
    assert isinstance(counts, str)
    assert np.array_equal(counts_to_mask(counts, H, W), mask.astype(bool))


def test_counts_are_csv_safe():
    counts = counts_of(rle(2, 40, 3, 31))
    assert not set(',"\'\r\n').intersection(counts)


def test_submission_roundtrip(tmp_path):
    predictions = {"20150125172714Mh": [rle(0, 10, 0, 10), rle(20, 30, 20, 30)]}
    frame = build_submission(predictions)
    assert list(frame["filament_id"]) == [
        "20150125172714Mh_1",
        "20150125172714Mh_2",
    ]

    path = write_submission(frame, tmp_path / "sub.csv")
    # read_submission assumes the fixed 2048x2048 competition size, so compare
    # counts rather than decoded masks here.
    restored = read_submission(path)
    assert list(restored) == ["20150125172714Mh"]
    assert [counts_of(r) for r in restored["20150125172714Mh"]] == list(
        frame["segmentation_rle"]
    )


def test_csv_hostile_counts_are_rejected():
    with pytest.raises(ValueError):
        validate_counts("abc,def")


def test_images_without_predictions_contribute_no_rows():
    frame = build_submission({"a": [rle(0, 5, 0, 5)], "b": []})
    assert list(frame["filament_id"]) == ["a_1"]


# --- IoU --------------------------------------------------------------------


def test_iou_matrix_shape_and_values():
    gt = [rle(0, 10, 0, 10)]
    pred = [rle(0, 10, 0, 10), rle(0, 10, 0, 5)]
    ious = iou_matrix(gt, pred)
    assert ious.shape == (1, 2)
    assert ious[0, 0] == pytest.approx(1.0)
    assert ious[0, 1] == pytest.approx(0.5)  # 50 / 100


def test_iou_matrix_handles_empty():
    assert iou_matrix([], []).shape == (0, 0)
    assert iou_matrix([rle(0, 4, 0, 4)], []).shape == (1, 0)


# --- Panoptic Quality -------------------------------------------------------


def test_perfect_prediction_scores_one():
    segments = [rle(0, 10, 0, 10), rle(30, 40, 30, 40)]
    result = evaluate_image("img", segments, segments)
    assert (result.tp, result.fp, result.fn) == (2, 0, 0)
    assert result.pq == pytest.approx(1.0)


def test_empty_ground_truth_and_prediction_scores_one():
    assert evaluate_image("img", [], []).pq == pytest.approx(1.0)


def test_missed_filament_matches_the_formula():
    gt = [rle(0, 10, 0, 10), rle(30, 40, 30, 40)]
    pred = [rle(0, 10, 0, 10)]
    result = evaluate_image("img", gt, pred)
    assert (result.tp, result.fp, result.fn) == (1, 0, 1)
    # PQ = 1.0 / (1 + 0.5*0 + 0.5*1)
    assert result.pq == pytest.approx(1.0 / 1.5)


def test_near_miss_costs_twice_a_silent_miss():
    """The rule that drives post-processing: below IoU 0.5, don't predict."""
    gt = [rle(0, 10, 0, 10)]
    silent = evaluate_image("img", gt, [])
    near_miss = evaluate_image("img", gt, [rle(0, 10, 0, 4)])  # IoU 0.4
    assert silent.denominator == pytest.approx(0.5)
    assert near_miss.denominator == pytest.approx(1.0)
    assert silent.pq == near_miss.pq == 0.0


def test_matching_is_unique_above_threshold():
    gt = [rle(0, 10, 0, 10)]
    # Two overlapping predictions; only the better one may be matched.
    pred = [rle(0, 10, 0, 10), rle(0, 10, 0, 9)]
    result = evaluate_image("img", gt, pred)
    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    assert result.matched_ious[0] == pytest.approx(1.0)


def test_fragmentation_is_counted():
    gt = [rle(0, 10, 0, 40)]
    pred = [rle(0, 10, 0, 18), rle(0, 10, 22, 40)]  # one filament, split in two
    result = evaluate_image("img", gt, pred)
    assert result.one_to_many == 1
    assert result.tp == 0  # neither piece clears IoU 0.5


def test_over_merge_is_counted():
    gt = [rle(0, 10, 0, 18), rle(0, 10, 22, 40)]
    pred = [rle(0, 10, 0, 40)]  # two filaments, merged into one
    result = evaluate_image("img", gt, pred)
    assert result.many_to_one == 1


def test_dice_follows_from_iou():
    gt = [rle(0, 10, 0, 10)]
    pred = [rle(0, 10, 0, 8)]  # IoU = 0.8
    result = evaluate_image("img", gt, pred)
    assert result.matched_ious[0] == pytest.approx(0.8)
    assert result.matched_dices[0] == pytest.approx(2 * 0.8 / 1.8)


# --- Aggregation ------------------------------------------------------------


def test_pooled_and_per_image_aggregation_differ():
    gt = {"a": [rle(0, 10, 0, 10)], "b": [rle(0, 10, 0, 10), rle(30, 40, 30, 40)]}
    pred = {"a": [rle(0, 10, 0, 10)], "b": [rle(0, 10, 0, 10)]}
    summary, _ = evaluate(gt, pred)

    assert summary["tp"] == 2 and summary["fn"] == 1 and summary["fp"] == 0
    assert summary["pq_pooled"] == pytest.approx(2.0 / 2.5)
    assert summary["pq_per_image_mean"] == pytest.approx((1.0 + 1.0 / 1.5) / 2)
    assert summary["sq"] == pytest.approx(1.0)


def test_predictions_for_unknown_images_are_false_positives():
    gt = {"a": [rle(0, 10, 0, 10)]}
    pred = {"a": [rle(0, 10, 0, 10)], "ghost": [rle(0, 10, 0, 10)]}
    summary, _ = evaluate(gt, pred)
    assert summary["fp"] == 1
    assert summary["pq_pooled"] == pytest.approx(1.0 / 1.5)
