"""Conversion between masks, COCO polygons and the competition's RLE format.

The submission asks for *RLE counts only* -- one row per predicted filament,
no ``size`` field, no quoting. The size is fixed at 2048x2048 for every image.

Two things bite people here:

1. ``pycocotools`` expects Fortran-ordered (column-major) arrays. Passing a
   C-ordered mask silently produces a transposed encoding that scores ~0.
2. ``counts`` comes back as ``bytes``; it must be decoded to ``str`` before it
   reaches the CSV.

The COCO counts alphabet spans ASCII 48..111 (``0``..``o``), which contains no
comma, quote or newline -- so a plain ``to_csv`` never adds quoting. We assert
this rather than trust it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .config import IMAGE_HEIGHT, IMAGE_SIZE, IMAGE_WIDTH

#: Characters that would force CSV quoting and therefore corrupt a submission.
_FORBIDDEN_IN_COUNTS = set(',"\'\r\n')

Rle = dict  # {"size": [h, w], "counts": bytes}


def mask_to_rle(mask: np.ndarray) -> Rle:
    """Encode a boolean/0-1 mask as a COCO RLE dict."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask_utils.encode(np.asfortranarray(mask))


def rle_to_mask(rle: Rle) -> np.ndarray:
    """Decode a COCO RLE dict to a boolean mask."""
    return mask_utils.decode(rle).astype(bool)


def mask_to_counts(mask: np.ndarray) -> str:
    """Encode a mask to the bare counts string expected in the submission."""
    return counts_of(mask_to_rle(mask))


def counts_of(rle: Rle) -> str:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return counts


def rle_from_counts(
    counts: str, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH
) -> Rle:
    """Rebuild a full RLE dict from a bare counts string."""
    return {"size": [height, width], "counts": counts.encode("ascii")}


def counts_to_mask(
    counts: str, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH
) -> np.ndarray:
    return rle_to_mask(rle_from_counts(counts, height, width))


def polygon_to_rle(
    polygons: Sequence[Sequence[float]],
    height: int = IMAGE_HEIGHT,
    width: int = IMAGE_WIDTH,
) -> Rle:
    """Convert a COCO ``segmentation`` field (list of flat polygons) to one RLE.

    MAGFiLO stores exactly one single-piece polygon per filament, but merging
    handles the general case without extra cost.
    """
    rles = mask_utils.frPyObjects([list(map(float, p)) for p in polygons], height, width)
    return mask_utils.merge(rles)


def labels_to_rles(labels: np.ndarray) -> list[Rle]:
    """Split an integer label map (0 = background) into one RLE per instance."""
    out: list[Rle] = []
    for value in np.unique(labels):
        if value == 0:
            continue
        out.append(mask_to_rle(labels == value))
    return out


def rle_areas(rles: Sequence[Rle]) -> np.ndarray:
    if not rles:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(mask_utils.area(list(rles)), dtype=np.float64)


def rle_intersection_area(a: Rle, b: Rle) -> float:
    """Exact intersection area of two RLEs, without decoding either to a mask."""
    return float(mask_utils.area(mask_utils.merge([a, b], intersect=1)))


def iou_matrix(gt_rles: Sequence[Rle], pred_rles: Sequence[Rle]) -> np.ndarray:
    """Pairwise IoU with shape ``(n_gt, n_pred)``.

    Computed in C by pycocotools, so it stays cheap even at 2048x2048.
    """
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    # mask_utils.iou(dt, gt, iscrowd) -> (n_dt, n_gt); transpose to (n_gt, n_pred).
    ious = mask_utils.iou(list(pred_rles), list(gt_rles), [0] * len(gt_rles))
    return np.asarray(ious, dtype=np.float64).T


# --- Submission I/O ---------------------------------------------------------


def make_filament_id(image_stem: str, index: int) -> str:
    """Build a submission id, e.g. ``20150125172714Mh_1`` (1-based index)."""
    return f"{image_stem}_{index}"


def validate_counts(counts: str) -> None:
    bad = _FORBIDDEN_IN_COUNTS.intersection(counts)
    if bad:
        raise ValueError(f"RLE counts contain CSV-hostile characters: {sorted(bad)}")


def build_submission(
    predictions: dict[str, Sequence[Rle]],
    validate: bool = True,
) -> pd.DataFrame:
    """Turn ``{image_stem: [rle, ...]}`` into the submission DataFrame.

    Images with no predicted filament simply contribute no rows.
    """
    rows: list[dict[str, str]] = []
    for stem in sorted(predictions):
        for i, rle in enumerate(predictions[stem], start=1):
            counts = counts_of(rle)
            if validate:
                validate_counts(counts)
            rows.append(
                {"filament_id": make_filament_id(stem, i), "segmentation_rle": counts}
            )
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if frame["filament_id"].duplicated().any():
        raise ValueError("duplicate filament_id values in submission")
    return frame


def write_submission(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def read_submission(path: str | Path) -> dict[str, list[Rle]]:
    """Read a submission CSV back into ``{image_stem: [rle, ...]}``."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    grouped: dict[str, list[Rle]] = {}
    for filament_id, counts in zip(frame["filament_id"], frame["segmentation_rle"]):
        stem = filament_id.rsplit("_", 1)[0]
        grouped.setdefault(stem, []).append(rle_from_counts(counts, *IMAGE_SIZE))
    return grouped


def iter_masks(rles: Iterable[Rle]) -> Iterable[np.ndarray]:
    for rle in rles:
        yield rle_to_mask(rle)
