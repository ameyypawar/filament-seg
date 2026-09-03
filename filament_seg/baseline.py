"""A model-free filament detector.

Purpose: get a *valid submission on the leaderboard before any training starts*,
so the RLE encoding, the CSV format and the local evaluator are all proven
against the real scorer. It is also the honest floor that any learned model has
to beat.

Method -- essentially the classical H-alpha filament recipe:

1. locate the solar disk and mask off everything outside it
2. divide out limb darkening so one threshold works at any radius
3. call a pixel a filament when it is ``k`` standard deviations darker than the
   quiet Sun around it
4. morphological cleanup, gap-bridging and an area filter to form instances

Expect a modest PQ. The value is in the plumbing, not the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .disk import Disk, detect_disk, flatten, read_image
from .postprocess import PostprocessParams, binary_to_instances
from .rle import Rle, labels_to_rles


@dataclass
class BaselineParams:
    """Thresholding knobs. Tune ``k`` first -- it dominates the score."""

    #: Darkness threshold in standard deviations below the quiet-Sun level.
    k: float = 1.6
    #: Shrink factor applied to the disk radius; the limb itself is noisy.
    disk_shrink: float = 0.96
    #: Gaussian smoothing (in pixels) applied before thresholding.
    blur_sigma: float = 2.0
    postprocess: PostprocessParams = field(default_factory=PostprocessParams)


def segment(
    image: np.ndarray, params: BaselineParams | None = None
) -> tuple[np.ndarray, Disk]:
    """Return an integer instance-label map and the detected disk."""
    params = params or BaselineParams()

    disk = detect_disk(image)
    disk_mask = disk.mask(image.shape, shrink=params.disk_shrink)

    flat = flatten(image, disk)
    if params.blur_sigma > 0:
        flat = cv2.GaussianBlur(flat, (0, 0), sigmaX=params.blur_sigma)

    inside = flat[disk_mask]
    if inside.size == 0:
        return np.zeros(image.shape, dtype=np.int32), disk

    # Robust centre/spread: filaments are a small, dark minority of disk pixels,
    # so the median and the MAD are barely moved by them.
    median = float(np.median(inside))
    mad = float(np.median(np.abs(inside - median)))
    sigma = 1.4826 * mad if mad > 0 else float(inside.std())
    threshold = median - params.k * max(sigma, 1e-6)

    binary = (flat < threshold) & disk_mask
    labels = binary_to_instances(binary, params.postprocess, restrict_to=disk_mask)
    return labels, disk


def predict_rles(
    image_path: str | Path, params: BaselineParams | None = None
) -> list[Rle]:
    """Predict filament instances for one image file, as RLEs."""
    labels, _ = segment(read_image(image_path), params)
    return labels_to_rles(labels)
