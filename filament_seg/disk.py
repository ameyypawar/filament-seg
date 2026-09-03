"""Solar disk geometry and limb-darkening correction.

Two preprocessing steps that matter more than the choice of model:

1. **Disk masking.** Everything outside the solar limb is sky and detector
   noise. It is uniformly dark, so any "filaments are dark" rule fires on it.
   Masking it away removes a large class of false positives for free -- and
   under Panoptic Quality every false positive costs 0.5 in the denominator.

2. **Limb-darkening correction.** A full-disk H-alpha image is systematically
   darker toward the limb, so a single global darkness threshold is far too
   eager near the edge and far too shy at disk centre. Dividing by the median
   radial intensity profile makes one threshold mean the same thing everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Disk:
    cx: float
    cy: float
    radius: float

    def mask(self, shape: tuple[int, int], shrink: float = 0.98) -> np.ndarray:
        """Boolean mask of the disk, shrunk slightly to drop the noisy limb."""
        height, width = shape
        ys, xs = np.ogrid[:height, :width]
        r2 = (xs - self.cx) ** 2 + (ys - self.cy) ** 2
        return r2 <= (self.radius * shrink) ** 2

    def radius_map(self, shape: tuple[int, int]) -> np.ndarray:
        """Normalised radial distance (1.0 at the limb)."""
        height, width = shape
        ys, xs = np.ogrid[:height, :width]
        return np.sqrt((xs - self.cx) ** 2 + (ys - self.cy) ** 2) / max(self.radius, 1.0)


def read_image(path: str | Path) -> np.ndarray:
    """Load a GONG H-alpha JPEG as a single-channel uint8 array.

    The competition data is grayscale; reading it as BGR and averaging would
    only add rounding noise.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def detect_disk(image: np.ndarray) -> Disk:
    """Locate the solar disk.

    A full disk fills most of the frame, so thresholding the blurred image and
    taking the largest connected component is more robust than a Hough circle
    search (which is also far slower at 2048x2048).
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=5)
    threshold, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        # Degenerate image: fall back to the nominal full-frame disk.
        height, width = image.shape
        return Disk(width / 2, height / 2, min(height, width) * 0.45)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8)

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    return Disk(float(cx), float(cy), float(radius))


def limb_darkening_profile(
    image: np.ndarray, disk: Disk, n_bins: int = 128, smooth: int = 9
) -> np.ndarray:
    """Median intensity as a function of normalised radius."""
    radius = disk.radius_map(image.shape)
    inside = radius <= 1.0
    bins = np.clip((radius[inside] * n_bins).astype(np.int32), 0, n_bins - 1)
    values = image[inside].astype(np.float32)

    profile = np.ones(n_bins, dtype=np.float32)
    order = np.argsort(bins, kind="stable")
    bins_sorted, values_sorted = bins[order], values[order]
    edges = np.searchsorted(bins_sorted, np.arange(n_bins + 1))
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            profile[b] = np.median(values_sorted[lo:hi])

    # Fill empty bins, then smooth so the correction has no step artefacts.
    valid = profile > 0
    if valid.any():
        profile = np.interp(np.arange(n_bins), np.flatnonzero(valid), profile[valid])
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        profile = np.convolve(profile, kernel, mode="same")
    profile[profile <= 0] = 1.0
    return profile


def flatten(image: np.ndarray, disk: Disk, n_bins: int = 128) -> np.ndarray:
    """Divide out limb darkening.

    Returns a float32 image where 1.0 is "typical quiet Sun at this radius" and
    filaments sit clearly below 1.0, independent of distance from disk centre.
    """
    profile = limb_darkening_profile(image, disk, n_bins=n_bins)
    radius = disk.radius_map(image.shape)
    bin_index = np.clip((radius * n_bins).astype(np.int32), 0, n_bins - 1)
    expected = profile[bin_index]
    flat = image.astype(np.float32) / np.maximum(expected, 1e-3)
    flat[radius > 1.0] = 1.0
    return flat
