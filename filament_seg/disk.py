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
        return self.radius_window(0, 0, *shape)

    def radius_window(self, y0: int, x0: int, height: int, width: int) -> np.ndarray:
        """``radius_map`` for rows ``y0:y0+height`` and columns ``x0:x0+width`` only."""
        ys, xs = np.ogrid[y0 : y0 + height, x0 : x0 + width]
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


#: GONG H-alpha frames are usually registered to a standard geometry: the
#: disk centred and about 0.439 frame-widths in radius (899 px at 2048). Many
#: frames are not, so this is only one of several starting guesses.
_STANDARD_RADIUS_FRACTION = 0.439
#: Rays cast from a candidate centre when locating the limb.
_N_RAYS = 720
#: How far inside each starting guess to look as well, in px at 2048. The halo
#: can reach 150 px beyond the limb, so a guess locked onto it needs this.
_INWARD_OFFSETS = (0.0, 50.0, 100.0, 150.0)
#: A ray "agrees" with a circle if its sharpest edge lies within this many px
#: (at 2048) of it. Fixed, not data-adaptive, so scattered edges score low.
_AGREEMENT_TOLERANCE = 3.0
#: A circle counts as well supported if this fraction of the best circle's
#: agreeing rays also agree with it.
_SUPPORT_RATIO = 0.8


def detect_disk(image: np.ndarray) -> Disk:
    """Locate the solar disk by its limb.

    GONG frames carry a smooth "halo" annulus outside the limb -- scattered
    light, and the area the frame covered before it was registered -- that can
    be nearly as bright as the disk itself, and its outer border is a crisp
    circle too. Thresholding the image, which this function used to rely on
    alone, locks onto that border: about 35 px beyond the limb on a typical
    frame and up to 150 px on the worst.

    Instead, several starting circles (an intensity threshold, a texture
    threshold -- the disk is textured, the halo smooth -- and the standard GONG
    geometry), each also shrunk by up to 150 px, are snapped to the sharpest
    radial edge nearby and scored by how many rays agree with the result. The
    halo always lies outside the limb, so of the well-supported circles the
    smallest is the limb.
    """
    scale = min(image.shape) / 2048.0
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), max(1.5 * scale, 0.8))
    seeds = [_threshold_disk(image), _texture_disk(image), _standard_disk(image)]

    fits: list[tuple[Disk, float]] = []
    for seed in seeds:
        if seed is None:
            continue
        for offset in _INWARD_OFFSETS:
            start = Disk(seed.cx, seed.cy, seed.radius - offset * scale)
            if start.radius <= 0:
                continue
            disk, _ = _refine_limb(blurred, start, inside=60.0, outside=20.0)
            # A second, narrow pass lets the centre settle once the radius is close.
            fits.append(_refine_limb(blurred, disk, inside=10.0, outside=10.0))

    best_support = max((support for _, support in fits), default=0.0)
    # Fewer than half the rays agreeing means nothing found a clean limb; the
    # plain threshold circle is then the least surprising answer.
    if best_support < 0.5:
        return seeds[0]
    supported = [disk for disk, support in fits if support >= _SUPPORT_RATIO * best_support]
    return min(supported, key=lambda disk: disk.radius)


def _fit_circle(xs: np.ndarray, ys: np.ndarray, iterations: int = 4) -> tuple[Disk, np.ndarray]:
    """Least-squares (Kasa) circle fit, refitted without outliers.

    Returns the circle and a boolean inlier mask over the input points.
    """
    keep = np.ones(xs.size, dtype=bool)
    disk = Disk(0.0, 0.0, 1.0)
    for _ in range(iterations):
        if keep.sum() < 3:
            break
        design = np.column_stack([xs[keep], ys[keep], np.ones(int(keep.sum()))])
        target = -(xs[keep] ** 2 + ys[keep] ** 2)
        d, e, f = np.linalg.lstsq(design, target, rcond=None)[0]
        cx, cy = -d / 2.0, -e / 2.0
        disk = Disk(float(cx), float(cy), float(np.sqrt(max(cx * cx + cy * cy - f, 1.0))))
        residual = np.hypot(xs - disk.cx, ys - disk.cy) - disk.radius
        spread = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
        keep = np.abs(residual) <= max(3.0 * spread, 2.0)
    return disk, keep


def _refine_limb(
    blurred: np.ndarray, start: Disk, inside: float, outside: float
) -> tuple[Disk, float]:
    """Snap ``start`` to the sharpest radial edge within the search band.

    ``blurred`` is the lightly smoothed float image. Band lengths are in px at
    2048 and scale with the frame, so the same code works on small test
    images. Returns the refined circle and the fraction of all rays whose
    sharpest edge lies within ``_AGREEMENT_TOLERANCE`` of it.
    """
    height, width = blurred.shape
    scale = min(height, width) / 2048.0
    theta = np.linspace(0.0, 2.0 * np.pi, _N_RAYS, endpoint=False)[:, None]
    radii = np.arange(start.radius - inside * scale, start.radius + outside * scale + 1.0, 1.0)
    map_x = (start.cx + radii[None, :] * np.cos(theta)).astype(np.float32)
    map_y = (start.cy + radii[None, :] * np.sin(theta)).astype(np.float32)
    profiles = cv2.remap(blurred, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    in_frame = (map_x >= 1) & (map_x <= width - 2) & (map_y >= 1) & (map_y <= height - 2)
    gradient = np.abs(np.gradient(profiles, axis=1))
    gradient[~in_frame] = -1.0
    edge = np.argmax(gradient, axis=1)
    valid = gradient[np.arange(_N_RAYS), edge] > 0
    # A real edge is a peak inside the band. A maximum on the band's boundary
    # is just the smooth limb-darkening slope rising toward it -- perfectly
    # circular, so without this it would pass for a well-supported limb.
    valid &= (edge > 1) & (edge < radii.size - 2)
    if valid.sum() < 3:
        return start, 0.0
    # Sub-pixel peak position from a parabola through the peak and its two
    # neighbours: the flattening downstream is sensitive to fractions of a
    # pixel in the radius.
    rows = np.flatnonzero(valid)
    peak = edge[valid]
    before, at, after = (gradient[rows, peak - 1], gradient[rows, peak],
                         gradient[rows, peak + 1])
    curvature = before - 2.0 * at + after
    shift = np.divide(0.5 * (before - after), curvature,
                      out=np.zeros_like(at), where=curvature < 0)
    r = radii[peak] + np.clip(shift, -0.5, 0.5)
    xs = start.cx + r * np.cos(theta[valid, 0])
    ys = start.cy + r * np.sin(theta[valid, 0])
    disk, _ = _fit_circle(xs, ys)
    residual = np.hypot(xs - disk.cx, ys - disk.cy) - disk.radius
    agreeing = np.abs(residual) <= max(_AGREEMENT_TOLERANCE * scale, 1.0)
    return disk, float(agreeing.sum()) / _N_RAYS


def _standard_disk(image: np.ndarray) -> Disk:
    height, width = image.shape
    return Disk((width - 1) / 2.0, height / 2.0, min(height, width) * _STANDARD_RADIUS_FRACTION)


def _texture_disk(image: np.ndarray) -> Disk | None:
    """Disk from local texture: chromospheric structure is busy, the halo smooth."""
    scale = min(image.shape) / 2048.0
    img = image.astype(np.float32)
    detail = np.abs(img - cv2.GaussianBlur(img, (0, 0), max(3.0 * scale, 1.0)))
    texture = cv2.GaussianBlur(detail, (0, 0), max(15.0 * scale, 2.0))
    texture_u8 = np.clip(texture * (255.0 / max(float(texture.max()), 1e-6)), 0, 255)
    _, binary = cv2.threshold(texture_u8.astype(np.uint8), 0, 1,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8)
    # Fill the component's interior: smooth patches on the disk (and the whole
    # interior of a texture-free test image) leave holes.
    flood = component.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 1)
    component = component | (1 - flood)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(points) < 3:
        return None
    disk, _ = _fit_circle(points[:, 0], points[:, 1])
    return disk


def _threshold_disk(image: np.ndarray) -> Disk:
    """Disk from an intensity threshold: the brightest large component.

    Reliable for the centre of a clean frame, but the halo annulus usually
    passes the threshold too, so the radius it gives is an upper bound.
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
