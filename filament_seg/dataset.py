"""Cropped-tile dataset for training the filament segmentation network.

Two things shape this dataset more than anything else:

* **Filaments are rare.** They cover roughly 0.2% of disk pixels (median area
  1228 px in a 2048x2048 frame), so a uniformly random crop is empty the
  overwhelming majority of the time. ``filament_bias`` fixes this by centring
  most crops on an actual ground-truth filament, jittered so the network
  still has to localise rather than just find the middle of the crop.

* **Native resolution only.** Filaments are a few pixels wide; anything that
  resamples the image -- including the elastic/scale augmentation that is
  standard for most segmentation tasks -- destroys the structures the metric
  is graded on. Only flips, 90-degree rotations and brightness/contrast
  jitter are used here.

Everything expensive and parameter-independent (disk detection, limb
darkening) is expected to already live in ``scripts/preprocess.py``'s cache;
this module only reads that cache and crops it.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from .data import Annotations, deduplicate_by_file
from .disk import Disk

#: Flips and 90-degree rotations preserve every pixel exactly (no
#: interpolation), which is why they're safe for structures a few pixels
#: wide. Brightness/contrast jitter only ever touches the "image" target
#: below, never the "radius" one -- see the ``additional_targets`` note.
_AUGMENT = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ],
    # albumentations only applies colour transforms to "image"-typed targets;
    # tagging the radius map as "mask" gets it the flip/rotation but keeps it
    # immune to brightness/contrast, which would corrupt its meaning (it's a
    # normalised coordinate, not an intensity).
    additional_targets={"radius": "mask"},
)


def load_disk_geometry(path: str | Path) -> dict[str, Disk]:
    """Load ``scripts/preprocess.py``'s ``disk.json`` as ``{stem: Disk}``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {stem: Disk(**geometry) for stem, geometry in raw.items()}


def _instance_centroids(annotations: Annotations, image_id: str) -> list[tuple[float, float]]:
    """Bounding-box centre of each ground-truth filament in ``image_id``.

    A bbox centre is not the exact centroid of a curved ribbon, but it is a
    fine jitter target and, unlike a full decode, stays cheap even for the
    handful of images with two dozen instances.
    """
    centroids = []
    for rle in annotations.gt_rles(image_id):
        x, y, w, h = mask_utils.toBbox(rle)
        centroids.append((float(y + h / 2.0), float(x + w / 2.0)))
    return centroids


@dataclass
class _ImageCacheEntry:
    """Per-observation state shared by every crop drawn from that image."""

    #: Cached flat intensity, uint8, as written by scripts/preprocess.py.
    flat_u8: np.ndarray
    #: Normalised radius map (see ``Disk.radius_map``), one value per pixel.
    radius: np.ndarray


class _BoundedCache:
    """Tiny LRU cache, bounded so a full epoch cannot pin every observation's
    2048x2048 radius map in memory at once (707 stems x 16 MB would be ~11 GB).
    A handful of slots is enough to absorb the ``crops_per_image`` repeats of
    whichever image the sampler happens to be drawing from right now.
    """

    def __init__(self, maxsize: int = 8) -> None:
        self._maxsize = maxsize
        self._data: "OrderedDict[str, _ImageCacheEntry]" = OrderedDict()

    def get(self, key: str) -> _ImageCacheEntry | None:
        entry = self._data.get(key)
        if entry is not None:
            self._data.move_to_end(key)
        return entry

    def put(self, key: str, entry: _ImageCacheEntry) -> None:
        self._data[key] = entry
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)


class FilamentCrops(Dataset):
    """Random crops from cached flattened images, biased toward filaments.

    ``target="annotator"`` (the default) trains against whichever single
    annotator's view was selected -- the original behaviour. ``target=
    "consensus"`` trains against ``scripts/preprocess.py``'s per-stem soft
    consensus mask instead: MAGFiLO's measured inter-annotator Panoptic
    Quality is only 0.343, so one annotator's view carries a lot of
    annotator-specific noise that a consensus average regresses out. The
    consensus mask is keyed by observation stem, not by view, so under this
    target ``image_ids`` is deduplicated to one view per stem -- the target
    no longer depends on which view was picked, only the crop geometry does.
    """

    def __init__(
        self,
        image_ids: Sequence[str],
        annotations: Annotations,
        cache_dir: str | Path,
        crop_size: int = 512,
        crops_per_image: int = 8,
        filament_bias: float = 0.7,
        augment: bool = True,
        seed: int = 0,
        target: str = "annotator",
    ) -> None:
        if target not in ("annotator", "consensus"):
            raise ValueError(f"unknown target: {target!r}")

        self.annotations = annotations
        self.cache_dir = Path(cache_dir)
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.filament_bias = filament_bias
        self.augment = augment
        self.seed = seed
        self.target = target

        image_ids = list(image_ids)
        if target == "consensus":
            image_ids = deduplicate_by_file(annotations, image_ids, seed=seed)
        self.image_ids = image_ids

        self._flat_dir = self.cache_dir / "flat"
        self._mask_dir = self.cache_dir / "mask"
        self._consensus_dir = self.cache_dir / "consensus"
        self._disk = load_disk_geometry(self.cache_dir / "disk.json")
        self._centroids: dict[str, list[tuple[float, float]]] = {
            image_id: _instance_centroids(annotations, image_id) for image_id in self.image_ids
        }

        self._cache = _BoundedCache()
        # One RNG per DataLoader worker process -- see _rng().
        self._rngs: dict[int, random.Random] = {}

    def __len__(self) -> int:
        return len(self.image_ids) * self.crops_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[index // self.crops_per_image]
        record = self.annotations.images[image_id]
        stem = record.stem
        shape = (record.height, record.width)
        rng = self._rng()

        entry = self._image_entry(stem, shape)
        y0, x0 = self._sample_crop_origin(rng, image_id, stem, shape)
        size = self.crop_size

        flat_crop = entry.flat_u8[y0 : y0 + size, x0 : x0 + size].astype(np.float32) / 255.0
        radius_crop = entry.radius[y0 : y0 + size, x0 : x0 + size]
        mask_crop = self._load_mask(image_id, stem, shape)[
            y0 : y0 + size, x0 : x0 + size
        ].astype(np.float32)

        if self.augment:
            out = _AUGMENT(image=flat_crop, mask=mask_crop, radius=radius_crop)
            flat_crop, mask_crop, radius_crop = out["image"], out["mask"], out["radius"]

        x = np.stack([flat_crop, radius_crop], axis=0).astype(np.float32)
        y = mask_crop[None, :, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    # --- internals -----------------------------------------------------

    def _rng(self) -> random.Random:
        """Per-worker RNG.

        ``self`` -- and any ``random.Random`` already stored on it -- is
        copied wholesale into each DataLoader worker process at fork time;
        keying by worker id and creating the generator lazily means sibling
        workers don't replay an identical crop sequence.
        """
        worker = torch.utils.data.get_worker_info()
        key = 0 if worker is None else worker.id + 1
        rng = self._rngs.get(key)
        if rng is None:
            rng = random.Random(self.seed + key)
            self._rngs[key] = rng
        return rng

    def _image_entry(self, stem: str, shape: tuple[int, int]) -> _ImageCacheEntry:
        entry = self._cache.get(stem)
        if entry is not None:
            return entry

        flat_path = self._flat_dir / f"{stem}.png"
        flat_u8 = cv2.imread(str(flat_path), cv2.IMREAD_GRAYSCALE)
        if flat_u8 is None:
            raise FileNotFoundError(
                f"missing cached flat image: {flat_path} -- run scripts/preprocess.py first"
            )
        radius = self._disk[stem].radius_map(shape).astype(np.float32)
        entry = _ImageCacheEntry(flat_u8=flat_u8, radius=radius)
        self._cache.put(stem, entry)
        return entry

    def _load_mask(self, image_id: str, stem: str, shape: tuple[int, int]) -> np.ndarray:
        if self.target == "consensus":
            consensus_path = self._consensus_dir / f"{stem}.png"
            mask = cv2.imread(str(consensus_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(
                    f"missing cached consensus mask: {consensus_path} -- "
                    "run scripts/preprocess.py first"
                )
            # Soft agreement fraction, not thresholded -- 255 means every
            # annotator of this stem agreed, not just a majority.
            return mask.astype(np.float32) / 255.0

        mask_path = self._mask_dir / f"{image_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # No cached mask just means "no filaments annotated in this
            # view" -- scripts/preprocess.py never writes all-zero PNGs for
            # nothing, but an all-zero view is a valid (if rare) outcome.
            return np.zeros(shape, dtype=np.uint8)
        return (mask > 0).astype(np.uint8)

    def _sample_crop_origin(
        self, rng: random.Random, image_id: str, stem: str, shape: tuple[int, int]
    ) -> tuple[int, int]:
        """Pick the crop's top-left corner, in image coordinates."""
        height, width = shape
        size = self.crop_size
        centroids = self._centroids[image_id]

        if centroids and rng.random() < self.filament_bias:
            cy, cx = rng.choice(centroids)
            jitter = size / 4.0
            cy += rng.uniform(-jitter, jitter)
            cx += rng.uniform(-jitter, jitter)
        else:
            # Uniform over the disk's *area*, not its bounding box -- sampling
            # (angle, radius) uniformly would crowd crops toward the centre.
            disk = self._disk[stem]
            angle = rng.uniform(0.0, 2.0 * np.pi)
            r = disk.radius * (rng.random() ** 0.5)
            cy = disk.cy + r * np.sin(angle)
            cx = disk.cx + r * np.cos(angle)

        y0 = int(np.clip(cy - size / 2.0, 0, max(height - size, 0)))
        x0 = int(np.clip(cx - size / 2.0, 0, max(width - size, 0)))
        return y0, x0
