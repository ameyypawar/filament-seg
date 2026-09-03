"""Loading MAGFiLO annotations and building leak-free train/validation splits.

The dataset has one structural trap that quietly inflates validation scores:

* **The same observation is annotated by several annotators independently.** They
  appear as distinct ``image["id"]`` values that differ only in the batch prefix
  (``010101-2016...`` vs ``010102-2016...``) but share a ``file_name``. Splitting
  on ``image_id`` puts the same pixels in both train and validation.

* **Filaments live for days to weeks.** Two observations from the same day (or
  the same week) are near-duplicates. Splitting randomly by file name still
  leaks. ``group_by="date"`` is the default here; ``"month"`` is the stricter
  option and gives a more honest estimate of generalisation at the cost of a
  pessimistic bias relative to the leaderboard.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN_ANNOTATIONS
from .rle import Rle, polygon_to_rle

GroupBy = str  # "file" | "date" | "month"


@dataclass(frozen=True)
class ImageRecord:
    """One annotated view of one observation."""

    image_id: str  # e.g. "010401-20160920230134Lh"
    file_name: str  # e.g. "20160920230134Lh.jpg"
    width: int
    height: int

    @property
    def stem(self) -> str:
        """Observation name without extension, e.g. ``20160920230134Lh``."""
        return Path(self.file_name).stem

    @property
    def annotator(self) -> str:
        """Annotator batch prefix, or ``""`` when the id carries none."""
        return self.image_id.split("-", 1)[0] if "-" in self.image_id else ""

    @property
    def date(self) -> str:
        return self.stem[:8]

    @property
    def month(self) -> str:
        return self.stem[:6]

    @property
    def observatory(self) -> str:
        """Two-letter GONG site code, e.g. ``Bh`` (Big Bear), ``Lh`` (Learmonth)."""
        return self.stem[14:16]

    def group_key(self, group_by: GroupBy = "date") -> str:
        return {"file": self.stem, "date": self.date, "month": self.month}[group_by]


@dataclass
class Annotations:
    """Parsed COCO file, indexed the ways this project actually needs."""

    images: dict[str, ImageRecord]
    annotations_by_image: dict[str, list[dict]]
    categories: dict[int, str]

    def __len__(self) -> int:
        return len(self.images)

    @property
    def image_ids(self) -> list[str]:
        return sorted(self.images)

    def by_file_stem(self) -> dict[str, list[str]]:
        """``{observation stem: [image_id per annotator]}``."""
        grouped: dict[str, list[str]] = {}
        for image_id, record in self.images.items():
            grouped.setdefault(record.stem, []).append(image_id)
        return {k: sorted(v) for k, v in sorted(grouped.items())}

    def gt_rles(self, image_id: str) -> list[Rle]:
        """Ground-truth instance RLEs for one annotated image."""
        record = self.images[image_id]
        out: list[Rle] = []
        for ann in self.annotations_by_image.get(image_id, []):
            segmentation = ann.get("segmentation")
            if not segmentation:
                continue
            out.append(polygon_to_rle(segmentation, record.height, record.width))
        return out

    def gt_dict(self, image_ids: Iterable[str]) -> dict[str, list[Rle]]:
        return {image_id: self.gt_rles(image_id) for image_id in image_ids}


def load_annotations(path: str | Path = TRAIN_ANNOTATIONS) -> Annotations:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    images = {}
    for entry in raw.get("images", []):
        images[str(entry["id"])] = ImageRecord(
            image_id=str(entry["id"]),
            file_name=entry["file_name"],
            width=int(entry.get("width") or IMAGE_WIDTH),
            height=int(entry.get("height") or IMAGE_HEIGHT),
        )

    annotations_by_image: dict[str, list[dict]] = {}
    for ann in raw.get("annotations", []):
        annotations_by_image.setdefault(str(ann["image_id"]), []).append(ann)

    categories = {int(c["id"]): c["name"] for c in raw.get("categories", [])}
    return Annotations(images, annotations_by_image, categories)


def make_split(
    annotations: Annotations,
    val_fraction: float = 0.2,
    group_by: GroupBy = "date",
    seed: int = 0,
) -> dict[str, list[str]]:
    """Split image ids into train/val so that no group spans both sides.

    Returns ``{"train": [...], "val": [...], "group_by": ...}`` with image ids
    (annotator-prefixed), which is what the loaders and the evaluator key on.
    """
    groups: dict[str, list[str]] = {}
    for image_id, record in annotations.images.items():
        groups.setdefault(record.group_key(group_by), []).append(image_id)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_val = max(1, round(len(keys) * val_fraction))
    val_keys = set(keys[:n_val])

    train_ids = sorted(i for k in keys if k not in val_keys for i in groups[k])
    val_ids = sorted(i for k in keys if k in val_keys for i in groups[k])
    return {
        "group_by": group_by,
        "seed": seed,
        "val_fraction": val_fraction,
        "n_groups": len(keys),
        "train": train_ids,
        "val": val_ids,
    }


def deduplicate_by_file(
    annotations: Annotations, image_ids: Sequence[str], seed: int = 0
) -> list[str]:
    """Keep one annotator per observation.

    Useful for training (the duplicates are label noise, not extra signal) and
    for validation (scoring against every annotator double-counts easy images).
    """
    rng = random.Random(seed)
    chosen: list[str] = []
    grouped: dict[str, list[str]] = {}
    for image_id in image_ids:
        grouped.setdefault(annotations.images[image_id].stem, []).append(image_id)
    for stem in sorted(grouped):
        chosen.append(rng.choice(sorted(grouped[stem])))
    return sorted(chosen)


def test_image_paths(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    patterns = ("*.jpeg", "*.jpg", "*.JPEG", "*.JPG", "*.png")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(directory.glob(pattern))
    return sorted(set(paths))
