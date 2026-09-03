"""Project-wide paths and constants.

Every path can be overridden with an environment variable so the same code runs
unchanged locally and inside a Kaggle notebook (where the data lives under
``/kaggle/input``).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Layout -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(
    os.environ.get("FILAMENT_DATA_ROOT", REPO_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026")
)
OUTPUT_ROOT = Path(os.environ.get("FILAMENT_OUTPUT_ROOT", REPO_ROOT / "outputs"))

TRAIN_DIR = DATA_ROOT / "train"
TRAIN_IMAGE_DIR = TRAIN_DIR / "train_images"
TRAIN_ANNOTATIONS = TRAIN_DIR / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TEST_IMAGE_DIR = DATA_ROOT / "test" / "test_images"

SPLIT_PATH = OUTPUT_ROOT / "splits.json"

# --- Dataset constants ------------------------------------------------------

#: All GONG H-alpha observations in this competition are 2048x2048 grayscale.
#: The submission format relies on this being fixed (RLE size is not submitted).
IMAGE_HEIGHT = 2048
IMAGE_WIDTH = 2048
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

#: Chirality classes exist in MAGFiLO but are irrelevant to this competition,
#: which is scored on masks only.
CATEGORY_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}

# --- Evaluation constants ---------------------------------------------------

#: Panoptic Quality matches a prediction to a ground-truth segment when their
#: IoU exceeds this value. Above 0.5 the matching is guaranteed unique.
PQ_IOU_THRESHOLD = 0.5

#: Minimum overlap fraction used when counting one-to-many / many-to-one
#: relations for the fragmentation diagnostics (part of the judging rubric).
FRAGMENT_OVERLAP_THRESHOLD = 0.10


def ensure_output_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT
