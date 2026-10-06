"""Cache a U-Net's full-resolution logits for the test or validation observations.

The expensive half of every pipeline, run on its own: one float16 map per
observation, written by ``filament_seg.logit_cache.ensure_logits`` together
with a record of the checkpoint, TTA and tiling that made it, so the fusion
scripts refuse a cache from another model. Several checkpoints are averaged as
an ensemble; caching each member separately and combining the caches with
scripts/average_logits.py gives the same ensemble and lets every member run on
its own GPU, which is how notebooks/reproduce.ipynb uses it.

    python scripts/cache_logits.py --checkpoint outputs/kaggle_v5/model_best_last.pt \\
        --subset test --out-dir outputs/step3/logits_v5last_dihedral
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from filament_seg.config import CACHE_DIR, SPLIT_PATH, TEST_IMAGE_DIR, TRAIN_ANNOTATIONS
from filament_seg.data import load_annotations, load_split, stems_of, test_image_paths
from filament_seg.dataset import load_disk_geometry
from filament_seg.logit_cache import ensure_logits
from filament_seg.model import TTA_MODES, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="one or more checkpoints; several are averaged as an ensemble")
    parser.add_argument("--subset", choices=["test", "val"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tta", choices=list(TTA_MODES), default="dihedral")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    parser.add_argument("--annotations", default=str(TRAIN_ANNOTATIONS))
    parser.add_argument("--split", default=str(SPLIT_PATH))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--limit", type=int, default=0, help="debug: first N observations")
    args = parser.parse_args()

    if args.subset == "test":
        stems = sorted(p.stem for p in test_image_paths(TEST_IMAGE_DIR))
    else:
        stems = stems_of(load_annotations(args.annotations), load_split(args.split)["val"])
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"no {args.subset} observations found")

    cache_dir = Path(args.cache_dir)
    disk_geometry = load_disk_geometry(cache_dir / "disk.json")
    uncached = [s for s in stems if s not in disk_geometry]
    if uncached:
        raise SystemExit(f"{len(uncached)}/{len(stems)} {args.subset} observations are missing from "
                         f"the preprocessing cache (e.g. {uncached[:3]}) -- run scripts/preprocess.py")
    ensure_logits(stems, args.checkpoint, disk_geometry, cache_dir / "flat", Path(args.out_dir),
                  args.tile, args.overlap, select_device(), args.force, tta=args.tta)
    print(f"logits for {len(stems)} {args.subset} observations in {args.out_dir}")


if __name__ == "__main__":
    main()
