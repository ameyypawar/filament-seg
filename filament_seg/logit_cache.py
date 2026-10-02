"""A disk cache of full-resolution model logits, safe against staleness.

Tiled inference with test-time augmentation is the expensive half of every
evaluation, and it does not depend on any post-processing setting, so its
output -- one 2048x2048 logit map per observation, float16 (8 MB) -- is cached
and reused by every grid point and every later experiment. The cache is keyed
by observation name alone, so it also records which model, TTA mode and tiling
produced it and refuses to mix them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .dataset import load_model_input
from .disk import Disk
from .model import ensemble_predict, load_trained


def cache_path(logit_dir: Path, stem: str) -> Path:
    return logit_dir / f"{stem}.npy"


def check_cache_provenance(
    logit_dir: Path,
    checkpoint_paths: list[str],
    tta: str,
    tile: int,
    overlap: int,
    force: bool,
) -> None:
    """Make sure every cached logit map was produced by this model and setup.

    The cache is keyed by observation name alone, so without this a sweep run
    against a new checkpoint, TTA mode or tiling would silently reuse stale
    logits and tune against the wrong predictions. ``force`` deletes the old
    maps *before* recording the new provenance: otherwise a forced run that
    died part-way would leave the previous model's maps under the new record,
    and the next plain run would accept them.
    """
    checkpoints = []
    for path in checkpoint_paths:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        checkpoints.append({"path": str(resolved), "size": stat.st_size,
                            "mtime": int(stat.st_mtime)})
    wanted = {"checkpoints": checkpoints, "tta": tta, "tile": tile, "overlap": overlap}

    logit_dir.mkdir(parents=True, exist_ok=True)
    meta_path = logit_dir / "cache_meta.json"
    cached = sorted(logit_dir.glob("*.npy"))
    if force:
        for path in cached:
            path.unlink()
    elif meta_path.exists():
        found = json.loads(meta_path.read_text(encoding="utf-8"))
        if found != wanted:
            raise SystemExit(
                f"{logit_dir} holds logits from a different model or setup:\n"
                f"  cached: {found}\n  wanted: {wanted}\n"
                "use a different --out-dir, or --force to recompute"
            )
    elif cached:
        raise SystemExit(
            f"{logit_dir} holds cached logits with no record of which model made them; "
            "use a different --out-dir, or --force to recompute"
        )
    meta_path.write_text(json.dumps(wanted, indent=2), encoding="utf-8")


def ensure_logits(
    stems: list[str],
    checkpoint_paths: list[str],
    disk_geometry: dict[str, Disk],
    flat_dir: Path,
    logit_dir: Path,
    tile: int,
    overlap: int,
    device: torch.device,
    force: bool,
    tta: str = "none",
) -> None:
    """Cache the (ensemble-averaged, TTA-averaged) logits for every stem, once."""
    check_cache_provenance(logit_dir, checkpoint_paths, tta, tile, overlap, force)
    pending = [s for s in stems if not cache_path(logit_dir, s).exists()]
    if not pending:
        print(f"logits: {len(stems)}/{len(stems)} already cached, skipping model load")
        return

    print(f"device: {device}")
    models = [load_trained(path, device) for path in checkpoint_paths]
    print(f"models: {len(models)}; computing logits for {len(pending)}/{len(stems)} observations")
    for n, stem in enumerate(pending, start=1):
        x = load_model_input(flat_dir, stem, disk_geometry[stem])
        if x is None:
            raise SystemExit(f"no cached flat image for {stem} -- run scripts/preprocess.py")
        logits = ensemble_predict(models, x, tta=tta, tile=tile, overlap=overlap, device=device)
        # Written under a temporary name and renamed, so an interrupted run can
        # never leave a truncated map that a later run would take as complete.
        partial = logit_dir / f"{stem}.npy.partial"
        with open(partial, "wb") as handle:
            np.save(handle, logits.astype(np.float16))
        os.replace(partial, cache_path(logit_dir, stem))
        if n % 5 == 0 or n == len(pending):
            print(f"  {n}/{len(pending)}", flush=True)
