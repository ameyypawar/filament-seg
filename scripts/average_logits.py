"""Average cached logit maps of several models into the cache an ensemble of them would produce.

``filament_seg.model.ensemble_predict`` averages its members' TTA-averaged
logits with equal weights, so averaging the members' cached maps gives the
same ensemble without re-running any model. The output directory records the
members' checkpoints, in order, exactly as ``logit_cache.ensure_logits``
would for ``--checkpoint member1 member2 ...``, so the sweep and fusion
scripts accept it as that ensemble's cache.

    python scripts/average_logits.py --out outputs/step3/logits_ens2 \\
        outputs/step3/logits_v5last_dihedral outputs/step3/logits_u2seed1last_dihedral
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make `python scripts/foo.py` work from a fresh clone, with no install step.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("members", nargs="+", help="logit cache directories, one per model")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metas = [json.loads((Path(d) / "cache_meta.json").read_text(encoding="utf-8"))
             for d in args.members]
    setup = {key: metas[0][key] for key in ("tta", "tile", "overlap")}
    for directory, meta in zip(args.members, metas):
        if {key: meta[key] for key in setup} != setup:
            raise SystemExit(f"{directory} used a different TTA or tiling: {meta}")
    stems = sorted(set.intersection(*({p.stem for p in Path(d).glob("*.npy")}
                                      for d in args.members)))
    if not stems:
        raise SystemExit("the member caches have no observation in common")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.npy")):
        raise SystemExit(f"{out} already holds logits; pick a new directory")
    meta = {"checkpoints": [c for m in metas for c in m["checkpoints"]], **setup}
    (out / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for stem in stems:
        total = sum(np.load(Path(d) / f"{stem}.npy").astype(np.float32) for d in args.members)
        partial = out / f"{stem}.npy.partial"
        with open(partial, "wb") as handle:
            np.save(handle, (total / len(args.members)).astype(np.float16))
        os.replace(partial, out / f"{stem}.npy")
    print(f"wrote {out}: {len(stems)} observations averaged over {len(args.members)} models")


if __name__ == "__main__":
    main()
