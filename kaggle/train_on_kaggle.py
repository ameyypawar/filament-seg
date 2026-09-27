"""Train, tune and predict on a Kaggle GPU, end to end.

Pushed as a private batch script with

    kaggle kernels push -p kaggle/

It clones this repository, builds the preprocessing cache, trains the U-Net to
completion, sweeps post-processing on half of the validation observations,
checks the winner against the defaults on the other half, and writes a
submission for each. Locally there is only Apple MPS, where the same training
takes about four and a half hours; a Kaggle GPU does it in a fraction of that.

Large intermediates (the preprocessing cache, cached logits) live under /tmp so
they are not saved as notebook output. Only what is worth keeping is copied to
/kaggle/working, and the checkpoint is copied the moment training finishes, so
a failure in any later stage can never cost the trained model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_URL = "https://github.com/ameyypawar/filament-seg"
WORK = Path("/tmp/work")
REPO = WORK / "filament-seg"
OUT = WORK / "outputs"
KEEP = Path("/kaggle/working")

EPOCHS = 30
BATCH_SIZE = 16
#: Validation observations scored after every epoch to pick the checkpoint.
LIMIT_VAL = 48
#: Half of the 144 validation observations; the other half judges the result.
SWEEP_IMAGES = 72
#: 144 combinations, around the current defaults. The full default grid is 540,
#: which would spend hours of CPU on combinations far from anything sensible.
SWEEP_GRID = {
    "--threshold": ["0.3", "0.4", "0.5", "0.6"],
    "--min-area": ["200", "400"],
    "--bridge-gap": ["8", "12", "16"],
    "--close-radius": ["3", "5"],
    "--open-radius": ["0", "1", "2"],
}
#: PostprocessParams defaults plus the 0.5 probability threshold; the baseline
#: every tuned setting has to beat.
DEFAULTS = {"threshold": 0.5, "min_area": 400, "bridge_gap": 12,
            "close_radius": 5, "open_radius": 2}

STATUS: dict[str, str] = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(args: list, env: dict, log_to: Path | None = None) -> None:
    """Run a command inside the repo, streaming its output and optionally saving it."""
    args = [str(a) for a in args]
    log("$ " + " ".join(args))
    handle = open(log_to, "w", encoding="utf-8") if log_to else None
    try:
        proc = subprocess.Popen(args, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            if handle:
                handle.write(line)
        code = proc.wait()
    finally:
        if handle:
            handle.close()
    if code != 0:
        raise RuntimeError(f"exit code {code}: {' '.join(args)}")


def attempt(name: str, fn) -> bool:
    """Run a later stage without letting its failure take the others down."""
    try:
        fn()
        STATUS[name] = "ok"
        return True
    except Exception as error:  # noqa: BLE001 -- recorded and reported, never swallowed silently
        STATUS[name] = f"failed: {error}"
        log(f"stage '{name}' failed:\n{traceback.format_exc()}")
        return False


def keep(*paths: Path) -> None:
    for path in map(Path, paths):
        if path.exists():
            shutil.copy2(path, KEEP / path.name)
            log(f"kept {path.name}")


def find_data_root() -> Path:
    """Locate the competition data by structure, not by an assumed mount path."""
    for train_images in sorted(Path("/kaggle/input").rglob("train_images")):
        root = train_images.parent.parent
        if train_images.is_dir() and (root / "test" / "test_images").is_dir():
            return root
    raise SystemExit("competition data not found under /kaggle/input: "
                     "is the competition attached to this notebook?")


def write_summary(**extra) -> None:
    (KEEP / "run_summary.json").write_text(
        json.dumps({"stages": STATUS, **extra}, indent=2, default=str), encoding="utf-8")


def main() -> None:
    KEEP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- fail fast on anything that would waste a long run --------------------
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no GPU in this session; refusing to train for hours on CPU")
    gpu = torch.cuda.get_device_name(0)
    log(f"GPU: {gpu} | torch {torch.__version__} | CPUs: {os.cpu_count()}")

    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    log(f"cloned {REPO_URL} at {commit}")

    try:
        import segmentation_models_pytorch  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "segmentation-models-pytorch"], check=True)

    data_root = find_data_root()
    log(f"data: {data_root}")
    env = {**os.environ, "FILAMENT_DATA_ROOT": str(data_root),
           "FILAMENT_OUTPUT_ROOT": str(OUT), "PYTHONUNBUFFERED": "1"}
    workers = min(4, os.cpu_count() or 2)
    py = sys.executable
    write_summary(commit=commit, gpu=gpu)

    # --- 1. cache, split, train: everything after depends on these -------------
    run([py, "scripts/preprocess.py", "--workers", workers], env)
    run([py, "scripts/make_splits.py", "--group-by", "date"], env)
    checkpoint = OUT / "model_best.pt"
    started = time.time()
    run([py, "scripts/train.py", "--epochs", EPOCHS, "--batch-size", BATCH_SIZE,
         "--workers", workers, "--limit-val", LIMIT_VAL, "--out", checkpoint],
        env, log_to=OUT / "train.log")
    STATUS["train"] = f"ok ({(time.time() - started) / 60:.0f} min)"
    keep(checkpoint, OUT / "model_best_last.pt", OUT / "train.log", OUT / "splits.json")
    write_summary(commit=commit, gpu=gpu)

    # --- 2. tune post-processing on half of the validation observations -------
    def sweep() -> None:
        args = [py, "scripts/sweep_postprocess.py", "--checkpoint", checkpoint,
                "--device", "cuda", "--n-images", SWEEP_IMAGES, "--workers", workers]
        for flag, values in SWEEP_GRID.items():
            args += [flag, *values]
        run(args, env, log_to=OUT / "sweep.log")
        keep(OUT / "sweep_postprocess.json", OUT / "sweep.log")

    tuned = dict(DEFAULTS)
    if attempt("sweep", sweep):
        best = json.loads((OUT / "sweep_postprocess.json").read_text())[0]
        tuned = {key: best[key] for key in DEFAULTS}
        log(f"sweep winner: {tuned} (PQ {best['pq']:.4f} on the sweep's own images)")

    # --- 3. predict validation and test with each setting ---------------------
    variants = {"default": DEFAULTS}
    if tuned != DEFAULTS:
        variants["tuned"] = tuned

    def predict(subset: str, settings: dict, filename: str) -> None:
        run([py, "scripts/predict.py", "--checkpoint", checkpoint, "--subset", subset,
             "--threshold", settings["threshold"], "--min-area", settings["min_area"],
             "--bridge-gap", settings["bridge_gap"], "--close-radius", settings["close_radius"],
             "--open-radius", settings["open_radius"], "--out", OUT / filename], env)
        keep(OUT / filename)

    for label, settings in variants.items():
        attempt(f"predict val ({label})", lambda: predict("val", settings, f"val_{label}.csv"))
        attempt(f"predict test ({label})",
                lambda: predict("test", settings, f"submission_{label}.csv"))

    # --- 4. judge tuned against default on the half the sweep never saw -------
    def compare() -> None:
        args = [py, "scripts/compare_holdout.py", "--sweep-images", SWEEP_IMAGES,
                "--out", OUT / "holdout_comparison.json"]
        for label in variants:
            if (OUT / f"val_{label}.csv").exists():
                args += ["--submission", OUT / f"val_{label}.csv"]
        run(args, env, log_to=OUT / "holdout.log")
        keep(OUT / "holdout_comparison.json", OUT / "holdout.log")

    attempt("holdout comparison", compare)
    write_summary(commit=commit, gpu=gpu, epochs=EPOCHS, defaults=DEFAULTS, tuned=tuned)
    log(f"done: {STATUS}")


if __name__ == "__main__":
    main()
