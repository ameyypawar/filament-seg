"""Train, tune and predict on a Kaggle GPU, end to end.

Pushed as a private batch script with

    kaggle kernels push -p kaggle/

It clones this repository, builds the preprocessing cache, trains the U-Net to
completion, sweeps post-processing on half of the validation observations, and
predicts validation and test with the winning settings; the other half of the
validation set is kept for judging the result against earlier runs. Locally
there is only Apple MPS, where the same training takes about four and a half
hours; a Kaggle GPU does it in a fraction of that.

TARGET and TTA below select the experiment. Run 1 (annotator labels, no TTA)
scored 0.30 public, 0.32 once 8-orientation TTA was added locally; this
configuration trains on the consensus of all annotators and applies the same
TTA on the GPU, so its result is directly comparable to that 0.32.

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
#: "consensus" trains against the per-observation average of every annotator's
#: mask instead of one arbitrarily chosen annotator (inter-annotator PQ is only
#: 0.34, so a single annotator's labels carry a lot of personal noise).
TARGET = "consensus"
#: Test-time augmentation for the sweep and the final predictions: all eight
#: orientations, which is what lifted run 1 from 0.30 to 0.32 public.
TTA = "dihedral"
#: Validation observations scored after every epoch to pick the checkpoint.
LIMIT_VAL = 48
#: Half of the 144 validation observations; the other half judges the result.
SWEEP_IMAGES = 72
#: Centred on what won with TTA in run 1 (threshold 0.7, open 0, bridge 16),
#: with the threshold range widened: soft consensus targets change how the
#: model's probabilities are calibrated, so the best cut-off may move.
SWEEP_GRID = {
    "--threshold": ["0.4", "0.5", "0.6", "0.7", "0.8"],
    "--min-area": ["400"],
    "--bridge-gap": ["12", "16"],
    "--close-radius": ["3"],
    "--open-radius": ["0", "1"],
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


def environment_report() -> dict:
    """What this session actually got, logged before any decision is made on it.

    "No GPU" has two very different causes -- no accelerator assigned at all
    (typically an account that isn't phone-verified, or no GPU quota left), or a
    GPU that is present but too old for the installed PyTorch build -- and a bare
    "no GPU" message cannot tell them apart. Environment variables whose names
    suggest credentials are never printed.
    """
    import torch
    import urllib.request

    report: dict = {"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count()}
    smi = shutil.which("nvidia-smi")
    report["nvidia_smi"] = (subprocess.run([smi, "-L"], capture_output=True, text=True)
                            .stdout.strip() or "present, lists no devices") if smi else "not installed"
    if torch.cuda.device_count():
        major, minor = torch.cuda.get_device_capability(0)
        report["device_capability"] = f"sm_{major}{minor}"
        report["torch_arch_list"] = torch.cuda.get_arch_list()
    secret = ("TOKEN", "SECRET", "KEY", "PASS", "CRED", "AUTH")
    report["accelerator_env"] = {
        k: v for k, v in os.environ.items()
        if any(s in k for s in ("GPU", "ACCELERATOR", "CUDA", "NVIDIA"))
        and not any(s in k for s in secret)
    }
    try:
        urllib.request.urlopen("https://github.com", timeout=10)
        report["internet"] = "ok"
    except Exception as error:  # noqa: BLE001
        report["internet"] = f"unavailable ({error.__class__.__name__})"
    for key, value in report.items():
        log(f"env | {key}: {value}")
    return report


def write_summary(**extra) -> None:
    (KEEP / "run_summary.json").write_text(
        json.dumps({"stages": STATUS, **extra}, indent=2, default=str), encoding="utf-8")


def main() -> None:
    KEEP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- fail fast on anything that would waste a long run --------------------
    import torch
    report = environment_report()
    write_summary(environment=report)
    if not torch.cuda.is_available():
        cause = ("a GPU is present but this PyTorch build cannot use it"
                 if "GPU" in str(report["nvidia_smi"]) else
                 "no GPU was assigned to this session")
        raise SystemExit(f"no usable GPU: {cause}; refusing to train for hours on CPU")
    if report["internet"] != "ok":
        raise SystemExit("no internet in this session, so the repository cannot be cloned")
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
         "--workers", workers, "--limit-val", LIMIT_VAL, "--target", TARGET,
         "--out", checkpoint],
        env, log_to=OUT / "train.log")
    STATUS["train"] = f"ok ({(time.time() - started) / 60:.0f} min)"
    keep(checkpoint, OUT / "model_best_last.pt", OUT / "train.log", OUT / "splits.json")
    write_summary(commit=commit, gpu=gpu)

    # --- 2. tune post-processing on half of the validation observations -------
    def sweep() -> None:
        args = [py, "scripts/sweep_postprocess.py", "--checkpoint", checkpoint,
                "--device", "cuda", "--n-images", SWEEP_IMAGES, "--workers", workers,
                "--tta", TTA]
        for flag, values in SWEEP_GRID.items():
            args += [flag, *values]
        run(args, env, log_to=OUT / "sweep.log")
        keep(OUT / "sweep_postprocess.json", OUT / "sweep.log")

    tuned = dict(DEFAULTS)
    if attempt("sweep", sweep):
        best = json.loads((OUT / "sweep_postprocess.json").read_text())[0]
        tuned = {key: best[key] for key in DEFAULTS}
        log(f"sweep winner: {tuned} (PQ {best['pq']:.4f} on the sweep's own images)")

    # --- 3. predict validation and test with the winning settings -------------
    # Only the winner: with TTA every image is predicted eight times, and the
    # comparison that matters -- against the 0.32 run -- is made on the held-out
    # validation half after download, not against this run's own defaults.
    name = f"{TARGET}_{TTA}"

    def predict(subset: str, filename: str) -> None:
        run([py, "scripts/predict.py", "--checkpoint", checkpoint, "--subset", subset,
             "--tta", TTA,
             "--threshold", tuned["threshold"], "--min-area", tuned["min_area"],
             "--bridge-gap", tuned["bridge_gap"], "--close-radius", tuned["close_radius"],
             "--open-radius", tuned["open_radius"], "--out", OUT / filename], env)
        keep(OUT / filename)

    attempt("predict val", lambda: predict("val", f"val_{name}.csv"))
    attempt("predict test", lambda: predict("test", f"submission_{name}.csv"))

    # --- 4. score it on the half the sweep never saw ---------------------------
    def compare() -> None:
        run([py, "scripts/compare_holdout.py", "--sweep-images", SWEEP_IMAGES,
             "--out", OUT / "holdout_comparison.json",
             "--submission", OUT / f"val_{name}.csv"], env, log_to=OUT / "holdout.log")
        keep(OUT / "holdout_comparison.json", OUT / "holdout.log")

    attempt("holdout comparison", compare)
    write_summary(commit=commit, gpu=gpu, epochs=EPOCHS, target=TARGET, tta=TTA,
                  defaults=DEFAULTS, tuned=tuned)
    log(f"done: {STATUS}")


if __name__ == "__main__":
    main()
