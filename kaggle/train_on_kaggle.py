"""Train U-Nets on a Kaggle GPU, and optionally tune and predict with one of them.

Pushed as a private batch script with

    kaggle kernels push -p kaggle/

It clones this repository, builds the preprocessing cache and trains every
entry of VARIANTS -- side by side, one per GPU, when the session has enough of
them (a T4 x2 session has two). Locally there is only Apple MPS, where one
training run takes about four and a half hours.

With EXPLORE on, the session stops there: the U-Net is judged inside
detector-guided fusion, which is tuned on the Mac from logits cached with the
downloaded checkpoints, so a post-processing sweep here would measure the wrong
pipeline. With EXPLORE off and a single variant, it also sweeps U-Net-only
post-processing on half of the validation observations, predicts validation and
test with the winner and scores it on the other half, as runs 1 to 3 did.

Large intermediates (the preprocessing cache, cached logits) live under /tmp so
they are not saved as notebook output. The checkpoints are written straight to
/kaggle/working by train.py after every epoch, so neither a crash in a later
stage nor one in the last epochs of training (out of memory, the session time
limit) can cost the best model trained so far.
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
#: This session's training runs: a name and the scripts/train.py flags that set
#: it apart from run 3's recipe (ResNet34, one annotator's view per
#: observation, Dice + BCE). Run 4 trained "views_all" (every annotator's view,
#: "--crops-per-image 5" to keep run 3's ~4.5k crops an epoch) and "union";
#: run 5 "seed1" and "tversky" (--fn-weight 0.7). In fusion none beat run 3,
#: and seed 1 of run 3's own recipe scored 0.007 below it: the spread between
#: seeds is as large as any of those changes. So the recipe stays, and this
#: run trains it on every labelled observation, twice, as candidates for the
#: final entry; validation observations are in training, so nothing is scored.
#: Seeds 0 and 1 ran first (kernel v8); seeds 2 and 3 make a four-model ensemble.
VARIANTS = [
    ("alldata_s2", ["--all-data", "--seed", "2"]),
    ("alldata_s3", ["--all-data", "--seed", "3"]),
]
#: Stop after training; see the module docstring.
EXPLORE = True
#: Test-time augmentation for the sweep and the final predictions: all eight
#: orientations, which is what lifted run 1 from 0.30 to 0.32 public.
TTA = "dihedral"
#: Validation observations scored after every epoch to pick the checkpoint.
LIMIT_VAL = 48
#: Half of the 144 validation observations; the other half judges the result.
SWEEP_IMAGES = 72
#: Centred on what won for run 1 under the competition's scoring (threshold
#: 0.7, min area 400, bridge 16, close 3, open 0), which a 144-point sweep on
#: 2026-10-02 confirmed: smaller and larger areas, wider bridges and a
#: confidence filter all scored lower. A new model's calibration can move the
#: best cut-off, so the threshold and area still get some room.
SWEEP_GRID = {
    "--threshold": ["0.5", "0.6", "0.7", "0.8"],
    "--min-area": ["300", "400", "600"],
    "--bridge-gap": ["12", "16", "24"],
    "--close-radius": ["3"],
    "--open-radius": ["0"],
}
#: The settings that produced the best public score so far (0.32, run 1 with
#: TTA). The sweep reports every candidate's held-out difference from these.
BASELINE = "threshold=0.7,min_area=400,bridge_gap=16,close_radius=3,open_radius=0"
#: Post-processing settings the sweep may hand to predict.py.
SETTINGS = ("threshold", "min_area", "bridge_gap", "close_radius", "open_radius",
            "min_confidence")

STATUS: dict[str, str] = {}
#: Everything run_summary.json reports, accumulated so later writes add to the
#: environment report instead of replacing it.
SUMMARY: dict = {}


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
    report["nvidia_smi"] = (subprocess.run([smi, "-L"], capture_output=True, text=True,
                                           timeout=60)
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
    SUMMARY.update(extra, stages=STATUS)
    (KEEP / "run_summary.json").write_text(
        json.dumps(SUMMARY, indent=2, default=str), encoding="utf-8")


def train_variants(py: str, env: dict, workers: int, gpus: int) -> list[Path]:
    """Train every VARIANTS entry and return their checkpoint paths.

    With at least one GPU per variant they run side by side, each pinned to its
    own GPU and sharing the CPUs' data-loading workers; otherwise one after
    another. A path does not exist if that variant failed before its first epoch.
    """
    side_by_side = 1 < len(VARIANTS) <= gpus
    per_run = max(1, workers // len(VARIANTS)) if side_by_side else workers

    def command(name: str, flags: list) -> list:
        return [py, "scripts/train.py", "--epochs", EPOCHS, "--batch-size", BATCH_SIZE,
                "--workers", per_run, "--limit-val", LIMIT_VAL, "--preload",
                "--out", KEEP / f"model_{name}.pt", *flags]

    started = time.time()
    if side_by_side:
        runs = {}
        for gpu, (name, flags) in enumerate(VARIANTS):
            args = [str(a) for a in command(name, flags)]
            log(f"$ CUDA_VISIBLE_DEVICES={gpu} " + " ".join(args))
            handle = open(OUT / f"train_{name}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(args, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                                    env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)})
            runs[name] = (proc, handle)
        follow({name: OUT / f"train_{name}.log" for name in runs},
               lambda: any(proc.poll() is None for proc, _ in runs.values()))
        for name, (proc, handle) in runs.items():
            handle.close()
            STATUS[f"train {name}"] = ("ok" if proc.returncode == 0
                                       else f"failed: exit code {proc.returncode}")
    else:
        for name, flags in VARIANTS:
            attempt(f"train {name}", lambda name=name, flags=flags: run(
                command(name, flags), env, log_to=OUT / f"train_{name}.log"))
    log(f"training took {(time.time() - started) / 60:.0f} min")
    keep(*(OUT / f"train_{name}.log" for name, _ in VARIANTS))
    return [KEEP / f"model_{name}.pt" for name, _ in VARIANTS]


def follow(logs: dict[str, Path], still_running, every: float = 60.0) -> None:
    """Echo the epoch summaries and errors from each log until no run is left."""
    offsets = dict.fromkeys(logs, 0)
    while True:
        running = still_running()
        for name, path in logs.items():
            with open(path, "rb") as handle:
                handle.seek(offsets[name])
                chunk = handle.read()
            # Only whole lines; the rest is read on the next pass.
            cut = chunk.rfind(b"\n") + 1
            offsets[name] += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if (line.startswith(("epoch ", "train views", "best val", "trained without"))
                        or "Error" in line or "Traceback" in line):
                    log(f"[{name}] {line}")
        if not running:
            return
        time.sleep(every)


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
    keep(OUT / "splits.json")
    checkpoints = train_variants(py, env, workers, torch.cuda.device_count())
    write_summary(commit=commit, gpu=gpu, epochs=EPOCHS, variants=VARIANTS, explore=EXPLORE)
    if EXPLORE or len(VARIANTS) != 1:
        log(f"done: {STATUS}")
        return
    checkpoint = checkpoints[0]
    if not checkpoint.exists():
        raise SystemExit("training produced no checkpoint; nothing to tune or predict")

    # --- 2. tune post-processing on half of the validation observations -------
    def sweep() -> None:
        args = [py, "scripts/sweep_postprocess.py", "--checkpoint", checkpoint,
                "--device", "cuda", "--n-images", SWEEP_IMAGES, "--workers", workers,
                "--tta", TTA, "--holdout", "--baseline", BASELINE]
        for flag, values in SWEEP_GRID.items():
            args += [flag, *values]
        run(args, env, log_to=OUT / "sweep.log")
        keep(OUT / "sweep_postprocess.json", OUT / "sweep_postprocess_subset.json",
             OUT / "sweep.log")

    # With no tuned settings, predict.py uses PostprocessParams' own defaults.
    tuned: dict = {}
    if attempt("sweep", sweep):
        best = json.loads((OUT / "sweep_postprocess.json").read_text())[0]
        tuned = {key: best[key] for key in SETTINGS}
        log(f"sweep winner: {tuned} (PQ {best['pq']:.4f} on the sweep's own images)")

    # --- 3. predict validation and test with the winning settings -------------
    # Only the winner: with TTA every image is predicted eight times, and the
    # comparison that matters -- against the 0.32 run -- is made on the held-out
    # validation half after download, not against this run's own defaults.
    name = f"{VARIANTS[0][0]}_{TTA}"

    def predict(subset: str, filename: str) -> None:
        flags = [part for key, value in tuned.items()
                 for part in (f"--{key.replace('_', '-')}", value)]
        run([py, "scripts/predict.py", "--checkpoint", checkpoint, "--subset", subset,
             "--tta", TTA, *flags, "--out", OUT / filename], env)
        keep(OUT / filename)

    attempt("predict val", lambda: predict("val", f"val_{name}.csv"))
    attempt("predict test", lambda: predict("test", f"submission_{name}.csv"))

    # --- 4. score it on the half the sweep never saw ---------------------------
    def compare() -> None:
        run([py, "scripts/compare_holdout.py",
             "--sweep-subset", OUT / "sweep_postprocess_subset.json",
             "--out", OUT / "holdout_comparison.json",
             "--submission", OUT / f"val_{name}.csv"], env, log_to=OUT / "holdout.log")
        keep(OUT / "holdout_comparison.json", OUT / "holdout.log")

    attempt("holdout comparison", compare)
    write_summary(tta=TTA, baseline=BASELINE, tuned=tuned)
    log(f"done: {STATUS}")


if __name__ == "__main__":
    main()
