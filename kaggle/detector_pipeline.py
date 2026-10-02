"""Kaggle GPU: train the filament detector and predict validation and test instances.

Started by kaggle/detector/run.py, which clones this repository first, and
reuses the environment checks and stage helpers of kaggle/train_on_kaggle.py.
The U-Net half of fusion is not retrained here -- run 3's checkpoint is reused
-- so the outputs are the detector and its detections. Fusion is tuned locally
by scripts/fuse_detections.py against cached U-Net logits.

The detector trains straight into /kaggle/working, where Ultralytics saves its
best and last weights after every epoch, so a crash or the session time limit
cannot cost the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_on_kaggle as k  # noqa: E402

MODEL = "yolo11s-seg.pt"
IMGSZ = 1024
EPOCHS = 60
BATCH = 8
#: Pinned below the next major version: the training and prediction calls
#: were written against the 8.x API.
ULTRALYTICS = "ultralytics>=8.3,<9"


def main() -> None:
    k.KEEP.mkdir(parents=True, exist_ok=True)
    k.OUT.mkdir(parents=True, exist_ok=True)

    import torch
    report = k.environment_report()
    k.write_summary(environment=report)
    if not torch.cuda.is_available():
        raise SystemExit("no usable GPU; refusing to train on CPU")
    gpu = torch.cuda.get_device_name(0)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=k.REPO,
                            capture_output=True, text=True).stdout.strip()
    k.log(f"GPU: {gpu} | repo at {commit}")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS], check=True)
    version = subprocess.run([sys.executable, "-c", "import ultralytics; print(ultralytics.__version__)"],
                             capture_output=True, text=True).stdout.strip()
    k.log(f"ultralytics {version}")

    data_root = k.find_data_root()
    env = {**os.environ, "FILAMENT_DATA_ROOT": str(data_root),
           "FILAMENT_OUTPUT_ROOT": str(k.OUT), "PYTHONUNBUFFERED": "1"}
    workers = min(4, os.cpu_count() or 2)
    py = sys.executable
    k.write_summary(commit=commit, gpu=gpu, ultralytics=version, model=MODEL, imgsz=IMGSZ,
                    epochs=EPOCHS, batch=BATCH)

    k.run([py, "scripts/preprocess.py", "--workers", workers], env)
    k.run([py, "scripts/make_splits.py", "--group-by", "date"], env)
    k.keep(k.OUT / "splits.json")
    yolo_dir = k.WORK / "yolo"
    k.run([py, "scripts/export_yolo.py", "--out", yolo_dir], env)

    detector_dir = k.KEEP / "detector"
    k.attempt("train detector", lambda: k.run(
        [py, "scripts/train_detector.py", "--data", yolo_dir / "data.yaml", "--out", detector_dir,
         "--model", MODEL, "--imgsz", IMGSZ, "--epochs", EPOCHS, "--batch", BATCH,
         "--device", "0", "--workers", workers],
        env, log_to=k.OUT / "detector_train.log"))
    k.keep(k.OUT / "detector_train.log")
    weights = detector_dir / "detector_best.pt"
    if not weights.exists():
        # Training died part-way: fall back to the best epoch Ultralytics saved.
        weights = detector_dir / "train" / "weights" / "best.pt"
    if not weights.exists():
        k.write_summary()
        raise SystemExit("the detector produced no weights; nothing to predict")

    for subset in ("val", "test"):
        out = k.OUT / f"detections_{subset}.json"
        k.attempt(f"detect {subset}", lambda s=subset, o=out: k.run(
            [py, "scripts/yolo_predict.py", "--weights", weights, "--subset", s,
             "--imgsz", IMGSZ, "--device", "0", "--out", o], env))
        k.keep(out)

    k.write_summary(weights=str(weights))
    k.log(f"done: {k.STATUS}")


if __name__ == "__main__":
    main()
