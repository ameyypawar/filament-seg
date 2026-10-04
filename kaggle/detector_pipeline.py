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

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_on_kaggle as k  # noqa: E402

#: Detector settings come from kaggle/detector/run.py's command line, so each
#: new detector is a one-line change there. Defaults are detector 2's
#: (yolo11m-seg, 1280 px, 80 epochs): detector 1 (yolo11s-seg, 1024 px, 60
#: epochs) lifted held-out PQ by +0.037, detector 2 by +0.045, and the two
#: combined by +0.051.
DEFAULTS = {"model": "yolo11m-seg.pt", "imgsz": 1280, "epochs": 80,
            #: Total across all GPUs (Ultralytics splits it per device).
            "batch": 8,
            #: Both T4s; if distributed training fails, one GPU is tried from scratch.
            "devices": "0,1", "seed": 0}
#: Pinned below the next major version: the training and prediction calls
#: were written against the 8.x API.
ULTRALYTICS = "ultralytics>=8.3,<9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--devices", default=DEFAULTS["devices"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    # The first import prints a settings banner; the version is the last line.
    version = subprocess.run([sys.executable, "-c", "import ultralytics; print(ultralytics.__version__)"],
                             capture_output=True, text=True).stdout.strip().splitlines()[-1]
    k.log(f"ultralytics {version}")

    data_root = k.find_data_root()
    env = {**os.environ, "FILAMENT_DATA_ROOT": str(data_root),
           "FILAMENT_OUTPUT_ROOT": str(k.OUT), "PYTHONUNBUFFERED": "1"}
    workers = min(4, os.cpu_count() or 2)
    py = sys.executable
    k.write_summary(commit=commit, gpu=gpu, gpus=torch.cuda.device_count(), ultralytics=version,
                    **vars(args))

    k.run([py, "scripts/preprocess.py", "--workers", workers], env)
    k.run([py, "scripts/make_splits.py", "--group-by", "date"], env)
    k.keep(k.OUT / "splits.json")
    yolo_dir = k.WORK / "yolo"
    k.run([py, "scripts/export_yolo.py", "--out", yolo_dir], env)

    detector_dir = k.KEEP / "detector"

    def train(devices: str) -> bool:
        return k.attempt(f"train detector on {devices}", lambda: k.run(
            [py, "scripts/train_detector.py", "--data", yolo_dir / "data.yaml",
             "--out", detector_dir, "--model", args.model, "--imgsz", args.imgsz,
             "--epochs", args.epochs, "--batch", args.batch, "--device", devices,
             "--workers", workers, "--patience", 25, "--seed", args.seed],
            env, log_to=k.OUT / f"detector_train_{devices.replace(',', '')}.log"))

    if not train(args.devices) and args.devices != "0":
        train("0")
    k.keep(*k.OUT.glob("detector_train_*.log"))
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
             "--imgsz", args.imgsz, "--device", "0", "--out", o], env))
        k.keep(out)

    k.write_summary(weights=str(weights))
    k.log(f"done: {k.STATUS}")


if __name__ == "__main__":
    main()
