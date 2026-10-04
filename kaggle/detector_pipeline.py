"""Kaggle GPU: train the filament detector and predict validation and test instances.

Started by kaggle/detector/run.py, which clones this repository first, and
reuses the environment checks and stage helpers of kaggle/train_on_kaggle.py.
The U-Net half of fusion is not retrained here -- run 3's checkpoint is reused
-- so the outputs are the detector and its detections. Fusion is tuned locally
by scripts/fuse_detections.py against cached U-Net logits.

The detector trains straight into /kaggle/working, where Ultralytics saves its
best and last weights after every epoch, so a crash or the session time limit
cannot cost the run.

Both checkpoints predict: best.pt into detections_{val,test}.json, last.pt
into detections_{val,test}_last.json. best.pt is the epoch Ultralytics scored
highest on the validation views, a choice made on the data that later judges
it; last.pt is not, and it is what an ``--all-data`` run (validation views in
training, no early stopping) produces, so the final detectors are validated
as last.pt.
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
    parser.add_argument("--all-data", action="store_true",
                        help="final detector: train on the validation views too")
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
    k.run([py, "scripts/export_yolo.py", "--out", yolo_dir,
           *(["--all-data"] if args.all_data else [])], env)

    detector_dir = k.KEEP / "detector"
    # An all-data run has no validation views to stop early on.
    schedule = ["--patience", 0, "--no-val"] if args.all_data else ["--patience", 25]

    def train(devices: str, batch: int) -> bool:
        return k.attempt(f"train detector on {devices}, batch {batch}", lambda: k.run(
            [py, "scripts/train_detector.py", "--data", yolo_dir / "data.yaml",
             "--out", detector_dir, "--model", args.model, "--imgsz", args.imgsz,
             "--epochs", args.epochs, "--batch", batch, "--device", devices,
             "--workers", workers, "--seed", args.seed, *schedule],
            env, log_to=k.OUT / f"detector_train_{devices.replace(',', '')}_{batch}.log"))

    # If distributed training fails, one GPU from scratch with the same batch
    # per GPU; if that runs out of memory too, half of it.
    for devices, batch in dict.fromkeys([(args.devices, args.batch),
                                         ("0", max(2, args.batch // 2)),
                                         ("0", max(2, args.batch // 4))]):
        if train(devices, batch):
            break
    k.keep(*k.OUT.glob("detector_train_*.log"))

    # suffix of the detections file -> weights. If training died part-way,
    # train_detector.py never copied them, but Ultralytics' own copies remain.
    checkpoints = {}
    for name, suffix in (("best", ""), ("last", "_last")):
        if args.all_data and name == "best":
            continue
        for weights in (detector_dir / f"detector_{name}.pt",
                        detector_dir / "train" / "weights" / f"{name}.pt"):
            if weights.exists():
                checkpoints[suffix] = weights
                break
    if not checkpoints:
        k.write_summary()
        raise SystemExit("the detector produced no weights; nothing to predict")

    for suffix, weights in checkpoints.items():
        for subset in ("val", "test"):
            out = k.OUT / f"detections_{subset}{suffix}.json"
            k.attempt(f"detect {subset}{suffix}", lambda s=subset, o=out, w=weights: k.run(
                [py, "scripts/yolo_predict.py", "--weights", w, "--subset", s,
                 "--imgsz", args.imgsz, "--device", "0", "--out", o], env))
            k.keep(out)

    k.write_summary(weights={suffix or "_best": str(w) for suffix, w in checkpoints.items()})
    k.log(f"done: {k.STATUS}")


if __name__ == "__main__":
    main()
