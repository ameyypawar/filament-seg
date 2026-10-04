"""Kaggle entry point for the detector run: clone the repository and run its pipeline.

Kept to a few lines on purpose; everything that matters lives, versioned, in
kaggle/detector_pipeline.py. DETECTOR below is the only thing that changes
between detector runs: detector 1 was yolo11s-seg at 1024 px for 60 epochs,
detector 2 yolo11m-seg at 1280 px for 80. Detector 3 adds a third, different
member to the ensemble.

    kaggle kernels push -p kaggle/detector/
"""

import subprocess
import sys

REPO_URL = "https://github.com/ameyypawar/filament-seg"
REPO = "/tmp/work/filament-seg"
DETECTOR = ["--model", "yolo11s-seg.pt", "--imgsz", "1280", "--epochs", "80", "--seed", "1"]

subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
sys.exit(subprocess.run([sys.executable, f"{REPO}/kaggle/detector_pipeline.py", *DETECTOR]).returncode)
