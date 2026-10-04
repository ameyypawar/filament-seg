"""Kaggle entry point for the detector run: clone the repository and run its pipeline.

Kept to a few lines on purpose; everything that matters lives, versioned, in
kaggle/detector_pipeline.py. DETECTOR below is the only thing that changes
between detector runs:

* detector 1: yolo11s-seg, 1024 px, 60 epochs
* detector 2: yolo11m-seg, 1280 px, 80 epochs
* detector 3: yolo11s-seg, 1280 px, 80 epochs, seed 1
* detector 4: yolo11m-seg at 1536 px, for the small filaments the others miss
  (missed filaments have a median area of 917 px, matched ones 1394 px)

    kaggle kernels push -p kaggle/detector/
"""

import subprocess
import sys

REPO_URL = "https://github.com/ameyypawar/filament-seg"
REPO = "/tmp/work/filament-seg"
DETECTOR = ["--model", "yolo11m-seg.pt", "--imgsz", "1536", "--epochs", "80", "--seed", "2"]

subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
sys.exit(subprocess.run([sys.executable, f"{REPO}/kaggle/detector_pipeline.py", *DETECTOR]).returncode)
