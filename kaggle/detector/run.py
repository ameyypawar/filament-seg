"""Kaggle entry point for the detector run: clone the repository and run its pipeline.

Kept to a few lines on purpose; everything that matters lives, versioned, in
kaggle/detector_pipeline.py.

    kaggle kernels push -p kaggle/detector/
"""

import subprocess
import sys

REPO_URL = "https://github.com/ameyypawar/filament-seg"
REPO = "/tmp/work/filament-seg"

subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
sys.exit(subprocess.run([sys.executable, f"{REPO}/kaggle/detector_pipeline.py"]).returncode)
