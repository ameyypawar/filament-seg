#!/usr/bin/env bash
# Download and unpack the competition data into ./data/
#
# Prerequisites (both are one-time, and both need you, not the script):
#   1. Accept the competition rules at
#      https://www.kaggle.com/competitions/filament-segmentation-2026/rules
#   2. Create an API token: Kaggle -> Settings -> API -> "Create New API Token",
#      then save the downloaded file to ~/.kaggle/kaggle.json and
#      chmod 600 ~/.kaggle/kaggle.json
#
# The archive is ~751 MB and unpacks to roughly 1.5 GB.

set -euo pipefail

COMPETITION="filament-segmentation-2026"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "error: the 'kaggle' CLI is not on PATH." >&2
  echo "       install it with:  uv pip install kaggle   (or pip install kaggle)" >&2
  exit 1
fi

if [ ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]; then
  echo "error: no kaggle.json found. See the header of this script." >&2
  exit 1
fi

mkdir -p "$DATA_DIR"
echo "downloading $COMPETITION into $DATA_DIR ..."
kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

echo "unzipping ..."
unzip -q -o "$DATA_DIR/${COMPETITION}.zip" -d "$DATA_DIR"

echo "done. tree:"
find "$DATA_DIR" -maxdepth 3 -type d | head -20
