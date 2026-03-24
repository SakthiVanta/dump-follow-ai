#!/usr/bin/env bash
# One-shot install that ensures opencv GUI build wins over headless.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$ROOT"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

pip install --upgrade pip
# Install everything EXCEPT opencv first (albumentations pulls headless)
pip install -r requirements.txt --constraint /dev/null
# Then force the GUI build on top — this must come last
pip install opencv-python --force-reinstall --no-deps

echo ""
echo "Install complete. Run:  python main.py demo"
