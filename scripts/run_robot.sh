#!/usr/bin/env bash
# Run the full robot with venv activated
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

cd "$ROOT"

if [ ! -d "venv" ]; then
  echo "Creating venv..."
  python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

echo "Starting robot..."
python main.py robot "$@"
