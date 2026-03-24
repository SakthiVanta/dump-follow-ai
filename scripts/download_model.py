#!/usr/bin/env python3
"""
Download YOLOv8 nano weights for offline use.
Usage: python scripts/download_model.py [--model yolov8n.pt]
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt"])
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / args.model

    if dest.exists():
        print(f"Already exists: {dest}")
        return

    from ultralytics import YOLO
    print(f"Downloading {args.model}…")
    model = YOLO(args.model)
    # Ultralytics caches in ~/.config/Ultralytics/ — copy to our models/
    import shutil, os
    # Find the cached file
    cache_dir = Path.home() / ".config" / "Ultralytics"
    cached = list(cache_dir.glob(f"**/{args.model}"))
    if cached:
        shutil.copy2(cached[0], dest)
        print(f"Saved to: {dest}")
    else:
        # Save directly
        model.save(str(dest))
        print(f"Saved to: {dest}")


if __name__ == "__main__":
    main()
