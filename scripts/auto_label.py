#!/usr/bin/env python3
"""
Bootstrap a YOLO dataset by auto-labelling images with an existing YOLO model.

Typical flow:
  1. Capture raw images from your robot camera.
  2. Run this script to create YOLO txt labels automatically.
  3. Review/fix the labels in a labelling tool.
  4. Fine-tune a custom model on the reviewed dataset.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder with raw .jpg/.png images")
    parser.add_argument("--out", required=True, help="Output dataset root")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["person"],
        help="Class names to keep, e.g. person bottle chair",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    output_dir = Path(args.out)

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    model = YOLO(args.model)
    names = model.names
    class_to_id = {name: idx for idx, name in enumerate(args.classes)}

    records: list[tuple[Path, list[str]]] = []

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            verbose=False,
        )
        lines: list[str] = []
        for result in results:
            if result.boxes is None:
                continue
            width = float(result.orig_shape[1])
            height = float(result.orig_shape[0])
            for box in result.boxes:
                label = names.get(int(box.cls[0]), str(int(box.cls[0])))
                if label not in class_to_id:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1 + x2) / 2.0) / width
                cy = ((y1 + y2) / 2.0) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                lines.append(
                    f"{class_to_id[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )
        if lines:
            records.append((image_path, lines))

    if not records:
        raise SystemExit("No matching detections found to label")

    write_dataset(records, output_dir, args.classes, tuple(args.split))
    print(f"Auto-labelled dataset written to: {output_dir}")
    print("Review the generated labels before training your custom model.")


def write_dataset(
    records: list[tuple[Path, list[str]]],
    output_dir: Path,
    class_names: list[str],
    split: tuple[float, float, float],
) -> None:
    total = len(records)
    train_end = int(total * split[0])
    val_end = train_end + int(total * split[1])
    split_map = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    for split_name, items in split_map.items():
        image_out = output_dir / split_name / "images"
        label_out = output_dir / split_name / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for image_path, lines in items:
            shutil.copy2(image_path, image_out / image_path.name)
            (label_out / f"{image_path.stem}.txt").write_text("\n".join(lines))

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                f"nc: {len(class_names)}",
                f"names: {class_names}",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
