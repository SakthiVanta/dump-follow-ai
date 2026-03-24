#!/usr/bin/env python3
"""
Prepare YOLO dataset from raw images + VOC/COCO annotations.

Usage:
  # From VOC XML
  python scripts/prepare_dataset.py --format voc --src data/raw --out data/yolo_dataset

  # From COCO JSON
  python scripts/prepare_dataset.py --format coco --src data/raw --ann data/raw/annotations.json --out data/yolo_dataset

  # Optionally augment
  python scripts/prepare_dataset.py --format voc --src data/raw --out data/yolo_dataset --augment --multiplier 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import DatasetBuilder
from training.augmentation import augment_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["voc", "coco"], required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ann", default=None, help="COCO annotation JSON (coco mode)")
    parser.add_argument("--classes", nargs="+", default=["person"])
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--multiplier", type=int, default=3)
    args = parser.parse_args()

    split = tuple(args.split)
    builder = DatasetBuilder(
        source_dir=args.src,
        output_dir=args.out,
        class_names=args.classes,
        split=split,
    )

    if args.format == "voc":
        data_yaml = builder.build_from_voc()
    else:
        if not args.ann:
            print("--ann required for coco format")
            sys.exit(1)
        data_yaml = builder.build_from_coco(args.ann)

    print(f"Dataset ready: {data_yaml}")

    if args.augment:
        train_img = Path(args.out) / "train" / "images"
        train_lbl = Path(args.out) / "train" / "labels"
        n = augment_dataset(
            images_dir=train_img,
            labels_dir=train_lbl,
            output_images_dir=train_img,
            output_labels_dir=train_lbl,
            multiplier=args.multiplier,
        )
        print(f"Augmentation: +{n} images")


if __name__ == "__main__":
    main()
