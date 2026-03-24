"""
Offline augmentation pipeline using albumentations.
Used to artificially expand small training sets.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from robot.logger import logger

try:
    import albumentations as A
    _ALBUMENTATIONS = True
except ImportError:
    _ALBUMENTATIONS = False
    logger.warning("albumentations not installed — augmentation unavailable")


def _build_pipeline() -> "A.Compose":
    import albumentations as A
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.HueSaturationValue(p=0.4),
            A.Rotate(limit=15, p=0.4),
            A.RandomScale(scale_limit=0.2, p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def augment_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_images_dir: str | Path,
    output_labels_dir: str | Path,
    multiplier: int = 3,
    seed: int = 42,
) -> int:
    """
    Apply random augmentations to each image `multiplier` times.
    Returns total number of generated images.
    """
    if not _ALBUMENTATIONS:
        raise RuntimeError("albumentations not installed")

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_img = Path(output_images_dir)
    out_lbl = Path(output_labels_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline()
    random.seed(seed)
    count = 0

    for img_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = _parse_yolo_labels(lbl_path)

        for i in range(multiplier):
            try:
                result = pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
                aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
                stem = f"{img_path.stem}_aug{i}"
                cv2.imwrite(str(out_img / f"{stem}.jpg"), aug_img)
                _write_yolo_labels(
                    out_lbl / f"{stem}.txt",
                    result["bboxes"],
                    result["class_labels"],
                )
                count += 1
            except Exception as exc:
                logger.warning(f"Augmentation failed for {img_path.name}: {exc}")

    logger.info(f"Augmentation complete — {count} images generated")
    return count


def _parse_yolo_labels(lbl_path: Path) -> tuple[list, list]:
    bboxes, class_labels = [], []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) == 5:
            cls, cx, cy, w, h = parts
            bboxes.append((float(cx), float(cy), float(w), float(h)))
            class_labels.append(int(cls))
    return bboxes, class_labels


def _write_yolo_labels(path: Path, bboxes: list, class_labels: list) -> None:
    lines = [
        f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for (cx, cy, w, h), cls in zip(bboxes, class_labels)
    ]
    path.write_text("\n".join(lines))
