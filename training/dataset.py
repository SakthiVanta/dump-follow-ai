"""
Dataset builder for YOLO fine-tuning.

Supports:
  - COCO-format JSON annotation → YOLO txt conversion
  - VOC XML → YOLO txt conversion
  - Train/val/test split
  - Auto data.yaml generation
"""
from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import yaml
from robot.logger import logger


class DatasetBuilder:
    """
    Prepares a YOLO-compatible dataset from raw images + annotations.

    Parameters
    ----------
    source_dir: Root directory containing 'images/' and 'annotations/'
    output_dir: Where to write the YOLO dataset
    class_names: List of class names to include (filters others)
    split: (train, val, test) ratios — must sum to 1.0
    seed: Random seed for reproducible splits
    """

    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        class_names: list[str],
        split: tuple[float, float, float] = (0.7, 0.2, 0.1),
        seed: int = 42,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.class_map = {name: i for i, name in enumerate(class_names)}
        self.split = split
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_coco(self, annotation_file: str | Path) -> Path:
        """
        Convert COCO JSON annotations to YOLO format and write dataset.
        Returns path to generated data.yaml.
        """
        ann_path = Path(annotation_file)
        with ann_path.open() as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Group annotations by image
        ann_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        records: list[tuple[Path, list[str]]] = []
        img_dir = self.source_dir / "images"

        for img_id, annotations in ann_by_image.items():
            img_info = images[img_id]
            img_path = img_dir / img_info["file_name"]
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            W, H = img_info["width"], img_info["height"]
            lines: list[str] = []

            for ann in annotations:
                cat_name = categories.get(ann["category_id"], "")
                if cat_name not in self.class_map:
                    continue
                cls_id = self.class_map[cat_name]
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw = w / W
                nh = h / H
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if lines:
                records.append((img_path, lines))

        return self._write_dataset(records)

    def build_from_voc(self, annotations_dir: Optional[str | Path] = None) -> Path:
        """
        Convert Pascal VOC XML annotations to YOLO format.
        Returns path to generated data.yaml.
        """
        ann_dir = Path(annotations_dir) if annotations_dir else self.source_dir / "annotations"
        records: list[tuple[Path, list[str]]] = []

        for xml_file in sorted(ann_dir.glob("*.xml")):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            img_filename = root.findtext("filename", "")
            img_path = self.source_dir / "images" / img_filename
            if not img_path.exists():
                continue

            size = root.find("size")
            W = int(size.findtext("width", "1"))  # type: ignore
            H = int(size.findtext("height", "1"))  # type: ignore

            lines: list[str] = []
            for obj in root.findall("object"):
                name = obj.findtext("name", "")
                if name not in self.class_map:
                    continue
                cls_id = self.class_map[name]
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.findtext("xmin", "0"))  # type: ignore
                ymin = float(bndbox.findtext("ymin", "0"))  # type: ignore
                xmax = float(bndbox.findtext("xmax", "0"))  # type: ignore
                ymax = float(bndbox.findtext("ymax", "0"))  # type: ignore
                cx = ((xmin + xmax) / 2) / W
                cy = ((ymin + ymax) / 2) / H
                nw = (xmax - xmin) / W
                nh = (ymax - ymin) / H
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if lines:
                records.append((img_path, lines))

        return self._write_dataset(records)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _write_dataset(self, records: list[tuple[Path, list[str]]]) -> Path:
        random.seed(self.seed)
        random.shuffle(records)

        n = len(records)
        n_train = int(n * self.split[0])
        n_val = int(n * self.split[1])

        splits = {
            "train": records[:n_train],
            "val": records[n_train : n_train + n_val],
            "test": records[n_train + n_val :],
        }

        for split_name, items in splits.items():
            img_out = self.output_dir / split_name / "images"
            lbl_out = self.output_dir / split_name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_path, lines in items:
                shutil.copy2(img_path, img_out / img_path.name)
                label_file = lbl_out / (img_path.stem + ".txt")
                label_file.write_text("\n".join(lines))

        # Write data.yaml
        data_yaml = self.output_dir / "data.yaml"
        yaml_content = {
            "path": str(self.output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        with data_yaml.open("w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        logger.info(
            f"Dataset written to {self.output_dir} — "
            f"train={len(splits['train'])} val={len(splits['val'])} "
            f"test={len(splits['test'])}"
        )
        return data_yaml
