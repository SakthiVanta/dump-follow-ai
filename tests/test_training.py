"""Unit tests for training utilities."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal fake dataset for testing DatasetBuilder."""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    # Create 5 dummy images and VOC annotations
    import cv2
    for i in range(5):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_path = img_dir / f"img_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)

        xml = f"""<annotation>
  <filename>img_{i:03d}.jpg</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin><ymin>100</ymin><xmax>300</xmax><ymax>400</ymax>
    </bndbox>
  </object>
</annotation>"""
        (ann_dir / f"img_{i:03d}.xml").write_text(xml)

    return tmp_path


@pytest.fixture
def tmp_coco_dataset(tmp_path):
    """Create a minimal COCO-format dataset."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    import cv2
    images, annotations = [], []
    for i in range(3):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        fname = f"img_{i:03d}.jpg"
        cv2.imwrite(str(img_dir / fname), img)
        images.append({"id": i, "file_name": fname, "width": 640, "height": 480})
        annotations.append({
            "id": i, "image_id": i, "category_id": 1,
            "bbox": [100, 100, 200, 300]
        })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(coco))
    return tmp_path, ann_file


class TestDatasetBuilder:
    def test_build_from_voc(self, tmp_dataset, tmp_path):
        from training.dataset import DatasetBuilder
        out = tmp_path / "yolo_dataset"
        builder = DatasetBuilder(
            source_dir=tmp_dataset,
            output_dir=out,
            class_names=["person"],
            split=(0.6, 0.2, 0.2),
            seed=42,
        )
        data_yaml = builder.build_from_voc()
        assert data_yaml.exists()
        assert (out / "train" / "images").exists()
        assert (out / "val" / "images").exists()

    def test_build_from_coco(self, tmp_coco_dataset, tmp_path):
        from training.dataset import DatasetBuilder
        src_dir, ann_file = tmp_coco_dataset
        out = tmp_path / "coco_out"
        builder = DatasetBuilder(
            source_dir=src_dir,
            output_dir=out,
            class_names=["person"],
        )
        data_yaml = builder.build_from_coco(ann_file)
        assert data_yaml.exists()

    def test_data_yaml_has_correct_classes(self, tmp_dataset, tmp_path):
        import yaml
        from training.dataset import DatasetBuilder
        out = tmp_path / "yolo_out2"
        builder = DatasetBuilder(
            source_dir=tmp_dataset,
            output_dir=out,
            class_names=["person", "robot"],
        )
        data_yaml = builder.build_from_voc()
        with open(data_yaml) as f:
            d = yaml.safe_load(f)
        assert d["nc"] == 2
        assert "person" in d["names"]

    def test_label_files_are_yolo_format(self, tmp_dataset, tmp_path):
        from training.dataset import DatasetBuilder
        out = tmp_path / "yolo_fmt"
        builder = DatasetBuilder(
            source_dir=tmp_dataset,
            output_dir=out,
            class_names=["person"],
            split=(1.0, 0.0, 0.0),
        )
        builder.build_from_voc()
        labels = list((out / "train" / "labels").glob("*.txt"))
        assert len(labels) > 0
        for lbl in labels:
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                assert len(parts) == 5
                # class id is int, rest are floats in [0,1]
                assert int(parts[0]) >= 0
                for val in parts[1:]:
                    assert 0.0 <= float(val) <= 1.0
