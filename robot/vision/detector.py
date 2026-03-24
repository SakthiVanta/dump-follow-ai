"""YOLO-based object detector wrapping Ultralytics."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from robot.logger import logger


@dataclass
class Detection:
    label: str
    confidence: float
    x: int        # top-left x
    y: int        # top-left y
    w: int        # width
    h: int        # height
    track_id: Optional[int] = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "area": self.area,
            "track_id": self.track_id,
        }


class YOLODetector:
    """
    Thin wrapper around Ultralytics YOLO.

    Parameters
    ----------
    model_path: Path to .pt file.  Downloads yolov8n.pt automatically if absent.
    conf_threshold: Minimum confidence to keep a detection.
    iou_threshold:  NMS IoU threshold.
    target_classes: If set, filter detections to these class names only.
    device:         'cpu', 'cuda', 'mps', or empty string for auto.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_classes: list[str] | None = None,
        device: str = "",
    ) -> None:
        from ultralytics import YOLO  # lazy import — keeps tests fast

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes: set[str] = set(target_classes or [])

        model_file = Path(model_path)
        if not model_file.exists():
            logger.warning(
                f"Model not found at {model_path}, using yolov8n.pt (auto-download)"
            )
            model_path = "yolov8n.pt"

        logger.info(f"Loading YOLO model from: {model_path}")
        self._model = YOLO(model_path)
        if device:
            self._model.to(device)

        # Build class-name → id mapping
        self._class_names: dict[int, str] = self._model.names  # type: ignore[assignment]
        logger.info(f"Detector ready. Classes: {len(self._class_names)}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame. Returns list of Detection."""
        results = self._model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self._class_names.get(cls_id, str(cls_id))

                if self.target_classes and label not in self.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        x=x1,
                        y=y1,
                        w=x2 - x1,
                        h=y2 - y1,
                    )
                )
        return detections

    def warmup(self, frame_size: tuple[int, int] = (240, 320)) -> None:
        """Run one dummy inference to initialise CUDA / model weights."""
        dummy = np.zeros((*frame_size, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.debug("Detector warm-up complete")
