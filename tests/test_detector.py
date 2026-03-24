"""Unit tests for Detection dataclass and YOLODetector (mocked)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from robot.vision.detector import Detection, YOLODetector


class TestDetection:
    def test_cx_cy(self):
        d = Detection("person", 0.9, x=100, y=50, w=80, h=120)
        assert d.cx == pytest.approx(140.0)
        assert d.cy == pytest.approx(110.0)

    def test_area(self):
        d = Detection("person", 0.9, x=0, y=0, w=80, h=120)
        assert d.area == 9600

    def test_to_dict_keys(self):
        d = Detection("person", 0.9, x=100, y=50, w=80, h=120, track_id=3)
        keys = d.to_dict().keys()
        for k in ("label", "confidence", "x", "y", "w", "h", "cx", "cy", "area", "track_id"):
            assert k in keys

    def test_track_id_none_by_default(self):
        d = Detection("person", 0.9, x=0, y=0, w=10, h=10)
        assert d.track_id is None


class TestYOLODetectorMocked:
    """Tests that mock the YOLO model to avoid downloading weights."""

    @patch("robot.vision.detector.YOLO", create=True)
    def test_detect_returns_detections(self, mock_yolo_cls, blank_frame):
        # Build a fake result with one detection
        box = MagicMock()
        box.cls = [0]
        box.conf = [0.9]
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = [10.0, 20.0, 90.0, 140.0]

        result = MagicMock()
        result.boxes = [box]

        mock_model = MagicMock()
        mock_model.return_value = [result]
        mock_model.names = {0: "person"}
        mock_yolo_cls.return_value = mock_model

        # Patch the import inside detector
        import robot.vision.detector as det_mod
        det_mod.YOLO = mock_yolo_cls  # type: ignore

        detector = YOLODetector.__new__(YOLODetector)
        detector.conf_threshold = 0.5
        detector.iou_threshold = 0.45
        detector.target_classes = {"person"}
        detector._model = mock_model
        detector._class_names = {0: "person"}

        detections = detector.detect(blank_frame)
        assert len(detections) == 1
        assert detections[0].label == "person"
        assert detections[0].x == 10
        assert detections[0].w == 80
        assert detections[0].h == 120

    @patch("robot.vision.detector.YOLO", create=True)
    def test_target_class_filter(self, mock_yolo_cls, blank_frame):
        box = MagicMock()
        box.cls = [1]
        box.conf = [0.9]
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = [0.0, 0.0, 50.0, 50.0]

        result = MagicMock()
        result.boxes = [box]

        mock_model = MagicMock()
        mock_model.return_value = [result]
        mock_model.names = {1: "dog"}
        mock_yolo_cls.return_value = mock_model

        import robot.vision.detector as det_mod
        det_mod.YOLO = mock_yolo_cls  # type: ignore

        detector = YOLODetector.__new__(YOLODetector)
        detector.conf_threshold = 0.5
        detector.iou_threshold = 0.45
        detector.target_classes = {"person"}
        detector._model = mock_model
        detector._class_names = {1: "dog"}

        # dog is not in target_classes → filtered
        detections = detector.detect(blank_frame)
        assert len(detections) == 0
