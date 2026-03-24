"""
FramePipeline — orchestrates capture → detect → track → select target.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from robot.config import VisionConfig
from robot.vision.detector import Detection, YOLODetector
from robot.vision.tracker import ByteTrackWrapper, DeepSORTWrapper
from robot.logger import logger


class FramePipeline:
    """
    Full vision pipeline:
      1. Capture frame (real camera or injected frame)
      2. Resize to configured resolution
      3. Run YOLO detection
      4. Optionally apply tracker
      5. Select best target (largest bounding box of target class)
      6. Return annotated frame + selected target
    """

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._detector: Optional[YOLODetector] = None
        self._tracker: Optional[ByteTrackWrapper | DeepSORTWrapper] = None
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._locked_target: Optional[Detection] = None
        self._selection_radius_px = 120.0

    def start(self) -> None:
        """Open camera and initialise detector."""
        self._detector = YOLODetector(
            model_path=self.config.model_path,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            target_classes=[self.config.target_class],
        )
        self._detector.warmup((self.config.frame_height, self.config.frame_width))

        if self.config.tracker == "deepsort":
            self._tracker = DeepSORTWrapper()
        elif self.config.tracker == "bytetrack":
            self._tracker = ByteTrackWrapper()

        self._cap = cv2.VideoCapture(self.config.camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera id={self.config.camera_id}"
            )
        logger.info("FramePipeline started")

    def stop(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        logger.info("FramePipeline stopped")

    def read_frame(self) -> Optional[np.ndarray]:
        """Grab next frame. Returns None if camera fails."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("Camera read failed")
            return None
        return cv2.resize(
            frame, (self.config.frame_width, self.config.frame_height)
        )

    def process(
        self, frame: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, list[Detection], Optional[Detection]]:
        """
        Run full pipeline on given frame (or captured frame).

        Returns
        -------
        (annotated_frame, all_detections, best_target)
        """
        if frame is None:
            frame = self.read_frame()
        if frame is None:
            blank = np.zeros(
                (self.config.frame_height, self.config.frame_width, 3),
                dtype=np.uint8,
            )
            return blank, [], None

        assert self._detector is not None, "Call start() first"

        detections = self._detector.detect(frame)

        if self._tracker is not None:
            if isinstance(self._tracker, DeepSORTWrapper):
                detections = self._tracker.update(detections, frame)
            else:
                detections = self._tracker.update(detections)

        target = self._select_target(detections)
        annotated = self._annotate(frame.copy(), detections, target)

        self._update_fps()
        return annotated, detections, target

    def select_target_at_point(
        self,
        x: int,
        y: int,
        detections: list[Detection],
    ) -> Optional[Detection]:
        """Lock onto the smallest box under the click point."""
        matches = [
            det
            for det in detections
            if det.x <= x <= det.x + det.w and det.y <= y <= det.y + det.h
        ]
        if not matches:
            self.clear_target_selection()
            return None

        selected = min(matches, key=lambda det: det.area)
        self.lock_target(selected)
        return selected

    def lock_target(self, detection: Optional[Detection]) -> None:
        """Persist a user-selected detection for future frames."""
        self._locked_target = detection

    def clear_target_selection(self) -> None:
        self._locked_target = None

    @property
    def locked_target(self) -> Optional[Detection]:
        return self._locked_target

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_target(
        self, detections: list[Detection]
    ) -> Optional[Detection]:
        """Return the locked target if present, otherwise the largest target_class."""
        locked = self._match_locked_target(detections)
        if locked is not None:
            self._locked_target = locked
            return locked

        candidates = [
            d for d in detections if d.label == self.config.target_class
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.area)

    def _match_locked_target(
        self, detections: list[Detection]
    ) -> Optional[Detection]:
        if self._locked_target is None:
            return None

        locked = self._locked_target

        if locked.track_id is not None:
            for det in detections:
                if det.track_id == locked.track_id:
                    return det

        same_label = [det for det in detections if det.label == locked.label]
        if not same_label:
            return None

        best = min(same_label, key=lambda det: _centroid_distance_sq(det, locked))
        if _centroid_distance_sq(best, locked) <= self._selection_radius_px ** 2:
            if locked.track_id is not None and best.track_id is None:
                best.track_id = locked.track_id
            return best
        return None

    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        target: Optional[Detection],
    ) -> np.ndarray:
        for det in detections:
            color = (0, 255, 0) if det is target else (200, 200, 200)
            cv2.rectangle(frame, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 2)
            label = f"{det.label} {det.confidence:.2f}"
            if det.track_id is not None:
                label += f" #{det.track_id}"
            cv2.putText(
                frame, label, (det.x, det.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        # FPS overlay
        cv2.putText(
            frame, f"FPS: {self._fps:.1f}",
            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1
        )
        return frame

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

    @property
    def fps(self) -> float:
        return self._fps


def _centroid_distance_sq(a: Detection, b: Detection) -> float:
    dx = a.cx - b.cx
    dy = a.cy - b.cy
    return dx * dx + dy * dy
