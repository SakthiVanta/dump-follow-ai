"""Tracker wrappers: ByteTrack (built-in Ultralytics) and DeepSORT."""
from __future__ import annotations

from typing import Optional

import numpy as np

from robot.vision.detector import Detection
from robot.logger import logger


class ByteTrackWrapper:
    """
    Uses Ultralytics built-in ByteTrack to assign persistent track IDs.
    Operates on raw detections from YOLODetector.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, Detection] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        """
        Assign synthetic track IDs based on IoU overlap with previous frame.
        Production systems should use the ultralytics tracker directly on the
        YOLO result object — this is a standalone convenience wrapper.
        """
        if not detections:
            self._tracks.clear()
            return detections

        # For each detection find best-matching previous track by IoU
        updated: list[Detection] = []
        used_ids: set[int] = set()

        for det in detections:
            best_id: Optional[int] = None
            best_iou = 0.3  # min IoU threshold to continue a track

            for tid, prev in self._tracks.items():
                if tid in used_ids:
                    continue
                iou = _iou(det, prev)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is None:
                best_id = _next_id()

            used_ids.add(best_id)
            det.track_id = best_id
            updated.append(det)

        # Refresh track memory
        self._tracks = {d.track_id: d for d in updated if d.track_id is not None}
        return updated

    def reset(self) -> None:
        self._tracks.clear()


class DeepSORTWrapper:
    """
    Wrapper around deep-sort-realtime for appearance-based tracking.
    Falls back gracefully if library not installed.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(max_age=max_age, n_init=n_init)
            self._available = True
            logger.info("DeepSORT tracker initialised")
        except ImportError:
            logger.warning(
                "deep-sort-realtime not installed; falling back to ByteTrack"
            )
            self._available = False
            self._fallback = ByteTrackWrapper()

    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[Detection]:
        if not self._available:
            return self._fallback.update(detections)

        raw = [
            ([d.x, d.y, d.w, d.h], d.confidence, d.label)
            for d in detections
        ]
        tracks = self._tracker.update_tracks(raw, frame=frame)

        id_map: dict[tuple, int] = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            key = (int(ltrb[0]), int(ltrb[1]))
            id_map[key] = track.track_id

        for det in detections:
            det.track_id = id_map.get((det.x, det.y))

        return detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_id_counter = 0


def _next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


def _iou(a: Detection, b: Detection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h

    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
