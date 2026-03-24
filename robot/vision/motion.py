"""Motion direction estimation for a tracked detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from robot.vision.detector import Detection


@dataclass
class MotionEstimate:
    horizontal: str = "stationary"
    depth: str = "stationary"
    summary: str = "stationary"
    dx: float = 0.0
    dy: float = 0.0
    scale_change: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "horizontal": self.horizontal,
            "depth": self.depth,
            "summary": self.summary,
            "dx": round(self.dx, 2),
            "dy": round(self.dy, 2),
            "scale_change": round(self.scale_change, 4),
        }


class TargetMotionEstimator:
    """
    Estimates whether the followed target moved left/right or forward/backward.

    Forward/backward is inferred from bounding-box area change:
    - area grows   -> target moved closer to camera -> forward
    - area shrinks -> target moved away             -> backward
    """

    def __init__(
        self,
        horizontal_threshold_px: float = 12.0,
        vertical_threshold_px: float = 10.0,
        scale_threshold: float = 0.08,
    ) -> None:
        self._horizontal_threshold_px = horizontal_threshold_px
        self._vertical_threshold_px = vertical_threshold_px
        self._scale_threshold = scale_threshold
        self._previous: Optional[Detection] = None
        self._latest = MotionEstimate()

    @property
    def latest(self) -> MotionEstimate:
        return self._latest

    def reset(self) -> None:
        self._previous = None
        self._latest = MotionEstimate()

    def update(self, target: Optional[Detection]) -> MotionEstimate:
        if target is None:
            self.reset()
            return self._latest

        if self._previous is None:
            self._previous = _copy_detection(target)
            self._latest = MotionEstimate()
            return self._latest

        dx = target.cx - self._previous.cx
        dy = target.cy - self._previous.cy
        prev_area = max(float(self._previous.area), 1.0)
        scale_change = (target.area - self._previous.area) / prev_area

        horizontal = _horizontal_direction(dx, self._horizontal_threshold_px)
        _ = _vertical_direction(dy, self._vertical_threshold_px)
        depth = _depth_direction(scale_change, self._scale_threshold)

        summary_parts = [part for part in (depth, horizontal) if part != "stationary"]
        summary = " + ".join(summary_parts) if summary_parts else "stationary"

        self._latest = MotionEstimate(
            horizontal=horizontal,
            depth=depth,
            summary=summary,
            dx=dx,
            dy=dy,
            scale_change=scale_change,
        )
        self._previous = _copy_detection(target)
        return self._latest


def _horizontal_direction(dx: float, threshold: float) -> str:
    if dx <= -threshold:
        return "left"
    if dx >= threshold:
        return "right"
    return "stationary"


def _vertical_direction(dy: float, threshold: float) -> str:
    if dy <= -threshold:
        return "up"
    if dy >= threshold:
        return "down"
    return "stationary"


def _depth_direction(scale_change: float, threshold: float) -> str:
    if scale_change >= threshold:
        return "forward"
    if scale_change <= -threshold:
        return "backward"
    return "stationary"


def _copy_detection(target: Detection) -> Detection:
    return Detection(
        label=target.label,
        confidence=target.confidence,
        x=target.x,
        y=target.y,
        w=target.w,
        h=target.h,
        track_id=target.track_id,
    )
