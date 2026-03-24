"""Unit tests for target motion estimation."""
from robot.vision.detector import Detection
from robot.vision.motion import TargetMotionEstimator


def _det(x: int, y: int, w: int = 80, h: int = 120) -> Detection:
    return Detection(label="person", confidence=0.9, x=x, y=y, w=w, h=h, track_id=1)


class TestTargetMotionEstimator:
    def test_first_detection_is_stationary(self):
        estimator = TargetMotionEstimator()
        motion = estimator.update(_det(100, 50))
        assert motion.summary == "stationary"

    def test_detects_left_motion(self):
        estimator = TargetMotionEstimator()
        estimator.update(_det(120, 50))
        motion = estimator.update(_det(80, 50))
        assert motion.horizontal == "left"
        assert "left" in motion.summary

    def test_detects_right_motion(self):
        estimator = TargetMotionEstimator()
        estimator.update(_det(80, 50))
        motion = estimator.update(_det(120, 50))
        assert motion.horizontal == "right"
        assert "right" in motion.summary

    def test_detects_forward_motion_from_box_growth(self):
        estimator = TargetMotionEstimator()
        estimator.update(_det(100, 50, w=60, h=90))
        motion = estimator.update(_det(100, 50, w=90, h=135))
        assert motion.depth == "forward"
        assert "forward" in motion.summary

    def test_detects_backward_motion_from_box_shrink(self):
        estimator = TargetMotionEstimator()
        estimator.update(_det(100, 50, w=90, h=135))
        motion = estimator.update(_det(100, 50, w=60, h=90))
        assert motion.depth == "backward"
        assert "backward" in motion.summary

    def test_combines_forward_and_right(self):
        estimator = TargetMotionEstimator()
        estimator.update(_det(80, 50, w=60, h=90))
        motion = estimator.update(_det(120, 50, w=90, h=135))
        assert motion.horizontal == "right"
        assert motion.depth == "forward"
        assert motion.summary == "forward + right"
