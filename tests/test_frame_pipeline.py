"""Unit tests for FramePipeline target locking."""
from robot.config import VisionConfig
from robot.vision.detector import Detection
from robot.vision.frame_pipeline import FramePipeline


def _det(label: str, x: int, y: int, w: int, h: int, track_id=None) -> Detection:
    return Detection(
        label=label,
        confidence=0.9,
        x=x,
        y=y,
        w=w,
        h=h,
        track_id=track_id,
    )


class TestFramePipelineSelection:
    def test_select_target_at_point_locks_smallest_box_under_click(self):
        pipeline = FramePipeline(VisionConfig(target_class="person", tracker="none"))
        outer = _det("person", 20, 20, 140, 140, track_id=1)
        inner = _det("person", 60, 60, 40, 40, track_id=2)

        selected = pipeline.select_target_at_point(70, 70, [outer, inner])

        assert selected is inner
        assert pipeline.locked_target is inner

    def test_locked_target_prefers_same_track_id(self):
        pipeline = FramePipeline(VisionConfig(target_class="person", tracker="none"))
        pipeline.lock_target(_det("person", 40, 50, 60, 80, track_id=9))

        matched = pipeline._select_target(
            [
                _det("person", 200, 40, 100, 120, track_id=3),
                _det("person", 45, 55, 60, 80, track_id=9),
            ]
        )

        assert matched is not None
        assert matched.track_id == 9

    def test_locked_target_does_not_jump_to_different_label(self):
        pipeline = FramePipeline(VisionConfig(target_class="person", tracker="none"))
        pipeline.lock_target(_det("bottle", 40, 50, 30, 70))

        matched = pipeline._select_target(
            [
                _det("person", 42, 52, 80, 120),
                _det("bottle", 48, 56, 30, 70),
            ]
        )

        assert matched is not None
        assert matched.label == "bottle"

    def test_without_lock_falls_back_to_largest_target_class(self):
        pipeline = FramePipeline(VisionConfig(target_class="person", tracker="none"))

        matched = pipeline._select_target(
            [
                _det("person", 10, 20, 40, 60),
                _det("person", 100, 20, 80, 120),
                _det("dog", 50, 50, 120, 120),
            ]
        )

        assert matched is not None
        assert matched.label == "person"
        assert matched.area == 80 * 120
