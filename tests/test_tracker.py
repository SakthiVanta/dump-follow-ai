"""Unit tests for ByteTrackWrapper."""
import pytest
from robot.vision.tracker import ByteTrackWrapper, _iou
from robot.vision.detector import Detection


def _make_det(x, y, w=80, h=120, label="person"):
    return Detection(label=label, confidence=0.9, x=x, y=y, w=w, h=h)


class TestIoU:
    def test_identical_boxes(self):
        a = _make_det(0, 0, 100, 100)
        b = _make_det(0, 0, 100, 100)
        assert _iou(a, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = _make_det(0, 0, 50, 50)
        b = _make_det(100, 100, 50, 50)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _make_det(0, 0, 100, 100)
        b = _make_det(50, 50, 100, 100)
        assert 0 < _iou(a, b) < 1.0


class TestByteTrackWrapper:
    def test_single_detection_gets_id(self):
        tracker = ByteTrackWrapper()
        dets = [_make_det(100, 50)]
        result = tracker.update(dets)
        assert result[0].track_id is not None

    def test_same_box_gets_same_id_on_second_frame(self):
        tracker = ByteTrackWrapper()
        d1 = [_make_det(100, 50)]
        r1 = tracker.update(d1)
        id1 = r1[0].track_id

        d2 = [_make_det(105, 55)]  # slight movement
        r2 = tracker.update(d2)
        id2 = r2[0].track_id

        assert id1 == id2

    def test_disjoint_box_gets_new_id(self):
        tracker = ByteTrackWrapper()
        d1 = [_make_det(0, 0)]
        r1 = tracker.update(d1)
        id1 = r1[0].track_id

        d2 = [_make_det(300, 200)]  # completely different position
        r2 = tracker.update(d2)
        id2 = r2[0].track_id

        assert id1 != id2

    def test_empty_detections_clears_tracks(self):
        tracker = ByteTrackWrapper()
        tracker.update([_make_det(100, 50)])
        result = tracker.update([])
        assert result == []
        assert tracker._tracks == {}

    def test_reset_clears_tracks(self):
        tracker = ByteTrackWrapper()
        tracker.update([_make_det(100, 50)])
        tracker.reset()
        assert tracker._tracks == {}

    def test_multiple_detections_get_unique_ids(self):
        tracker = ByteTrackWrapper()
        dets = [_make_det(0, 0), _make_det(200, 0), _make_det(100, 200)]
        result = tracker.update(dets)
        ids = [d.track_id for d in result]
        assert len(set(ids)) == 3
