"""Unit tests for PersonFollower state machine."""
import time
import pytest
from robot.control.follower import PersonFollower, RobotMode
from robot.vision.detector import Detection


FRAME_W, FRAME_H = 320, 240


class TestPersonFollower:
    def test_stop_mode_returns_zero(self, follower):
        follower.mode = RobotMode.STOP
        cmd = follower.update(None, FRAME_W, FRAME_H)
        assert cmd.left == 0 and cmd.right == 0

    def test_no_target_returns_search(self, follower):
        follower.mode = RobotMode.FOLLOW
        follower._last_seen = None
        cmd = follower.update(None, FRAME_W, FRAME_H)
        # Search spins: left < 0, right > 0
        assert cmd.right > 0

    def test_centered_target_goes_straight(self, follower, person_detection_center):
        follower.mode = RobotMode.FOLLOW
        cmd = follower.update(person_detection_center, FRAME_W, FRAME_H)
        # Very small error — nearly equal speeds
        assert abs(cmd.left - cmd.right) < 30

    def test_left_target_turns_left(self, follower, person_detection_left):
        follower.mode = RobotMode.FOLLOW
        cmd = follower.update(person_detection_left, FRAME_W, FRAME_H)
        # Target is left → left motor slower (or more negative)
        assert cmd.left < cmd.right

    def test_right_target_turns_right(self, follower, person_detection_right):
        follower.mode = RobotMode.FOLLOW
        cmd = follower.update(person_detection_right, FRAME_W, FRAME_H)
        assert cmd.right < cmd.left

    def test_too_close_stops(self, follower, person_detection_close):
        follower.mode = RobotMode.FOLLOW
        cmd = follower.update(person_detection_close, FRAME_W, FRAME_H)
        assert cmd.left == 0 and cmd.right == 0

    def test_mode_transition_resets_pid(self, follower):
        follower.mode = RobotMode.FOLLOW
        follower._pid.compute(50.0)  # dirty state
        follower.mode = RobotMode.STOP
        assert follower._pid._state.integral == 0.0

    def test_set_speed(self, follower):
        follower.set_speed(150)
        assert follower._ctrl.base_speed == 150

    def test_brief_loss_holds_stop(self, follower, person_detection_center):
        follower.mode = RobotMode.FOLLOW
        # First update establishes last_seen
        follower.update(person_detection_center, FRAME_W, FRAME_H)
        # Immediately lose target — within timeout → stop
        cmd = follower.update(None, FRAME_W, FRAME_H)
        assert cmd.left == 0 and cmd.right == 0

    def test_long_loss_triggers_search(self, follower, control_config):
        follower.mode = RobotMode.FOLLOW
        follower._last_seen = time.monotonic() - control_config.lost_target_timeout_s - 1
        cmd = follower.update(None, FRAME_W, FRAME_H)
        # Search — right > left
        assert cmd.right > 0
