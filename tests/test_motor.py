"""Unit tests for MotorCommand and MotorController."""
import pytest
from robot.control.motor import MotorCommand, MotorController, STOP
from robot.config import ControlConfig, MotorConfig


class TestMotorCommand:
    def test_clamp_upper(self):
        cmd = MotorCommand(300, 300).clamp(-200, 200)
        assert cmd.left == 200
        assert cmd.right == 200

    def test_clamp_lower(self):
        cmd = MotorCommand(-300, -300).clamp(-200, 200)
        assert cmd.left == -200
        assert cmd.right == -200

    def test_to_dict(self):
        d = MotorCommand(100, -50).to_dict()
        assert d == {"left": 100, "right": -50}

    def test_str_repr(self):
        s = str(MotorCommand(100, -50))
        assert "L=" in s and "R=" in s

    def test_stop_is_zero(self):
        assert STOP.left == 0 and STOP.right == 0

    def test_int_cast(self):
        cmd = MotorCommand(100.7, 99.3)  # type: ignore
        assert isinstance(cmd.left, int)
        assert isinstance(cmd.right, int)


class TestMotorController:
    def test_forward(self, motor_ctrl, control_config):
        cmd = motor_ctrl.forward()
        assert cmd.left == control_config.base_speed
        assert cmd.right == control_config.base_speed

    def test_stop(self, motor_ctrl):
        cmd = motor_ctrl.stop()
        assert cmd.left == 0 and cmd.right == 0

    def test_search_spins_in_place(self, motor_ctrl, control_config):
        cmd = motor_ctrl.search()
        assert cmd.left == -control_config.search_turn_speed
        assert cmd.right == control_config.search_turn_speed

    def test_from_pid_zero_turn_is_straight(self, motor_ctrl, control_config):
        cmd = motor_ctrl.from_pid(0.0)
        assert cmd.left == control_config.base_speed
        assert cmd.right == control_config.base_speed

    def test_from_pid_positive_turn_right(self, motor_ctrl, control_config):
        """Positive turn = target is right → right motor slower."""
        cmd = motor_ctrl.from_pid(50.0)
        assert cmd.right < cmd.left

    def test_from_pid_negative_turn_left(self, motor_ctrl, control_config):
        cmd = motor_ctrl.from_pid(-50.0)
        assert cmd.left < cmd.right

    def test_invert_left(self, control_config):
        mc = MotorController(control_config, MotorConfig(invert_left=True))
        fwd = mc.forward()
        assert fwd.left < 0   # inverted
        assert fwd.right > 0

    def test_invert_right(self, control_config):
        mc = MotorController(control_config, MotorConfig(invert_right=True))
        fwd = mc.forward()
        assert fwd.left > 0
        assert fwd.right < 0

    def test_backward_negative_speeds(self, motor_ctrl, control_config):
        cmd = motor_ctrl.backward()
        assert cmd.left < 0 and cmd.right < 0

    def test_speed_clamped(self, control_config, motor_config):
        mc = MotorController(control_config, motor_config)
        cmd = mc.from_pid(9999)  # extreme turn
        assert cmd.left >= control_config.min_speed
        assert cmd.right <= control_config.max_speed
