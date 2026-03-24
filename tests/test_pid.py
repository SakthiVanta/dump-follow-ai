"""Unit tests for PIDController."""
import pytest
from robot.control.pid import PIDController, _clamp


class TestClamp:
    def test_within_bounds(self):
        assert _clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_low(self):
        assert _clamp(-5.0, 0.0, 10.0) == 0.0

    def test_above_high(self):
        assert _clamp(15.0, 0.0, 10.0) == 10.0

    def test_symmetric(self):
        assert _clamp(-200, -150, 150) == -150
        assert _clamp(200, -150, 150) == 150


class TestPIDController:
    def test_zero_error_zero_output(self, pid):
        assert pid.compute(0.0) == 0.0

    def test_positive_error_positive_output(self, pid):
        out = pid.compute(1.0)
        assert out > 0

    def test_negative_error_negative_output(self, pid):
        out = pid.compute(-1.0)
        assert out < 0

    def test_output_clamped_at_limit(self):
        pid = PIDController(kp=1000, ki=0, kd=0, output_limit=150, use_dt=False)
        out = pid.compute(1.0)
        assert out == pytest.approx(150.0)

    def test_output_clamped_negative(self):
        pid = PIDController(kp=1000, ki=0, kd=0, output_limit=150, use_dt=False)
        out = pid.compute(-1.0)
        assert out == pytest.approx(-150.0)

    def test_integral_accumulates(self, pid):
        """Calling with same positive error repeatedly increases the internal integral."""
        pid.compute(0.5)
        integral_after_1 = pid._state.integral
        pid.compute(0.5)
        integral_after_2 = pid._state.integral
        assert integral_after_2 > integral_after_1  # integral grows

    def test_integral_anti_windup(self):
        pid = PIDController(kp=0, ki=1.0, kd=0, integral_limit=10.0, output_limit=999, use_dt=False)
        for _ in range(1000):
            pid.compute(1.0)
        # Integral should be capped at 10, so output = ki * 10 = 10
        assert abs(pid.compute(0.0)) <= 10.0 + 1e-6

    def test_reset_clears_state(self, pid):
        pid.compute(10.0)
        pid.reset()
        assert pid._state.integral == 0.0
        assert pid._state.prev_error == 0.0

    def test_repr(self, pid):
        r = repr(pid)
        assert "PIDController" in r
        assert "kp=" in r

    def test_gains_setter_resets_state(self, pid):
        pid.compute(5.0)  # dirty state
        pid.gains = (1.0, 0.0, 0.0)
        assert pid._state.integral == 0.0

    def test_proportional_only(self):
        pid = PIDController(kp=2.0, ki=0, kd=0, use_dt=False)
        assert pid.compute(3.0) == pytest.approx(6.0)

    def test_derivative_dampens_overshoot(self):
        """With kd>0 the first step should be larger than the second (dampened)."""
        pid = PIDController(kp=0, ki=0, kd=1.0, use_dt=False)
        first = abs(pid.compute(1.0))
        second = abs(pid.compute(1.0))  # derivative = 0 for same error
        assert second < first
