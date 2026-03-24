"""
Production PID controller with anti-windup and output clamping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PIDState:
    integral: float = 0.0
    prev_error: float = 0.0
    prev_time: float = field(default_factory=time.monotonic)


class PIDController:
    """
    Discrete PID controller.

    Parameters
    ----------
    kp, ki, kd     : PID gains
    integral_limit : Anti-windup clamp on integral term
    output_limit   : Symmetric clamp on final output
    use_dt         : If True, scale I/D by actual elapsed time (recommended
                     for variable loop rates).  If False, assumes dt=1.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float = 500.0,
        output_limit: float = 150.0,
        use_dt: bool = True,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.use_dt = use_dt
        self._state = PIDState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, error: float) -> float:
        """Compute PID output for a given error signal."""
        now = time.monotonic()
        dt = now - self._state.prev_time if self.use_dt else 1.0
        dt = max(dt, 1e-6)  # avoid divide-by-zero

        # Proportional
        p = self.kp * error

        # Integral with anti-windup
        self._state.integral += error * dt
        self._state.integral = _clamp(
            self._state.integral, -self.integral_limit, self.integral_limit
        )
        i = self.ki * self._state.integral

        # Derivative (on error, not measurement — suitable for setpoint changes)
        d = self.kd * (error - self._state.prev_error) / dt

        output = p + i + d
        output = _clamp(output, -self.output_limit, self.output_limit)

        self._state.prev_error = error
        self._state.prev_time = now
        return output

    def reset(self) -> None:
        """Reset internal state (call when switching modes)."""
        self._state = PIDState()

    # Tuneable properties
    @property
    def gains(self) -> tuple[float, float, float]:
        return self.kp, self.ki, self.kd

    @gains.setter
    def gains(self, value: tuple[float, float, float]) -> None:
        self.kp, self.ki, self.kd = value
        self.reset()

    def __repr__(self) -> str:
        return (
            f"PIDController(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"integral_limit={self.integral_limit}, output_limit={self.output_limit})"
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
