"""
Motor command dataclass and controller that bridges PID → serial driver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from robot.config import ControlConfig, MotorConfig
from robot.logger import logger


@dataclass
class MotorCommand:
    left: int
    right: int

    def __post_init__(self) -> None:
        self.left = int(self.left)
        self.right = int(self.right)

    def clamp(self, lo: int = -200, hi: int = 200) -> "MotorCommand":
        return MotorCommand(
            left=max(lo, min(hi, self.left)),
            right=max(lo, min(hi, self.right)),
        )

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "right": self.right}

    def __str__(self) -> str:
        return f"Motor(L={self.left:+d}, R={self.right:+d})"


STOP = MotorCommand(0, 0)


class MotorController:
    """
    Translates a PID turn signal into differential motor commands.

    Parameters
    ----------
    control_cfg: ControlConfig holding base_speed, max/min, search speed.
    motor_cfg:   MotorConfig holding invert flags.
    """

    def __init__(
        self,
        control_cfg: Optional[ControlConfig] = None,
        motor_cfg: Optional[MotorConfig] = None,
    ) -> None:
        self._ctrl = control_cfg or ControlConfig()
        self._motor = motor_cfg or MotorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_pid(self, turn_signal: float) -> MotorCommand:
        """
        Convert PID turn output to (left, right) motor speeds.

        A positive turn_signal means the target is to the right →
        robot should turn right (right motor slower / reverse).
        """
        base = self._ctrl.base_speed
        left = base + turn_signal
        right = base - turn_signal
        cmd = MotorCommand(int(left), int(right)).clamp(
            self._ctrl.min_speed, self._ctrl.max_speed
        )
        return self._apply_invert(cmd)

    def search(self) -> MotorCommand:
        """Spin in place to search for target."""
        speed = self._ctrl.search_turn_speed
        cmd = MotorCommand(-speed, speed)
        return self._apply_invert(cmd)

    def stop(self) -> MotorCommand:
        return STOP

    def forward(self, speed: Optional[int] = None) -> MotorCommand:
        s = speed if speed is not None else self._ctrl.base_speed
        cmd = MotorCommand(s, s).clamp(self._ctrl.min_speed, self._ctrl.max_speed)
        return self._apply_invert(cmd)

    def backward(self, speed: Optional[int] = None) -> MotorCommand:
        s = -(speed if speed is not None else self._ctrl.base_speed)
        cmd = MotorCommand(s, s).clamp(self._ctrl.min_speed, self._ctrl.max_speed)
        return self._apply_invert(cmd)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _apply_invert(self, cmd: MotorCommand) -> MotorCommand:
        left = -cmd.left if self._motor.invert_left else cmd.left
        right = -cmd.right if self._motor.invert_right else cmd.right
        return MotorCommand(left, right)
