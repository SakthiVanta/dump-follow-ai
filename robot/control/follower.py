"""
PersonFollower — fuses vision output + PID control into motor commands.
Handles mode transitions: follow, search, stop, manual.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from robot.config import ControlConfig, MotorConfig, PIDConfig
from robot.control.motor import MotorCommand, MotorController
from robot.control.pid import PIDController
from robot.vision.detector import Detection
from robot.logger import logger


class RobotMode(str, Enum):
    FOLLOW = "follow_person"
    SEARCH = "search"
    STOP = "stop"
    MANUAL = "manual"
    IDLE = "idle"


class PersonFollower:
    """
    Core brain: decides how to move given a vision detection.

    Usage
    -----
    follower = PersonFollower(control_cfg, motor_cfg, pid_cfg)
    cmd = follower.update(target_detection, frame_width, frame_height)
    """

    def __init__(
        self,
        control_cfg: Optional[ControlConfig] = None,
        motor_cfg: Optional[MotorConfig] = None,
        pid_cfg: Optional[PIDConfig] = None,
    ) -> None:
        self._ctrl = control_cfg or ControlConfig()
        pid = pid_cfg or self._ctrl.pid
        self._pid = PIDController(
            kp=pid.kp,
            ki=pid.ki,
            kd=pid.kd,
            integral_limit=pid.integral_limit,
            output_limit=pid.output_limit,
        )
        self._motors = MotorController(self._ctrl, motor_cfg or MotorConfig())
        self._mode: RobotMode = RobotMode.FOLLOW
        self._last_seen: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mode(self) -> RobotMode:
        return self._mode

    @mode.setter
    def mode(self, value: RobotMode) -> None:
        if value != self._mode:
            logger.info(f"Mode: {self._mode.value} → {value.value}")
            self._mode = value
            self._pid.reset()

    def update(
        self,
        target: Optional[Detection],
        frame_width: int,
        frame_height: int,
    ) -> MotorCommand:
        """
        Main control step.  Call once per frame.

        Returns the MotorCommand to send to the ESP32.
        """
        if self._mode == RobotMode.STOP:
            return self._motors.stop()

        if self._mode == RobotMode.MANUAL:
            return self._motors.stop()  # manual commands sent externally

        if target is None:
            return self._handle_no_target()

        # ---- Target acquired ----
        self._last_seen = time.monotonic()

        # Proximity guard: stop if too close
        if self._is_too_close(target, frame_height):
            logger.debug("Too close — stopping")
            return self._motors.stop()

        # Compute horizontal error (pixels from frame centre)
        error = target.cx - frame_width / 2.0

        # Normalise to [-1, 1]
        error_norm = error / (frame_width / 2.0)

        turn = self._pid.compute(error_norm)
        cmd = self._motors.from_pid(turn)
        logger.debug(f"Target cx={target.cx:.0f} err={error_norm:.3f} turn={turn:.1f} {cmd}")
        return cmd

    def set_speed(self, speed: int) -> None:
        """Dynamically update base speed."""
        self._ctrl.base_speed = speed

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _handle_no_target(self) -> MotorCommand:
        now = time.monotonic()
        if self._last_seen is None:
            return self._motors.search()

        elapsed = now - self._last_seen
        if elapsed > self._ctrl.lost_target_timeout_s:
            logger.debug("Target lost — searching")
            return self._motors.search()

        # Brief loss — hold last command (stop is safer)
        return self._motors.stop()

    def _is_too_close(self, target: Detection, frame_height: int) -> bool:
        ratio = target.h / frame_height
        return ratio > self._ctrl.safe_distance_ratio
