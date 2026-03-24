"""
Shared pytest fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest

from robot.config import (
    ControlConfig,
    MotorConfig,
    PIDConfig,
    SerialConfig,
    VisionConfig,
    VoiceConfig,
    RobotConfig,
)
from robot.control.follower import PersonFollower
from robot.control.motor import MotorController
from robot.control.pid import PIDController
from robot.vision.detector import Detection


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pid_config() -> PIDConfig:
    return PIDConfig(kp=0.4, ki=0.01, kd=0.1, integral_limit=500, output_limit=150)


@pytest.fixture
def control_config(pid_config) -> ControlConfig:
    return ControlConfig(
        base_speed=100,
        max_speed=200,
        min_speed=-200,
        pid=pid_config,
        safe_distance_ratio=0.35,
        search_turn_speed=50,
        lost_target_timeout_s=2.0,
    )


@pytest.fixture
def motor_config() -> MotorConfig:
    return MotorConfig(invert_left=False, invert_right=False)


@pytest.fixture
def serial_config() -> SerialConfig:
    return SerialConfig(port="COM99", baudrate=115200, mock=True)


@pytest.fixture
def vision_config() -> VisionConfig:
    return VisionConfig(
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        frame_width=320,
        frame_height=240,
        camera_id=0,
        tracker="none",
    )


@pytest.fixture
def robot_config(
    control_config, motor_config, serial_config, vision_config
) -> RobotConfig:
    return RobotConfig(
        control=control_config,
        motor=motor_config,
        serial=serial_config,
        vision=vision_config,
    )


# ---------------------------------------------------------------------------
# Component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pid(pid_config) -> PIDController:
    return PIDController(
        kp=pid_config.kp,
        ki=pid_config.ki,
        kd=pid_config.kd,
        integral_limit=pid_config.integral_limit,
        output_limit=pid_config.output_limit,
        use_dt=False,   # deterministic for tests
    )


@pytest.fixture
def motor_ctrl(control_config, motor_config) -> MotorController:
    return MotorController(control_config, motor_config)


@pytest.fixture
def follower(control_config, motor_config, pid_config) -> PersonFollower:
    return PersonFollower(control_config, motor_config, pid_config)


# ---------------------------------------------------------------------------
# Detection fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def person_detection_center() -> Detection:
    """Person bbox centred at x=160, y=120 in a 320x240 frame."""
    return Detection(label="person", confidence=0.9, x=120, y=60, w=80, h=120)


@pytest.fixture
def person_detection_left() -> Detection:
    """Person bbox to the left of centre — h=60 so not 'too close' (60/240=0.25 < 0.35)."""
    return Detection(label="person", confidence=0.85, x=20, y=100, w=80, h=60)


@pytest.fixture
def person_detection_right() -> Detection:
    """Person bbox to the right of centre — h=60 so not 'too close'."""
    return Detection(label="person", confidence=0.85, x=220, y=100, w=80, h=60)


@pytest.fixture
def person_detection_close() -> Detection:
    """Person bbox that fills most of the frame height (too close)."""
    return Detection(label="person", confidence=0.9, x=60, y=10, w=200, h=220)


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """320x240 BGR frame with a white rectangle simulating a person."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[60:180, 120:200] = 255
    return frame
