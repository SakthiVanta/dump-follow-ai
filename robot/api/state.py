"""Shared robot state accessible from all API routes."""
from __future__ import annotations

import asyncio
from typing import Optional

from robot.comms.serial_driver import ESP32SerialDriver
from robot.config import RobotConfig
from robot.control.follower import PersonFollower, RobotMode
from robot.control.motor import MotorCommand
from robot.logger import logger
from robot.vision.motion import MotionEstimate, TargetMotionEstimator
from robot.vision.svsp import (
    SVSPMotionPredictor,
    SVSPTrainingResult,
    load_svsp_model,
    train_svsp_model,
)


class RobotState:
    """
    Holds all runtime objects shared across API routes.
    Started/stopped by the FastAPI lifespan.
    """

    def __init__(self, config: RobotConfig, mock: bool = False) -> None:
        self.config = config
        self.mock = mock or config.serial.mock
        self.serial = ESP32SerialDriver(config.serial)
        self.follower = PersonFollower(
            config.control, config.motor, config.control.pid
        )
        self.current_mode: RobotMode = RobotMode.STOP
        self.last_command: Optional[MotorCommand] = None
        self.last_motion: MotionEstimate = MotionEstimate()
        self.motion_source: str = "heuristic"
        self.frame_count: int = 0
        self._cmd_lock = asyncio.Lock()
        self.motion_estimator = TargetMotionEstimator()
        self.svsp_predictor: Optional[SVSPMotionPredictor] = None

    async def startup(self) -> None:
        ok = self.serial.connect()
        if not ok:
            logger.warning("Serial not connected; running in mock mode")
        if self.config.motion_model.enabled:
            try:
                self.load_svsp_model()
                logger.info("SVSP motion model loaded")
            except FileNotFoundError:
                logger.warning(
                    "SVSP motion model enabled but file not found; using heuristic motion"
                )
        logger.info("RobotState ready")

    async def shutdown(self) -> None:
        await self.send_command(MotorCommand(0, 0))
        self.serial.disconnect()

    async def send_command(self, cmd: MotorCommand) -> bool:
        async with self._cmd_lock:
            self.last_command = cmd
            return await self.serial.send_motor_async(cmd)

    def set_mode(self, mode: RobotMode) -> None:
        self.follower.mode = mode
        self.current_mode = mode

    def update_motion(self, target) -> MotionEstimate:
        if self.svsp_predictor is not None:
            self.motion_source = "svsp"
            self.last_motion = self.svsp_predictor.update(target)
            return self.last_motion

        self.motion_source = "heuristic"
        self.last_motion = self.motion_estimator.update(target)
        return self.last_motion

    def train_svsp(self) -> SVSPTrainingResult:
        result = train_svsp_model(self.config.motion_model)
        self.load_svsp_model(result.model_path)
        return result

    def load_svsp_model(self, model_path: str | None = None) -> str:
        path = model_path or self.config.motion_model.model_path
        model = load_svsp_model(path)
        self.svsp_predictor = SVSPMotionPredictor(
            model=model,
            frame_width=self.config.vision.frame_width,
            frame_height=self.config.vision.frame_height,
        )
        self.motion_source = "svsp"
        self.config.motion_model.model_path = str(path)
        self.config.motion_model.enabled = True
        return str(path)

    def disable_svsp_model(self) -> None:
        self.svsp_predictor = None
        self.config.motion_model.enabled = False
        self.motion_source = "heuristic"
