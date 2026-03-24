"""Shared robot state accessible from all API routes."""
from __future__ import annotations

import asyncio
from typing import Optional

from robot.comms.serial_driver import ESP32SerialDriver
from robot.config import RobotConfig
from robot.control.follower import PersonFollower, RobotMode
from robot.control.motor import MotorCommand
from robot.logger import logger


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
        self.frame_count: int = 0
        self._cmd_lock = asyncio.Lock()

    async def startup(self) -> None:
        ok = self.serial.connect()
        if not ok:
            logger.warning("Serial not connected — running in mock mode")
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
