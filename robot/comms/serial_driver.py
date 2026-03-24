"""
ESP32SerialDriver — async-friendly serial communication with the motor firmware.
Supports mock mode for development without hardware.
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from typing import Generator, Optional

from robot.comms.protocol import decode_response, encode_motor_command, encode_ping
from robot.control.motor import MotorCommand
from robot.config import SerialConfig
from robot.logger import logger


class ESP32SerialDriver:
    """
    Manages serial connection to ESP32.

    If `config.mock=True`, commands are logged instead of sent.
    Thread-safe: uses a lock for all port operations.

    Usage
    -----
    driver = ESP32SerialDriver(config)
    driver.connect()
    driver.send_motor(MotorCommand(100, 80))
    driver.disconnect()

    Context manager supported:
    with ESP32SerialDriver(config) as d:
        d.send_motor(cmd)
    """

    def __init__(self, config: Optional[SerialConfig] = None) -> None:
        self._cfg = config or SerialConfig()
        self._port = None
        self._lock = threading.Lock()
        self._connected = False
        self._last_cmd: Optional[MotorCommand] = None
        self._cmd_count = 0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if self._cfg.mock:
            logger.info("SerialDriver: MOCK mode — no hardware needed")
            self._connected = True
            return True

        try:
            import serial
            self._port = serial.Serial(
                port=self._cfg.port,
                baudrate=self._cfg.baudrate,
                timeout=self._cfg.timeout_s,
            )
            time.sleep(2)  # ESP32 reboot after DTR toggle
            self._connected = True
            logger.info(f"Serial connected: {self._cfg.port} @ {self._cfg.baudrate}")
            self._ping()
            return True
        except Exception as exc:
            logger.error(f"Serial connect failed: {exc}")
            return False

    def disconnect(self) -> None:
        if self._port and self._port.is_open:
            self.send_motor(MotorCommand(0, 0))  # safety stop
            self._port.close()
        self._connected = False
        logger.info("Serial disconnected")

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Command sending
    # ------------------------------------------------------------------

    def send_motor(self, cmd: MotorCommand) -> bool:
        """Send motor command. Returns True on success."""
        with self._lock:
            self._last_cmd = cmd
            self._cmd_count += 1

            if self._cfg.mock:
                logger.debug(f"[MOCK] {cmd}")
                return True

            if not self._connected or self._port is None:
                logger.warning("send_motor called but not connected")
                return False

            try:
                data = encode_motor_command(cmd)
                self._port.write(data)
                self._port.flush()
                return True
            except Exception as exc:
                logger.error(f"Serial write error: {exc}")
                self._connected = False
                return False

    async def send_motor_async(self, cmd: MotorCommand) -> bool:
        """Async wrapper — runs serial write in executor thread."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.send_motor, cmd
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "mock": self._cfg.mock,
            "port": self._cfg.port,
            "cmd_count": self._cmd_count,
            "last_cmd": self._last_cmd.to_dict() if self._last_cmd else None,
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ESP32SerialDriver":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ping(self) -> None:
        if self._port:
            try:
                self._port.write(encode_ping())
                resp = self._port.readline()
                logger.debug(f"Ping response: {decode_response(resp)}")
            except Exception:
                pass
