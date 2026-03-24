"""Unit tests for ESP32SerialDriver (mock mode only)."""
import pytest
from robot.comms.serial_driver import ESP32SerialDriver
from robot.comms.protocol import (
    encode_motor_command,
    decode_response,
    encode_ping,
    encode_stop,
)
from robot.control.motor import MotorCommand
from robot.config import SerialConfig


@pytest.fixture
def mock_driver():
    cfg = SerialConfig(port="COM99", mock=True)
    driver = ESP32SerialDriver(cfg)
    driver.connect()
    return driver


class TestProtocol:
    def test_encode_motor_command(self):
        data = encode_motor_command(MotorCommand(100, -50))
        assert data == b"M,100,-50\n"

    def test_encode_motor_zero(self):
        data = encode_motor_command(MotorCommand(0, 0))
        assert data == b"M,0,0\n"

    def test_decode_ok(self):
        result = decode_response(b"OK\n")
        assert result["status"] == "ok"

    def test_decode_error(self):
        result = decode_response(b"ERR,overheat\n")
        assert result["status"] == "error"
        assert "overheat" in result["message"]

    def test_encode_ping(self):
        assert encode_ping() == b"PING\n"

    def test_encode_stop(self):
        assert encode_stop() == b"M,0,0\n"


class TestMockDriver:
    def test_connect_mock(self, mock_driver):
        assert mock_driver.connected is True

    def test_send_motor_returns_true(self, mock_driver):
        ok = mock_driver.send_motor(MotorCommand(100, 100))
        assert ok is True

    def test_cmd_count_increments(self, mock_driver):
        mock_driver.send_motor(MotorCommand(10, 20))
        mock_driver.send_motor(MotorCommand(30, 40))
        assert mock_driver.stats["cmd_count"] >= 2

    def test_last_cmd_tracked(self, mock_driver):
        cmd = MotorCommand(77, 88)
        mock_driver.send_motor(cmd)
        assert mock_driver.stats["last_cmd"] == {"left": 77, "right": 88}

    def test_context_manager(self):
        with ESP32SerialDriver(SerialConfig(mock=True)) as d:
            ok = d.send_motor(MotorCommand(50, 50))
        assert ok is True
        assert d.connected is False

    @pytest.mark.asyncio
    async def test_send_motor_async(self, mock_driver):
        ok = await mock_driver.send_motor_async(MotorCommand(60, 60))
        assert ok is True
