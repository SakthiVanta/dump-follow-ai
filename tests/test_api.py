"""Integration tests for FastAPI endpoints."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

from robot.api.app import create_app
from robot.control.follower import RobotMode
from robot.control.motor import MotorCommand


@pytest.fixture
def app():
    """Create test app in mock mode."""
    return create_app(mock=True)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
async def async_client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

class TestStatusEndpoints:
    def test_health(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_status_shape(self, client):
        resp = client.get("/api/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "mode" in data
        assert "serial" in data
        assert "motion" in data


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

class TestControlEndpoints:
    def test_set_mode_follow(self, client):
        resp = client.post("/api/v1/mode", json={"mode": "follow_person"})
        assert resp.status_code == 200
        assert resp.json()["mode"] == "follow_person"

    def test_set_mode_stop(self, client):
        resp = client.post("/api/v1/mode", json={"mode": "stop"})
        assert resp.status_code == 200

    def test_stop_endpoint(self, client):
        resp = client.post("/api/v1/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_manual_drive_requires_manual_mode(self, client):
        # Default mode is not manual
        resp = client.post("/api/v1/drive", json={"left": 100, "right": 100})
        assert resp.status_code == 409

    def test_manual_drive_in_manual_mode(self, client):
        client.post("/api/v1/mode", json={"mode": "manual"})
        resp = client.post("/api/v1/drive", json={"left": 100, "right": 80})
        assert resp.status_code == 200
        assert resp.json()["command"]["left"] == 100

    def test_drive_clamps_out_of_range(self, client):
        client.post("/api/v1/mode", json={"mode": "manual"})
        resp = client.post("/api/v1/drive", json={"left": 999, "right": 80})
        assert resp.status_code == 422  # Pydantic validation

    def test_set_speed(self, client):
        resp = client.post("/api/v1/speed", json={"speed": 150})
        assert resp.status_code == 200

    def test_voice_command_follow(self, client):
        resp = client.post("/api/v1/voice", json={"text": "follow me"})
        assert resp.status_code == 200
        assert resp.json()["intent"]["type"] == "follow_person"

    def test_voice_command_stop(self, client):
        resp = client.post("/api/v1/voice", json={"text": "stop"})
        assert resp.status_code == 200

    def test_invalid_mode(self, client):
        resp = client.post("/api/v1/mode", json={"mode": "fly_away"})
        assert resp.status_code == 422

    def test_train_svsp_endpoint(self, client):
        resp = client.post("/api/v1/train/svsp")
        assert resp.status_code == 200
        assert resp.json()["status"] == "trained"

    def test_disable_svsp_endpoint(self, client):
        resp = client.post("/api/v1/model/svsp/disable")
        assert resp.status_code == 200
        assert resp.json()["motion_source"] == "heuristic"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_get_config_defaults(self):
        from robot.config import get_config, RobotConfig
        # Clear lru_cache to get fresh config
        get_config.cache_clear()
        cfg = get_config()
        assert isinstance(cfg, RobotConfig)
        assert cfg.control.base_speed > 0
        assert cfg.vision.frame_width > 0

    def test_pid_config_valid(self):
        from robot.config import PIDConfig
        pid = PIDConfig()
        assert pid.kp > 0

    def test_serial_mock_default(self):
        from robot.config import SerialConfig
        cfg = SerialConfig()
        assert isinstance(cfg.mock, bool)
