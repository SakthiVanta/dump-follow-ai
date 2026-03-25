"""Centralised configuration loader using YAML + env overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Pydantic models — mirrors config/config.yaml
# ---------------------------------------------------------------------------

class PIDConfig(BaseModel):
    kp: float = 0.4
    ki: float = 0.01
    kd: float = 0.1
    integral_limit: float = 500.0
    output_limit: float = 150.0


class ControlConfig(BaseModel):
    base_speed: int = 100
    max_speed: int = 200
    min_speed: int = -200
    pid: PIDConfig = Field(default_factory=PIDConfig)
    safe_distance_ratio: float = 0.35
    search_turn_speed: int = 50
    lost_target_timeout_s: float = 2.0


class VisionConfig(BaseModel):
    model_path: str = "models/yolov8n.pt"
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    target_class: str = "person"
    frame_width: int = 320
    frame_height: int = 240
    camera_id: int = 0
    fps_target: int = 30
    tracker: str = "bytetrack"


class MotorConfig(BaseModel):
    invert_left: bool = False
    invert_right: bool = False
    pwm_frequency: int = 1000


class SerialConfig(BaseModel):
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout_s: float = 1.0
    mock: bool = True


class VoiceConfig(BaseModel):
    enabled: bool = True
    engine: str = "google"
    whisper_model: str = "base"
    language: str = "en-US"
    energy_threshold: int = 300
    phrase_time_limit_s: int = 5


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class TrainingConfig(BaseModel):
    data_dir: str = "data"
    model_base: str = "yolov8n.pt"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    output_dir: str = "models/trained"
    augment: bool = True


class MotionModelConfig(BaseModel):
    enabled: bool = False
    model_path: str = "models/svsp.pt"
    sequence_length: int = 8
    train_samples: int = 2000
    epochs: int = 120
    learning_rate: float = 0.15
    synthetic_noise: float = 0.03


class RobotMeta(BaseModel):
    name: str = "TinyAIRobot"
    version: str = "1.0.0"
    log_level: str = "INFO"


class ActiveLearningConfig(BaseModel):
    enabled: bool = True
    db_path: str = "data/labels.db"
    frames_dir: str = "data/review_frames"
    model_path: str = "models/tiny_nn.pkl"
    confidence_threshold: float = 0.6   # below this → goes to review queue
    sample_rate: float = 0.05           # fraction of high-conf frames saved
    retrain_threshold: int = 20         # new confirmed labels before retraining
    retrain_epochs: int = 300
    cooldown_s: float = 0.5             # min seconds between frame saves
    patch_size: int = 64                # saved image size in pixels


class RobotConfig(BaseModel):
    robot: RobotMeta = Field(default_factory=RobotMeta)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    motor: MotorConfig = Field(default_factory=MotorConfig)
    serial: SerialConfig = Field(default_factory=SerialConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    motion_model: MotionModelConfig = Field(default_factory=MotionModelConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _yaml_to_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> RobotConfig:
    """Load config from YAML, apply env overrides, return validated model."""
    default_path = Path(__file__).parent.parent / "config" / "config.yaml"
    path = Path(config_path) if config_path else default_path

    data = _yaml_to_dict(path)

    # Env overrides
    env_map = {
        "SERIAL_PORT": ("serial", "port"),
        "CAMERA_ID": ("vision", "camera_id"),
        "MOCK_SERIAL": ("serial", "mock"),
        "API_HOST": ("api", "host"),
        "API_PORT": ("api", "port"),
        "LOG_LEVEL": ("robot", "log_level"),
    }
    for env_key, (section, field) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if section not in data:
                data[section] = {}
            # Cast bools
            if val.lower() in ("true", "1"):
                val = True
            elif val.lower() in ("false", "0"):
                val = False
            else:
                try:
                    val = int(val)
                except ValueError:
                    pass
            data[section][field] = val

    return RobotConfig(**data)
