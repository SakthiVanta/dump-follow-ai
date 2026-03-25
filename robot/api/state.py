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
from robot.learning.label_store import LabelStore
from robot.learning.collector import FrameCollector
from robot.learning.active_trainer import ActiveTrainer
from robot.learning.tiny_nn import FeatureNN


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

        # Active learning
        al = config.active_learning
        self.label_store = LabelStore(db_path=al.db_path)
        self.collector = FrameCollector(
            store=self.label_store,
            save_dir=al.frames_dir,
            confidence_threshold=al.confidence_threshold,
            sample_rate=al.sample_rate,
            patch_size=al.patch_size,
            cooldown_s=al.cooldown_s,
        )
        self.active_trainer = ActiveTrainer(
            store=self.label_store,
            model_path=al.model_path,
            retrain_threshold=al.retrain_threshold,
            epochs=al.retrain_epochs,
        )
        self.tiny_nn: FeatureNN = self.active_trainer.model

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

    def collect_frame(self, frame, target) -> None:
        """
        Save frame + features into the active-learning review queue if needed.

        Call once per processed frame from the robot loop after update_motion().
        Confidence is read from svsp_predictor.last_confidence when available,
        otherwise from tiny_nn inference.
        """
        if not self.config.active_learning.enabled or target is None:
            return

        # Determine action + confidence from whichever predictor is active
        if self.svsp_predictor is not None and self.svsp_predictor.last_confidence > 0:
            action_str = self.svsp_predictor.last_label.upper()
            confidence = self.svsp_predictor.last_confidence
            source = "svsp"
        else:
            # Use TinyNN on current features if we have them
            features = self.collector._extract_features(target, frame.shape)
            action_str, confidence = self.tiny_nn.predict_action(features)
            source = "tiny_nn"

        # Map SVSP labels (left/right/forward/backward/stationary) → NN actions
        _svsp_to_action = {
            "left": "LEFT", "right": "RIGHT",
            "forward": "FORWARD", "backward": "FORWARD",
            "stationary": "STOP",
        }
        action_str = _svsp_to_action.get(action_str.lower(), action_str.upper())
        if action_str not in ("LEFT", "RIGHT", "FORWARD", "STOP"):
            action_str = "STOP"

        self.collector.maybe_collect(
            frame=frame,
            target=target,
            predicted_action=action_str,
            confidence=confidence,
            source=source,
        )

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
