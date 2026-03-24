"""Unit tests for the SVSP motion model."""
from robot.config import MotionModelConfig
from robot.vision.detector import Detection
from robot.vision.svsp import (
    SVSPMotionPredictor,
    generate_svsp_training_data,
    load_svsp_model,
    train_svsp_model,
)


def _det(x: int, y: int, w: int = 80, h: int = 120) -> Detection:
    return Detection(label="person", confidence=0.9, x=x, y=y, w=w, h=h, track_id=1)


class TestSVSPTraining:
    def test_generate_training_data_shapes(self):
        X, y = generate_svsp_training_data(sample_count=50, sequence_length=6, seed=7)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == (6 - 1) * 5

    def test_train_and_load_model(self, tmp_path):
        cfg = MotionModelConfig(
            model_path=str(tmp_path / "svsp.pt"),
            train_samples=100,
            sequence_length=6,
            epochs=40,
            learning_rate=0.2,
        )
        result = train_svsp_model(cfg)
        model = load_svsp_model(result.model_path)
        assert result.model_path.exists()
        assert model.sequence_length == 6

    def test_runtime_predictor_produces_motion_string(self, tmp_path):
        cfg = MotionModelConfig(
            model_path=str(tmp_path / "svsp.pt"),
            train_samples=100,
            sequence_length=5,
            epochs=30,
        )
        result = train_svsp_model(cfg)
        model = load_svsp_model(result.model_path)
        predictor = SVSPMotionPredictor(model, frame_width=320, frame_height=240)

        motion = None
        for det in [
            _det(100, 60, 60, 90),
            _det(120, 60, 60, 90),
            _det(140, 60, 60, 90),
            _det(160, 60, 60, 90),
            _det(180, 60, 60, 90),
        ]:
            motion = predictor.update(det)

        assert motion is not None
        assert isinstance(motion.summary, str)
