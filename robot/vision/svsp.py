"""SVSP motion model: predicts movement direction from bbox history."""
from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from robot.config import MotionModelConfig
from robot.vision.detector import Detection
from robot.vision.motion import MotionEstimate


SVSP_LABELS = ["left", "right", "forward", "backward", "stationary"]
_LABEL_TO_INDEX = {label: index for index, label in enumerate(SVSP_LABELS)}


@dataclass
class SVSPTrainingResult:
    model_path: Path
    accuracy: float
    samples: int
    sequence_length: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model_path": str(self.model_path),
            "accuracy": round(self.accuracy, 4),
            "samples": self.samples,
            "sequence_length": self.sequence_length,
        }


class SVSPClassifier:
    """Small softmax classifier over flattened bbox-sequence features."""

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.feature_dim = max(sequence_length - 1, 1) * 5
        self.weights = np.zeros((self.feature_dim, len(SVSP_LABELS)), dtype=np.float64)
        self.bias = np.zeros(len(SVSP_LABELS), dtype=np.float64)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 120,
        learning_rate: float = 0.15,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        y_one_hot = np.eye(len(SVSP_LABELS))[y]

        for _ in range(max(1, epochs)):
            logits = X @ self.weights + self.bias
            probs = _softmax(logits)
            error = probs - y_one_hot
            grad_w = (X.T @ error) / len(X)
            grad_b = error.mean(axis=0)
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.weights + self.bias
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float((pred == y).mean())

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as handle:
            pickle.dump(
                {
                    "sequence_length": self.sequence_length,
                    "weights": self.weights,
                    "bias": self.bias,
                    "labels": SVSP_LABELS,
                },
                handle,
            )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "SVSPClassifier":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        model = cls(sequence_length=int(payload["sequence_length"]))
        model.weights = np.asarray(payload["weights"], dtype=np.float64)
        model.bias = np.asarray(payload["bias"], dtype=np.float64)
        return model


class SVSPMotionPredictor:
    """Runtime wrapper that predicts motion direction from selected target history."""

    def __init__(
        self,
        model: SVSPClassifier,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self._model = model
        self._frame_width = max(frame_width, 1)
        self._frame_height = max(frame_height, 1)
        self._history: deque[np.ndarray] = deque(maxlen=model.sequence_length)
        self._current_track: Optional[int] = None
        self._current_label: Optional[str] = None

    def reset(self) -> None:
        self._history.clear()
        self._current_track = None
        self._current_label = None

    def update(self, target: Optional[Detection]) -> MotionEstimate:
        if target is None:
            self.reset()
            return MotionEstimate()

        if self._should_reset(target):
            self.reset()

        self._current_track = target.track_id
        self._current_label = target.label
        self._history.append(_bbox_vector(target, self._frame_width, self._frame_height))

        if len(self._history) < self._model.sequence_length:
            return MotionEstimate()

        X = np.asarray([_sequence_to_features(list(self._history))], dtype=np.float64)
        probs = self._model.predict_proba(X)[0]
        label = SVSP_LABELS[int(np.argmax(probs))]
        confidence = float(np.max(probs))
        horizontal = label if label in {"left", "right"} else "stationary"
        depth = label if label in {"forward", "backward"} else "stationary"
        summary = label if label != "stationary" else "stationary"
        return MotionEstimate(
            horizontal=horizontal,
            depth=depth,
            summary=f"{summary} ({confidence:.2f})",
        )

    def _should_reset(self, target: Detection) -> bool:
        if self._current_label is None:
            return False
        if target.label != self._current_label:
            return True
        if self._current_track is None or target.track_id is None:
            return False
        return target.track_id != self._current_track


def train_svsp_model(config: MotionModelConfig) -> SVSPTrainingResult:
    X, y = generate_svsp_training_data(
        sample_count=config.train_samples,
        sequence_length=config.sequence_length,
        noise=config.synthetic_noise,
    )
    model = SVSPClassifier(sequence_length=config.sequence_length)
    model.fit(X, y, epochs=config.epochs, learning_rate=config.learning_rate)
    accuracy = model.evaluate(X, y)
    path = model.save(config.model_path)
    return SVSPTrainingResult(
        model_path=path,
        accuracy=accuracy,
        samples=len(X),
        sequence_length=config.sequence_length,
    )


def generate_svsp_training_data(
    sample_count: int = 2000,
    sequence_length: int = 8,
    noise: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = list(range(len(SVSP_LABELS)))
    per_label = max(sample_count // len(labels), 1)
    features: list[np.ndarray] = []
    target: list[int] = []

    for label_idx in labels:
        for _ in range(per_label):
            seq = _generate_sequence(label_idx, sequence_length, noise, rng)
            features.append(_sequence_to_features(seq))
            target.append(label_idx)

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.int64)
    return X, y


def load_svsp_model(path: str | Path) -> SVSPClassifier:
    return SVSPClassifier.load(path)


def _generate_sequence(
    label_idx: int,
    sequence_length: int,
    noise: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    cx = rng.uniform(0.35, 0.65)
    cy = rng.uniform(0.35, 0.65)
    w = rng.uniform(0.12, 0.3)
    h = rng.uniform(0.22, 0.45)

    seq: list[np.ndarray] = []
    for _ in range(sequence_length):
        seq.append(np.asarray([cx, cy, w, h, w * h], dtype=np.float64))
        dx, dy, dw, dh = _label_step(label_idx, rng)
        cx = np.clip(cx + dx + rng.normal(0.0, noise / 2), 0.05, 0.95)
        cy = np.clip(cy + dy + rng.normal(0.0, noise / 3), 0.05, 0.95)
        w = np.clip(w + dw + rng.normal(0.0, noise / 4), 0.05, 0.65)
        h = np.clip(h + dh + rng.normal(0.0, noise / 4), 0.08, 0.85)

    return seq


def _label_step(label_idx: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    horizontal_step = rng.uniform(0.02, 0.05)
    depth_step = rng.uniform(0.015, 0.04)

    if SVSP_LABELS[label_idx] == "left":
        return -horizontal_step, 0.0, 0.0, 0.0
    if SVSP_LABELS[label_idx] == "right":
        return horizontal_step, 0.0, 0.0, 0.0
    if SVSP_LABELS[label_idx] == "forward":
        return 0.0, 0.0, depth_step, depth_step * 1.4
    if SVSP_LABELS[label_idx] == "backward":
        return 0.0, 0.0, -depth_step, -depth_step * 1.4
    return (
        rng.uniform(-0.004, 0.004),
        rng.uniform(-0.003, 0.003),
        rng.uniform(-0.003, 0.003),
        rng.uniform(-0.003, 0.003),
    )


def _bbox_vector(target: Detection, frame_width: int, frame_height: int) -> np.ndarray:
    w = target.w / max(frame_width, 1)
    h = target.h / max(frame_height, 1)
    return np.asarray(
        [
            target.cx / max(frame_width, 1),
            target.cy / max(frame_height, 1),
            w,
            h,
            w * h,
        ],
        dtype=np.float64,
    )


def _sequence_to_features(sequence: list[np.ndarray]) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float64)
    deltas = seq[1:] - seq[:-1]
    return deltas.reshape(-1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
