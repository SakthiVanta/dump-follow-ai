"""
Tiny Neural Network — pure numpy, from scratch.

Architecture (FeatureNN):
    Input(5) → Dense(16) → ReLU → Dense(8) → ReLU → Dense(4) → Softmax

Actions: LEFT=0  RIGHT=1  FORWARD=2  STOP=3
Features: [x_center, y_center, area, dx, dy]  (all normalised 0–1)

TinyCNN is also provided for motion-patch input (32×32 grayscale).
Requires PyTorch — only imported if available.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# ── Action vocabulary ───────────────────────────────────────────────────────
ACTIONS = ["LEFT", "RIGHT", "FORWARD", "STOP"]
ACTION_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)


# ── Math primitives ─────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float64)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    if x.ndim == 1:
        x = x[np.newaxis, :]
        s = x - x.max(axis=1, keepdims=True)
        e = np.exp(s)
        return (e / e.sum(axis=1, keepdims=True))[0]
    s = x - x.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def _cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    n = y.shape[0]
    return -float(np.log(probs[np.arange(n), y] + 1e-10).mean())


# ── FeatureNN ───────────────────────────────────────────────────────────────

class FeatureNN:
    """
    3-layer fully-connected network trained on 5 normalised bbox features.

    Usage
    -----
    >>> model = FeatureNN()
    >>> action, conf = model.predict_action([0.5, 0.4, 0.12, -0.03, 0.0])
    >>> losses = model.fit(X_train, y_train, epochs=200)
    >>> model.save("models/tiny_nn.pkl")
    >>> model = FeatureNN.load("models/tiny_nn.pkl")
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden1: int = 16,
        hidden2: int = 8,
        output_dim: int = N_ACTIONS,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        rng = np.random.default_rng(seed)

        # Xavier initialisation
        self.W1 = rng.standard_normal((input_dim, hidden1)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.standard_normal((hidden1, hidden2)) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.standard_normal((hidden2, output_dim)) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_dim)

        # Adam state
        self._t: int = 0
        self._b1, self._b2_adam = 0.9, 0.999
        self._eps = 1e-8
        keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self._m: dict[str, np.ndarray] = {k: np.zeros_like(getattr(self, k)) for k in keys}
        self._v: dict[str, np.ndarray] = {k: np.zeros_like(getattr(self, k)) for k in keys}

        # Cache for backprop
        self._cache: dict = {}

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return probability distribution (N, 4)."""
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]

        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        probs = _softmax(z3)

        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return probs[0] if single else probs

    # ── Inference ───────────────────────────────────────────────────────────

    def predict_action(self, features: list[float] | np.ndarray) -> tuple[str, float]:
        """Single sample → (action_name, confidence)."""
        probs = self.forward(np.asarray(features, dtype=np.float64))
        idx = int(np.argmax(probs))
        return ACTIONS[idx], float(probs[idx])

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch → (action_indices, confidences)."""
        probs = self.forward(X)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]
        return np.argmax(probs, axis=1), np.max(probs, axis=1)

    # ── Training ────────────────────────────────────────────────────────────

    def _backprop(self, X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        probs = self.forward(X)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]
        n = X.shape[0] if X.ndim > 1 else 1
        c = self._cache

        d3 = probs.copy()
        d3[np.arange(n), y] -= 1
        d3 /= n

        dW3 = c["a2"].T @ d3
        db3 = d3.sum(axis=0)

        d2 = (d3 @ self.W3.T) * _relu_grad(c["z2"])
        dW2 = c["a1"].T @ d2
        db2 = d2.sum(axis=0)

        d1 = (d2 @ self.W2.T) * _relu_grad(c["z1"])
        dW1 = c["X"].T @ d1
        db1 = d1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def _adam_step(self, grads: dict[str, np.ndarray]) -> None:
        self._t += 1
        for key, grad in grads.items():
            self._m[key] = self._b1 * self._m[key] + (1 - self._b1) * grad
            self._v[key] = self._b2_adam * self._v[key] + (1 - self._b2_adam) * grad ** 2
            m_hat = self._m[key] / (1 - self._b1 ** self._t)
            v_hat = self._v[key] / (1 - self._b2_adam ** self._t)
            param = getattr(self, key)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """One Adam step. Returns cross-entropy loss."""
        probs = self.forward(X)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]
        loss = _cross_entropy(probs, y)
        grads = self._backprop(X, y)
        self._adam_step(grads)
        return loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        batch_size: int = 32,
    ) -> list[float]:
        """Full training. Returns per-epoch loss list."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = X.shape[0]
        losses: list[float] = []

        for _ in range(epochs):
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]
            epoch_loss = 0.0
            steps = 0
            for i in range(0, n, batch_size):
                xb = Xs[i : i + batch_size]
                yb = ys[i : i + batch_size]
                epoch_loss += self.train_step(xb, yb)
                steps += 1
            losses.append(epoch_loss / max(steps, 1))

        return losses

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds, _ = self.predict_batch(np.asarray(X, dtype=np.float64))
        return float((preds == np.asarray(y, dtype=np.int64)).mean())

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "lr": self.lr,
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
            "adam_t": self._t,
            "adam_m": self._m,
            "adam_v": self._v,
        }
        with out.open("wb") as f:
            pickle.dump(payload, f)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNN":
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        model = cls(input_dim=d["input_dim"], output_dim=d["output_dim"], lr=d.get("lr", 1e-3))
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(model, key, np.asarray(d[key], dtype=np.float64))
        model._t = d.get("adam_t", 0)
        model._m = d.get("adam_m", {k: np.zeros_like(getattr(model, k)) for k in ["W1","b1","W2","b2","W3","b3"]})
        model._v = d.get("adam_v", {k: np.zeros_like(getattr(model, k)) for k in ["W1","b1","W2","b2","W3","b3"]})
        return model

    def param_count(self) -> int:
        return sum(getattr(self, k).size for k in ("W1", "b1", "W2", "b2", "W3", "b3"))

    def __repr__(self) -> str:
        return (
            f"FeatureNN(5→16→8→{self.output_dim}) "
            f"params={self.param_count()} "
            f"actions={ACTIONS}"
        )


# ── TinyCNN (optional, PyTorch) ─────────────────────────────────────────────

def build_tiny_cnn():  # type: ignore[return]
    """
    Returns a PyTorch TinyCNN model if torch is available, else None.

    Architecture:
        Conv(1→8, 3×3) → ReLU → MaxPool(2×2)
        Conv(8→16, 3×3) → ReLU → MaxPool(2×2)
        Flatten → Dense(576→64) → ReLU → Dense(64→4)

    Input: (N, 1, 32, 32) grayscale motion patch
    Output: (N, 4) action logits
    """
    try:
        import torch
        import torch.nn as nn

        class TinyCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=3, padding=0),   # 30×30
                    nn.ReLU(),
                    nn.MaxPool2d(2),                              # 15×15
                    nn.Conv2d(8, 16, kernel_size=3, padding=0),  # 13×13
                    nn.ReLU(),
                    nn.MaxPool2d(2),                              # 6×6
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(16 * 6 * 6, 64),
                    nn.ReLU(),
                    nn.Linear(64, N_ACTIONS),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

            def predict_action(self, patch: np.ndarray) -> tuple[str, float]:
                """patch: (32,32) uint8 grayscale → (action, confidence)"""
                import torch.nn.functional as F
                self.eval()
                with torch.no_grad():
                    t = torch.tensor(
                        patch.astype(np.float32) / 255.0,
                        dtype=torch.float32,
                    ).unsqueeze(0).unsqueeze(0)  # (1,1,32,32)
                    logits = self.forward(t)
                    probs = F.softmax(logits, dim=1).squeeze().numpy()
                return ACTIONS[int(probs.argmax())], float(probs.max())

        return TinyCNN()

    except ImportError:
        return None
