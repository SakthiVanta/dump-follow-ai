"""
ActiveTrainer — triggers retraining of FeatureNN when enough new labels exist.

Keeps track of the last training checkpoint (highest confirmed frame id seen)
and retrains whenever `retrain_threshold` new confirmed labels have appeared.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from robot.learning.label_store import LabelStore
from robot.learning.tiny_nn import FeatureNN, ACTIONS
from robot.logger import logger


class ActiveTrainer:
    """
    Lightweight online retrainer.

    Parameters
    ----------
    store : LabelStore
        Source of confirmed training samples.
    model_path : str | Path
        Where to save the retrained model.
    retrain_threshold : int
        Retrain once this many *new* confirmed labels accumulate.
    epochs : int
        Training epochs per retraining run.
    """

    def __init__(
        self,
        store: LabelStore,
        model_path: str | Path = "models/tiny_nn.pkl",
        retrain_threshold: int = 20,
        epochs: int = 300,
    ) -> None:
        self.store = store
        self.model_path = Path(model_path)
        self.retrain_threshold = retrain_threshold
        self.epochs = epochs

        self._last_trained_id: int = 0
        self._last_accuracy: float = 0.0
        self._retrain_count: int = 0
        self.model: Optional[FeatureNN] = self._load_or_init()

    # ── Public API ───────────────────────────────────────────────────────────

    def check_and_retrain(self) -> Optional[FeatureNN]:
        """
        Call periodically. Retrains if enough new labels have appeared.
        Returns the new model or None if no retraining happened.
        """
        new_count = self.store.get_new_confirmed_count(since_id=self._last_trained_id)
        if new_count < self.retrain_threshold:
            return None

        logger.info(
            f"ActiveTrainer: {new_count} new labels ≥ threshold {self.retrain_threshold} "
            "— retraining FeatureNN…"
        )
        return self._retrain()

    def force_retrain(self) -> FeatureNN:
        """Retrain immediately regardless of threshold."""
        return self._retrain()

    @property
    def last_accuracy(self) -> float:
        return self._last_accuracy

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    def stats(self) -> dict:
        new_count = self.store.get_new_confirmed_count(since_id=self._last_trained_id)
        return {
            "retrain_count": self._retrain_count,
            "last_accuracy": round(self._last_accuracy, 4),
            "new_labels_since_last_train": new_count,
            "retrain_threshold": self.retrain_threshold,
            "model_path": str(self.model_path),
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _retrain(self) -> FeatureNN:
        X, y = self.store.get_training_data()

        if len(X) < 4:
            logger.warning(
                f"ActiveTrainer: only {len(X)} confirmed samples — need ≥4 to train. "
                "Keeping existing model."
            )
            return self.model  # type: ignore[return-value]

        # Ensure all action classes have at least 1 sample (pad with dummy)
        present = set(y.tolist())
        missing = set(range(len(ACTIONS))) - present
        if missing:
            pad_x = np.zeros((len(missing), X.shape[1]), dtype=np.float64)
            pad_y = np.array(list(missing), dtype=np.int64)
            X = np.vstack([X, pad_x])
            y = np.concatenate([y, pad_y])

        model = FeatureNN()
        losses = model.fit(X, y, epochs=self.epochs)
        acc = model.accuracy(X, y)

        model.save(self.model_path)
        self._last_accuracy = acc
        self._retrain_count += 1

        # Advance checkpoint to current max id
        db_stats = self.store.get_stats()
        # Use total labeled count as proxy — exact id tracking needs a query
        # We'll simply re-query after save
        self._last_trained_id = self._get_max_confirmed_id()

        self.model = model
        logger.info(
            f"ActiveTrainer: retrain #{self._retrain_count} done — "
            f"accuracy={acc:.2%} samples={len(X)} "
            f"final_loss={losses[-1]:.4f}"
        )
        return model

    def _get_max_confirmed_id(self) -> int:
        rows = self.store._con.execute(
            "SELECT MAX(id) FROM frames WHERE is_reviewed=1 AND confirmed_action IS NOT NULL"
        ).fetchone()
        val = rows[0]
        return int(val) if val is not None else 0

    def _load_or_init(self) -> FeatureNN:
        if self.model_path.exists():
            try:
                model = FeatureNN.load(self.model_path)
                logger.info(f"ActiveTrainer: loaded existing model from {self.model_path}")
                return model
            except Exception as exc:
                logger.warning(f"ActiveTrainer: could not load model ({exc}), initialising fresh")
        return FeatureNN()
