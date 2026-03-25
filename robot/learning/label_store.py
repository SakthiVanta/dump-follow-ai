"""
SQLite-backed store for collected frames and human labels.

Schema
------
frames
  id              INTEGER PK
  timestamp       TEXT     ISO-8601
  image_path      TEXT     path to saved JPEG
  features_json   TEXT     JSON [x_c, y_c, area, dx, dy]
  predicted_action TEXT    LEFT/RIGHT/FORWARD/STOP
  confidence      REAL     0–1
  confirmed_action TEXT    NULL until reviewed
  is_reviewed     INTEGER  0/1
  needs_review    INTEGER  0/1  (1 = low-confidence → show in UI)
  source          TEXT     "svsp" | "tiny_nn" | "heuristic"
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from robot.learning.tiny_nn import ACTIONS, ACTION_TO_IDX


_DDL = """
CREATE TABLE IF NOT EXISTS frames (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    image_path       TEXT    NOT NULL,
    features_json    TEXT,
    predicted_action TEXT,
    confidence       REAL    DEFAULT 0.0,
    confirmed_action TEXT,
    is_reviewed      INTEGER DEFAULT 0,
    needs_review     INTEGER DEFAULT 1,
    source           TEXT    DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_needs_review ON frames (needs_review, is_reviewed);
CREATE INDEX IF NOT EXISTS idx_timestamp    ON frames (timestamp);
"""


class LabelStore:
    """Thread-safe SQLite label store."""

    def __init__(self, db_path: str | Path = "data/labels.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(_DDL)
        self._con.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def add_frame(
        self,
        image_path: str | Path,
        features: list[float] | None,
        predicted_action: str,
        confidence: float,
        source: str = "unknown",
        needs_review: bool = True,
    ) -> int:
        """Insert a new frame record. Returns the new row id."""
        ts = datetime.now(timezone.utc).isoformat()
        feats_json = json.dumps([round(f, 6) for f in features]) if features else None
        cur = self._con.execute(
            """INSERT INTO frames
               (timestamp, image_path, features_json, predicted_action,
                confidence, needs_review, source)
               VALUES (?,?,?,?,?,?,?)""",
            (ts, str(image_path), feats_json, predicted_action,
             float(confidence), int(needs_review), source),
        )
        self._con.commit()
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def confirm_label(self, frame_id: int, confirmed_action: str) -> bool:
        """Mark a frame as reviewed with the human-confirmed action."""
        if confirmed_action not in ACTIONS:
            return False
        self._con.execute(
            """UPDATE frames
               SET confirmed_action=?, is_reviewed=1, needs_review=0
               WHERE id=?""",
            (confirmed_action, frame_id),
        )
        self._con.commit()
        return True

    def skip_frame(self, frame_id: int) -> None:
        """Mark frame as reviewed without adding a label (not useful for training)."""
        self._con.execute(
            "UPDATE frames SET is_reviewed=1, needs_review=0 WHERE id=?",
            (frame_id,),
        )
        self._con.commit()

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_pending_review(self, limit: int = 100) -> list[dict]:
        """Frames waiting for human review, newest first."""
        rows = self._con.execute(
            """SELECT * FROM frames
               WHERE needs_review=1 AND is_reviewed=0
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_frame(self, frame_id: int) -> Optional[dict]:
        row = self._con.execute(
            "SELECT * FROM frames WHERE id=?", (frame_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (X, y) arrays for all confirmed+reviewed frames that have features.

        X shape: (N, 5)
        y shape: (N,) — integer action indices
        """
        rows = self._con.execute(
            """SELECT features_json, confirmed_action FROM frames
               WHERE is_reviewed=1
                 AND confirmed_action IS NOT NULL
                 AND features_json IS NOT NULL"""
        ).fetchall()

        X, y = [], []
        for row in rows:
            feats = json.loads(row["features_json"])
            action_idx = ACTION_TO_IDX.get(row["confirmed_action"])
            if action_idx is not None and len(feats) == 5:
                X.append(feats)
                y.append(action_idx)

        if not X:
            return np.empty((0, 5), dtype=np.float64), np.empty(0, dtype=np.int64)

        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)

    def get_new_confirmed_count(self, since_id: int = 0) -> int:
        """How many confirmed frames exist with id > since_id."""
        row = self._con.execute(
            "SELECT COUNT(*) FROM frames WHERE is_reviewed=1 AND confirmed_action IS NOT NULL AND id > ?",
            (since_id,),
        ).fetchone()
        return int(row[0])

    def get_stats(self) -> dict:
        rows = self._con.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN needs_review=1 AND is_reviewed=0 THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN is_reviewed=1 THEN 1 ELSE 0 END) as reviewed,
                SUM(CASE WHEN confirmed_action IS NOT NULL THEN 1 ELSE 0 END) as labeled,
                AVG(confidence) as avg_confidence
               FROM frames"""
        ).fetchone()
        return dict(rows)

    def get_all_frames(self, limit: int = 500, offset: int = 0) -> list[dict]:
        rows = self._con.execute(
            "SELECT * FROM frames ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._con.close()
