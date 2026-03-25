"""
FrameCollector — saves motion patches + features for active learning.

Saves a frame when:
  • Predicted confidence < confidence_threshold  (always)
  • OR random coin-flip at sample_rate            (diversity sampling)

Each saved frame gets:
  • A JPEG crop of the detected bbox saved to disk
  • A row in the LabelStore SQLite database
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from robot.learning.label_store import LabelStore
from robot.vision.detector import Detection
from robot.logger import logger


class FrameCollector:
    """
    Hooks into the vision pipeline to collect training samples.

    Parameters
    ----------
    store : LabelStore
        Database to write frame records into.
    save_dir : str | Path
        Directory where JPEG images are saved.
    confidence_threshold : float
        Frames with predicted confidence below this go to review queue.
    sample_rate : float
        Fraction of high-confidence frames also saved (diversity sampling).
    patch_size : int
        Saved image is resized to patch_size × patch_size.
    """

    def __init__(
        self,
        store: LabelStore,
        save_dir: str | Path = "data/review_frames",
        confidence_threshold: float = 0.6,
        sample_rate: float = 0.05,
        patch_size: int = 64,
        cooldown_s: float = 0.5,
    ) -> None:
        self.store = store
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.cooldown_s = cooldown_s

        self._prev_target: Optional[Detection] = None
        self._last_save_time: float = 0.0
        self._total_collected: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def maybe_collect(
        self,
        frame: np.ndarray,
        target: Optional[Detection],
        predicted_action: str,
        confidence: float,
        source: str = "unknown",
    ) -> bool:
        """
        Decide whether to save this frame. Returns True if saved.

        Call once per processed frame from the main robot loop.
        """
        if target is None:
            self._prev_target = None
            return False

        now = time.monotonic()
        if now - self._last_save_time < self.cooldown_s:
            self._prev_target = target
            return False

        low_conf = confidence < self.confidence_threshold
        sampled = random.random() < self.sample_rate

        if not (low_conf or sampled):
            self._prev_target = target
            return False

        features = self._extract_features(target, frame.shape)
        image_path = self._save_patch(frame, target)

        self.store.add_frame(
            image_path=image_path,
            features=features,
            predicted_action=predicted_action,
            confidence=confidence,
            source=source,
            needs_review=low_conf,  # only low-conf shown by default
        )

        self._last_save_time = now
        self._prev_target = target
        self._total_collected += 1
        logger.debug(
            f"Collected frame #{self._total_collected}: "
            f"action={predicted_action} conf={confidence:.2f} "
            f"{'[LOW CONF→review]' if low_conf else '[sampled]'}"
        )
        return True

    @property
    def total_collected(self) -> int:
        return self._total_collected

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_features(
        self, target: Detection, frame_shape: tuple
    ) -> list[float]:
        """
        5 normalised features: [x_center, y_center, area, dx, dy]

        dx/dy = change in centre from previous frame (0 if no history).
        """
        h, w = frame_shape[:2]
        x_c = target.cx / max(w, 1)
        y_c = target.cy / max(h, 1)
        area = (target.w * target.h) / max(w * h, 1)

        dx, dy = 0.0, 0.0
        if self._prev_target is not None:
            dx = (target.cx - self._prev_target.cx) / max(w, 1)
            dy = (target.cy - self._prev_target.cy) / max(h, 1)

        return [
            float(np.clip(x_c,  0.0, 1.0)),
            float(np.clip(y_c,  0.0, 1.0)),
            float(np.clip(area, 0.0, 1.0)),
            float(np.clip(dx,  -1.0, 1.0)),
            float(np.clip(dy,  -1.0, 1.0)),
        ]

    def _save_patch(self, frame: np.ndarray, target: Detection) -> Path:
        """Crop + resize the detection bbox and save as JPEG. Returns path."""
        # Clamp bbox to frame boundaries
        fh, fw = frame.shape[:2]
        x1 = max(0, target.x)
        y1 = max(0, target.y)
        x2 = min(fw, target.x + target.w)
        y2 = min(fh, target.y + target.h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame  # fallback: save full frame

        patch = cv2.resize(crop, (self.patch_size, self.patch_size))

        ts = int(time.time() * 1000)
        filename = self.save_dir / f"frame_{ts}.jpg"
        cv2.imwrite(str(filename), patch, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return filename
