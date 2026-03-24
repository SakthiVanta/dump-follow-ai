"""
Intent parser: maps raw transcribed text → robot mode/action.
Rule-based with keyword matching — lightweight, no external deps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class IntentType(str, Enum):
    FOLLOW = "follow_person"
    STOP = "stop"
    SEARCH = "search"
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SPEED_UP = "speed_up"
    SPEED_DOWN = "speed_down"
    IDLE = "idle"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    type: IntentType
    raw_text: str
    confidence: float = 1.0
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

_RULES: list[tuple[list[str], IntentType, dict]] = [
    # Patterns                              Intent            Extra params
    (["follow me", "follow", "track me"],  IntentType.FOLLOW,    {}),
    (["stop", "halt", "freeze", "stay"],   IntentType.STOP,      {}),
    (["search", "find", "look for"],       IntentType.SEARCH,    {}),
    (["go forward", "move forward",
      "forward", "ahead"],                 IntentType.FORWARD,   {}),
    (["go back", "backward", "reverse",
      "back up"],                          IntentType.BACKWARD,  {}),
    (["turn left", "go left", "left"],     IntentType.TURN_LEFT, {}),
    (["turn right", "go right", "right"],  IntentType.TURN_RIGHT,{}),
    (["faster", "speed up", "go faster"], IntentType.SPEED_UP,  {}),
    (["slower", "slow down", "go slow"],   IntentType.SPEED_DOWN,{}),
    (["idle", "wait", "pause"],            IntentType.IDLE,      {}),
]

_SPEED_PATTERN = re.compile(r"speed\s+(\d+)")


class IntentParser:
    """
    Lightweight rule-based intent parser.

    Example
    -------
    >>> parser = IntentParser()
    >>> intent = parser.parse("please follow me now")
    >>> intent.type
    <IntentType.FOLLOW: 'follow_person'>
    """

    def parse(self, text: str) -> Intent:
        text_lower = text.lower().strip()

        # Speed override: "speed 80" → SPEED_UP with param
        m = _SPEED_PATTERN.search(text_lower)
        if m:
            return Intent(
                type=IntentType.SPEED_UP,
                raw_text=text,
                confidence=0.9,
                params={"speed": int(m.group(1))},
            )

        for patterns, intent_type, params in _RULES:
            for pattern in patterns:
                if pattern in text_lower:
                    return Intent(
                        type=intent_type,
                        raw_text=text,
                        params=params,
                    )

        return Intent(type=IntentType.UNKNOWN, raw_text=text, confidence=0.0)
