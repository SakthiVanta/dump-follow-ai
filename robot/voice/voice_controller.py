"""
Async voice command loop — runs STT + intent parsing in a background thread.
Publishes intents via asyncio.Queue so main loop can consume them.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Optional

from robot.config import VoiceConfig
from robot.voice.intent import Intent, IntentParser
from robot.voice.stt import STTEngine
from robot.logger import logger


class VoiceController:
    """
    Starts a background thread that continuously listens for voice commands
    and puts parsed Intents onto an asyncio queue.

    Usage
    -----
    vc = VoiceController(config)
    vc.start()
    intent = await vc.queue.get()
    vc.stop()
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._cfg = config or VoiceConfig()
        self._stt = STTEngine(self._cfg)
        self._parser = IntentParser()
        self._queue: asyncio.Queue[Intent] = asyncio.Queue(maxsize=10)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def queue(self) -> asyncio.Queue[Intent]:
        return self._queue

    def start(self) -> bool:
        """Start background listen thread. Returns False if mic unavailable."""
        if not self._stt.start():
            return False
        self._running = True
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()

        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="VoiceThread"
        )
        self._thread.start()
        logger.info("VoiceController started")
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("VoiceController stopped")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        while self._running:
            text = self._stt.listen_once()
            if text:
                intent = self._parser.parse(text)
                logger.info(f"Intent: {intent.type.value!r} ← {text!r}")
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._queue.put(intent), self._loop
                    )
