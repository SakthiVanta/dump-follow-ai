"""
Speech-to-Text engine.
Supports Google (online) and Whisper (offline) backends.
"""
from __future__ import annotations

from typing import Optional

from robot.config import VoiceConfig
from robot.logger import logger


class STTEngine:
    """
    Wraps SpeechRecognition for mic capture + transcription.

    Parameters
    ----------
    config: VoiceConfig
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._cfg = config or VoiceConfig()
        self._recognizer = None
        self._microphone = None
        self._whisper_model = None
        self._available = False

    def start(self) -> bool:
        """Initialise audio devices. Returns True if successful."""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self._cfg.energy_threshold
            self._microphone = sr.Microphone()
            self._available = True

            if self._cfg.engine == "whisper":
                self._load_whisper()

            logger.info(f"STT engine started: {self._cfg.engine}")
            return True
        except ImportError as exc:
            logger.warning(f"SpeechRecognition not available: {exc}")
            return False
        except Exception as exc:
            logger.error(f"STT init failed: {exc}")
            return False

    def listen_once(self) -> Optional[str]:
        """
        Blocking listen + transcribe.  Returns transcript string or None.
        """
        if not self._available or self._recognizer is None:
            return None

        import speech_recognition as sr

        try:
            with self._microphone as source:  # type: ignore[union-attr]
                self._recognizer.adjust_for_ambient_noise(source, duration=0.3)
                logger.debug("Listening…")
                audio = self._recognizer.listen(
                    source,
                    phrase_time_limit=self._cfg.phrase_time_limit_s,
                )

            if self._cfg.engine == "whisper":
                return self._transcribe_whisper(audio)
            else:
                return self._transcribe_google(audio)

        except sr.WaitTimeoutError:
            logger.debug("STT timeout — no speech detected")
            return None
        except Exception as exc:
            logger.warning(f"STT error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_whisper(self) -> None:
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self._cfg.whisper_model}")
            self._whisper_model = whisper.load_model(self._cfg.whisper_model)
        except ImportError:
            logger.warning("openai-whisper not installed; falling back to Google STT")
            self._cfg.engine = "google"

    def _transcribe_whisper(self, audio) -> Optional[str]:
        import whisper
        import io
        import numpy as np
        import soundfile as sf

        wav_bytes = audio.get_wav_data()
        buf = io.BytesIO(wav_bytes)
        data, sr = sf.read(buf, dtype="float32")
        result = self._whisper_model.transcribe(data, language=self._cfg.language[:2])  # type: ignore
        return result.get("text", "").strip() or None

    def _transcribe_google(self, audio) -> Optional[str]:
        import speech_recognition as sr
        try:
            text = self._recognizer.recognize_google(  # type: ignore[union-attr]
                audio, language=self._cfg.language
            )
            logger.debug(f"Heard: {text!r}")
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as exc:
            logger.warning(f"Google STT request error: {exc}")
            return None
