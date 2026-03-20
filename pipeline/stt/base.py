from abc import ABC, abstractmethod
from typing import Literal, Callable, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import STTResult


class STTProvider(ABC):
    """
    Abstract interface for Speech-to-Text.

    Designed for streaming real-time transcription.
    """

    def __init__(self) -> None:
        self._on_realtime_update: Optional[Callable[[str], None]] = None
        self._on_transcription_complete: Optional[Callable[[STTResult], None]] = None

    def on_realtime_update(self, callback: Callable[[str], None]) -> None:
        """Register callback for instant real-time text updates."""
        self._on_realtime_update = callback

    def on_transcription_complete(self, callback: Callable[[STTResult], None]) -> None:
        """Register callback for final confirmed transcription of an utterance."""
        self._on_transcription_complete = callback

    @abstractmethod
    def feed_audio(self, pcm_chunk: bytes) -> None:
        """
        Feed a continuous stream of raw PCM audio bytes (16-bit, 16kHz, mono).
        The provider internally handles VAD and chunking.
        """
        ...

    @abstractmethod
    def set_task(self, task: Literal["transcribe", "translate"]) -> None:
        """Set the transcription task (transcribe vs translate)."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up threads and resources."""
        ...
