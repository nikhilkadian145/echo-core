"""Abstract base class for Voice Activity Detection providers."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import VADResult


class VADProvider(ABC):
    """
    Abstract interface for Voice Activity Detection.

    All VAD implementations must subclass this. Calling code should only
    reference this interface — never import a specific VAD library directly.
    """

    def __init__(self) -> None:
        self._on_speech_start: Optional[Callable[[], None]] = None
        self._on_speech_end: Optional[Callable[[], None]] = None

    def on_speech_start(self, callback: Callable[[], None]) -> None:
        """Register a callback for when speech begins."""
        self._on_speech_start = callback

    def on_speech_end(self, callback: Callable[[], None]) -> None:
        """Register a callback for when speech ends."""
        self._on_speech_end = callback

    @abstractmethod
    def process_chunk(self, pcm_chunk: bytes) -> VADResult:
        """
        Process a single audio chunk and return a VAD decision.

        Args:
            pcm_chunk: Raw PCM audio bytes (16-bit, 16kHz, mono).

        Returns:
            VADResult with is_speech flag and speech probability.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g. between utterances)."""
        ...
