"""Abstract base class for Speech-to-Text providers."""

from abc import ABC, abstractmethod
from typing import Literal

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import STTResult


class STTProvider(ABC):
    """
    Abstract interface for Speech-to-Text.

    All STT implementations must subclass this. Calling code should only
    reference this interface — never import a specific STT library directly.
    """

    @abstractmethod
    def transcribe_chunk(
        self,
        pcm_chunk: bytes,
        language_primary: str,
        language_secondary: str | None = None,
    ) -> STTResult:
        """
        Transcribe an accumulated audio chunk.

        Args:
            pcm_chunk: Raw PCM audio bytes (16-bit, 16kHz, mono) —
                       typically an entire utterance accumulated between
                       speech_start and speech_end.
            language_primary: Primary language code (e.g. "en", "hi").
            language_secondary: Optional secondary language code for
                                code-switching scenarios.

        Returns:
            STTResult with transcript text, word timestamps, finality flag,
            and detected language.
        """
        ...

    @abstractmethod
    def set_task(self, task: Literal["transcribe", "translate"]) -> None:
        """
        Set the transcription task.

        - "transcribe": output in the same language as input
        - "translate": output is always English regardless of input language
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state between utterances."""
        ...
