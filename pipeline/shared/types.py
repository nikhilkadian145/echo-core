"""Shared data types for the ECHO pipeline."""

from dataclasses import dataclass, field


@dataclass
class VADResult:
    """Result from a VAD provider's process_chunk call."""
    is_speech: bool
    speech_prob: float


@dataclass
class WordTimestamp:
    """A single word with its timing information."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class STTResult:
    """Result from an STT provider's transcribe call."""
    text: str
    words: list[WordTimestamp] = field(default_factory=list)
    is_final: bool = False
    language_detected: str = ""
