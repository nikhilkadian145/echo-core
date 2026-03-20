"""STT provider package."""

from stt.base import STTProvider
from stt.faster_whisper_provider import FasterWhisperSTTProvider

__all__ = ["STTProvider", "FasterWhisperSTTProvider"]
