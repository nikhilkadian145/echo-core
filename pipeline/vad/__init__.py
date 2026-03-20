"""VAD provider package."""

from vad.base import VADProvider
from vad.silero import SileroVADProvider

__all__ = ["VADProvider", "SileroVADProvider"]
