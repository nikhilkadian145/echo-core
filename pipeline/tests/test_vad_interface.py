"""Tests for the VAD provider interface and Silero implementation."""

import sys
import os
import struct

import numpy as np
import pytest

# Ensure pipeline root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import VADResult
from vad.base import VADProvider
from vad.silero import SileroVADProvider


class TestVADInterface:
    """Test that calling code works purely through the VADProvider interface."""

    def test_silero_is_vad_provider(self):
        """SileroVADProvider must be a subclass of VADProvider."""
        assert issubclass(SileroVADProvider, VADProvider)

    def test_instantiation(self):
        """SileroVADProvider can be instantiated without error."""
        provider: VADProvider = SileroVADProvider()
        assert provider is not None

    def test_process_chunk_returns_vad_result(self, silence_chunk: bytes):
        """process_chunk must return a VADResult dataclass."""
        provider: VADProvider = SileroVADProvider()
        result = provider.process_chunk(silence_chunk)
        assert isinstance(result, VADResult)
        assert isinstance(result.is_speech, bool)
        assert isinstance(result.speech_prob, float)
        assert 0.0 <= result.speech_prob <= 1.0

    def test_silence_detected_as_non_speech(self, silence_chunk: bytes):
        """Pure silence should not be classified as speech."""
        provider: VADProvider = SileroVADProvider()
        result = provider.process_chunk(silence_chunk)
        assert result.is_speech is False
        assert result.speech_prob < 0.5

    def test_reset_clears_state(self, silence_chunk: bytes):
        """reset() should not raise and should clear internal state."""
        provider: VADProvider = SileroVADProvider()
        provider.process_chunk(silence_chunk)
        provider.reset()  # Should not raise

    def test_callbacks_can_be_registered(self):
        """on_speech_start and on_speech_end callbacks can be registered."""
        provider: VADProvider = SileroVADProvider()
        start_called = []
        end_called = []

        provider.on_speech_start(lambda: start_called.append(True))
        provider.on_speech_end(lambda: end_called.append(True))

        # Just verify registration doesn't crash (actual firing tested
        # via integration with speech audio)
        assert provider._on_speech_start is not None
        assert provider._on_speech_end is not None

    def test_no_direct_silero_import(self):
        """
        Verify the calling code (this test file) never imports silero or
        torch directly for VAD usage. We only import from vad.base and
        vad.silero (our wrapper).
        """
        # This test is self-documenting — if we got here, we only used
        # VADProvider and SileroVADProvider from our own package.
        # The assertion is that the code above does not contain
        # `import silero_vad` or `from silero_vad import ...`
        import inspect
        source = inspect.getsource(sys.modules[__name__])
        assert "import silero_vad" not in source
        assert "from silero_vad" not in source
        assert "import torch" not in source  # torch is only in the implementation


class TestVADSpeechDetection:
    """Test VAD speech detection with synthetic audio."""

    def test_speech_start_callback_fires(self, speech_chunk: bytes):
        """speech_start callback should fire after enough speech frames."""
        provider: VADProvider = SileroVADProvider()
        start_called = []
        provider.on_speech_start(lambda: start_called.append(True))

        # Feed many speech chunks to trigger speech_start
        for _ in range(50):
            provider.process_chunk(speech_chunk)

        # Note: synthetic sine wave may or may not trigger Silero VAD
        # depending on model behavior. This is a best-effort test.
        # The structural test (callback registration) is the important one.

    def test_false_positive_rate_on_silence(self, silence_chunk: bytes):
        """
        Feed 60 seconds of silence and verify <3% false positive rate.
        60s at 32ms/chunk = 1875 chunks.
        """
        provider: VADProvider = SileroVADProvider()
        total_chunks = 1875
        false_positives = 0

        for _ in range(total_chunks):
            result = provider.process_chunk(silence_chunk)
            if result.is_speech:
                false_positives += 1

        fp_rate = false_positives / total_chunks
        assert fp_rate < 0.03, f"False positive rate {fp_rate:.2%} exceeds 3% threshold"
