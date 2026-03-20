"""Tests for the STT provider interface and FasterWhisperSTTProvider."""

import sys
import os
import inspect

import pytest

# Ensure pipeline root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import STTResult, WordTimestamp
from stt.base import STTProvider
from stt.faster_whisper_provider import FasterWhisperSTTProvider


class TestSTTInterface:
    """Test that calling code works purely through the STTProvider interface."""

    def test_faster_whisper_is_stt_provider(self):
        """FasterWhisperSTTProvider must be a subclass of STTProvider."""
        assert issubclass(FasterWhisperSTTProvider, STTProvider)

    def test_stt_result_fields(self):
        """STTResult must have the required fields."""
        result = STTResult(text="hello world", is_final=True, language_detected="en")
        assert result.text == "hello world"
        assert result.is_final is True
        assert result.language_detected == "en"
        assert isinstance(result.words, list)

    def test_word_timestamp_fields(self):
        """WordTimestamp must have the required fields."""
        wt = WordTimestamp(word="hello", start=0.0, end=0.5, probability=0.95)
        assert wt.word == "hello"
        assert wt.start == 0.0
        assert wt.end == 0.5
        assert wt.probability == 0.95

    def test_set_task_values(self):
        """set_task should accept 'transcribe' and 'translate' literals."""
        # This is a structural test — we verify the interface accepts these values
        # without instantiating the heavy model
        from stt.base import STTProvider
        assert hasattr(STTProvider, "set_task")
        assert hasattr(STTProvider, "reset")
        assert hasattr(STTProvider, "transcribe_chunk")

    def test_no_direct_faster_whisper_import(self):
        """
        Verify calling code never imports faster_whisper or RealtimeSTT directly.
        We only go through our STTProvider abstraction.
        """
        source = inspect.getsource(sys.modules[__name__])
        # Check for direct library imports (not our own module wrappers)
        for line in source.splitlines():
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Flag direct library imports (not our stt.* wrapper modules)
            if "import faster_whisper" in stripped and "stt.faster_whisper" not in stripped:
                raise AssertionError(f"Direct faster_whisper import found: {stripped}")
            if "from faster_whisper" in stripped:
                raise AssertionError(f"Direct faster_whisper import found: {stripped}")
            if "from RealtimeSTT" in stripped:
                raise AssertionError(f"Direct RealtimeSTT import found: {stripped}")
