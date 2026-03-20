"""Pytest fixtures for ECHO pipeline tests."""

import struct
import sys
import os
import numpy as np
import pytest

# Ensure pipeline root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def silence_chunk() -> bytes:
    """512 samples of silence as 16-bit PCM bytes."""
    return b"\x00" * (512 * 2)


@pytest.fixture
def speech_chunk() -> bytes:
    """
    512 samples of a loud 440Hz sine wave as 16-bit PCM bytes.
    Designed to trigger VAD with high probability.
    """
    sample_rate = 16000
    n_samples = 512
    t = np.arange(n_samples) / sample_rate
    # Generate a loud sine wave at 440 Hz
    sine = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)
    return sine.tobytes()


@pytest.fixture
def long_speech_audio() -> bytes:
    """
    2 seconds of loud sine wave audio at 16kHz as 16-bit PCM bytes.
    Useful for testing utterance accumulation.
    """
    sample_rate = 16000
    duration = 2.0
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    sine = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)
    return sine.tobytes()
