"""Silero VAD provider implementation."""

import logging
import struct

import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import VADResult
from vad.base import VADProvider
from vad.thresholds import (
    VAD_THRESHOLD,
    MIN_SPEECH_DURATION_MS,
    MIN_SILENCE_DURATION_MS,
    WINDOW_SIZE_SAMPLES,
    SPEECH_PAD_MS,
    VAD_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class SileroVADProvider(VADProvider):
    """
    VAD implementation using Silero VAD v5.

    Loads the model once at init, processes 512-sample (32ms @ 16kHz) chunks,
    and fires speech_start / speech_end callbacks based on a simple state machine.
    """

    def __init__(self) -> None:
        super().__init__()

        # Load Silero VAD model (once, not per-chunk)
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model.eval()

        # State machine
        self._is_speaking: bool = False
        self._speech_frames: int = 0
        self._silence_frames: int = 0

        # Derived thresholds in frames (each frame = WINDOW_SIZE_SAMPLES / SAMPLE_RATE seconds)
        frame_duration_ms = (WINDOW_SIZE_SAMPLES / VAD_SAMPLE_RATE) * 1000
        self._min_speech_frames = int(MIN_SPEECH_DURATION_MS / frame_duration_ms)
        self._min_silence_frames = int(MIN_SILENCE_DURATION_MS / frame_duration_ms)

        logger.info(
            "SileroVADProvider initialized: threshold=%.2f, "
            "min_speech_frames=%d, min_silence_frames=%d",
            VAD_THRESHOLD,
            self._min_speech_frames,
            self._min_silence_frames,
        )

    def process_chunk(self, pcm_chunk: bytes) -> VADResult:
        """
        Process a raw PCM chunk (16-bit LE, 16kHz, mono).

        Returns VADResult with speech probability and is_speech flag.
        Also manages the speech state machine and fires callbacks.
        """
        # Convert bytes to float32 tensor
        n_samples = len(pcm_chunk) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm_chunk)
        audio_float = np.array(samples, dtype=np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float)

        # Run Silero VAD
        with torch.no_grad():
            speech_prob = self._model(audio_tensor, VAD_SAMPLE_RATE).item()

        is_speech = speech_prob >= VAD_THRESHOLD

        # State machine transitions
        if is_speech:
            self._silence_frames = 0
            self._speech_frames += 1

            if (
                not self._is_speaking
                and self._speech_frames >= self._min_speech_frames
            ):
                self._is_speaking = True
                if self._on_speech_start:
                    self._on_speech_start()
                logger.debug("speech_start fired (prob=%.3f)", speech_prob)

        else:
            self._speech_frames = 0

            if self._is_speaking:
                self._silence_frames += 1
                if self._silence_frames >= self._min_silence_frames:
                    self._is_speaking = False
                    if self._on_speech_end:
                        self._on_speech_end()
                    logger.debug("speech_end fired after %d silence frames", self._silence_frames)
                    self._silence_frames = 0

        return VADResult(is_speech=is_speech, speech_prob=speech_prob)

    def reset(self) -> None:
        """Reset internal state between utterances."""
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._model.reset_states()
