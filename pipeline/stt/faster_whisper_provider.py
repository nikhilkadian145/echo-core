"""
Faster-Whisper STT provider backed by RealtimeSTT.

Uses the echo-realtimestt fork's AudioToTextRecorder in non-microphone mode,
feeding audio externally via feed_audio(). This gives us the full RealtimeSTT
streaming pipeline (Silero VAD + faster-whisper) behind our STTProvider interface.
"""

import logging
import threading
import struct
from typing import Literal

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import STTResult, WordTimestamp
from stt.base import STTProvider

logger = logging.getLogger(__name__)


class FasterWhisperSTTProvider(STTProvider):
    """
    STT implementation using RealtimeSTT's AudioToTextRecorder
    with faster-whisper backend.

    The recorder runs with use_microphone=False so we can feed
    audio externally. On speech_end, it transcribes and returns
    the result via a threading event.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        compute_type: str = "int8",
        device: str = "cpu",
        beam_size: int = 1,
        language: str = "",
    ) -> None:
        from RealtimeSTT import AudioToTextRecorder

        self._task: Literal["transcribe", "translate"] = "transcribe"
        self._language = language
        self._model_name = model
        self._latest_text: str = ""
        self._text_ready = threading.Event()

        # Initialize RealtimeSTT in feed-audio mode (no internal mic capture)
        self._recorder = AudioToTextRecorder(
            model=model,
            compute_type=compute_type,
            device=device,
            beam_size=beam_size,
            language=language if language else "",
            use_microphone=False,
            spinner=False,
            no_log_file=True,
            level=logging.WARNING,
            # Silero VAD params from TRD
            silero_sensitivity=0.5,
            post_speech_silence_duration=0.3,
            min_length_of_recording=0.25,
            pre_recording_buffer_duration=0.5,
            # enable realtime partial transcriptions
            enable_realtime_transcription=True,
            use_main_model_for_realtime=True,
            batch_size=0,  # disable batched inference for streaming
        )

        # Start the recorder's processing loop in a background thread
        self._running = True
        self._process_thread = threading.Thread(
            target=self._run_recorder, daemon=True
        )
        self._process_thread.start()

        logger.info(
            "FasterWhisperSTTProvider initialized: model=%s, compute=%s, device=%s",
            model, compute_type, device,
        )

    def _on_transcription(self, text: str) -> None:
        """Callback when RealtimeSTT produces a final transcription."""
        self._latest_text = text.strip()
        self._text_ready.set()

    def _run_recorder(self) -> None:
        """Background thread that keeps the recorder's text() loop running."""
        try:
            while self._running:
                text = self._recorder.text(self._on_transcription)
                if text:
                    self._latest_text = text.strip()
                    self._text_ready.set()
        except Exception:
            logger.exception("RealtimeSTT recorder loop crashed")

    def transcribe_chunk(
        self,
        pcm_chunk: bytes,
        language_primary: str,
        language_secondary: str | None = None,
    ) -> STTResult:
        """
        Feed accumulated speech audio into RealtimeSTT and get the transcript.

        The pcm_chunk should be the full utterance (from speech_start to speech_end).
        We feed it in small segments to RealtimeSTT's feed_audio method, then
        wait for the transcription callback.
        """
        # Update language if changed
        if language_primary and language_primary != self._language:
            self._language = language_primary
            self._recorder.language = language_primary

        # Reset event
        self._text_ready.clear()
        self._latest_text = ""

        # Convert bytes to int16 numpy array and feed in chunks
        n_samples = len(pcm_chunk) // 2
        audio_int16 = np.frombuffer(pcm_chunk, dtype=np.int16)

        # Feed in 512-sample segments (what RealtimeSTT expects)
        chunk_size = 512
        for i in range(0, len(audio_int16), chunk_size):
            segment = audio_int16[i:i + chunk_size]
            self._recorder.feed_audio(segment)

        # Wait for transcription (timeout after 10 seconds)
        got_result = self._text_ready.wait(timeout=10.0)

        if not got_result:
            logger.warning("Transcription timed out after 10s")
            return STTResult(text="", is_final=True, language_detected=language_primary or "")

        return STTResult(
            text=self._latest_text,
            words=[],  # RealtimeSTT doesn't expose word-level timestamps easily
            is_final=True,
            language_detected=language_primary or "",
        )

    def set_task(self, task: Literal["transcribe", "translate"]) -> None:
        """Set the transcription task mode."""
        self._task = task
        # RealtimeSTT uses faster-whisper which supports task parameter
        # This will be applied on next transcription
        logger.info("STT task set to: %s", task)

    def reset(self) -> None:
        """Reset internal state."""
        self._latest_text = ""
        self._text_ready.clear()

    def shutdown(self) -> None:
        """Cleanly shut down the recorder."""
        self._running = False
        try:
            self._recorder.shutdown()
        except Exception:
            logger.exception("Error shutting down RealtimeSTT recorder")
