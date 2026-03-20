"""
Faster-Whisper STT provider backed by RealtimeSTT.

Uses the echo-realtimestt fork's AudioToTextRecorder in non-microphone mode,
feeding audio externally via feed_audio(). Provides continuous real-time 
transcription callbacks as well as final finalized text callbacks.
"""

import logging
import threading
from typing import Literal

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import STTResult
from stt.base import STTProvider

logger = logging.getLogger(__name__)


class FasterWhisperSTTProvider(STTProvider):
    """
    STT implementation using RealtimeSTT's AudioToTextRecorder.

    Handles real-time streaming text (instant) and then yields
    a final transcription at the end of an utterance.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        compute_type: str = "int8",
        device: str = "cpu",
        beam_size: int = 1,
        language: str = "en",
    ) -> None:
        super().__init__()
        from RealtimeSTT import AudioToTextRecorder

        self._task: Literal["transcribe", "translate"] = "transcribe"
        self._language = language

        logger.info(
            "Initializing RealtimeSTT (main model=%s, compute=%s, device=%s)",
            model, compute_type, device,
        )

        # Initialize RealtimeSTT in feed-audio mode (no internal mic capture)
        self._recorder = AudioToTextRecorder(
            model=model,
            compute_type=compute_type,
            device=device,
            beam_size=beam_size,
            language=language if language else "en",
            use_microphone=False,
            spinner=False,
            no_log_file=True,
            level=logging.WARNING,
            # Real-time partial transcriptions (instant updates)
            enable_realtime_transcription=True,
            use_main_model_for_realtime=False,
            realtime_model_type="tiny", # Tiny model for instant partials
            realtime_processing_pause=0.05,
            on_realtime_transcription_update=self._handle_realtime_update,
            # Silero VAD settings
            silero_sensitivity=0.4,
            webrtc_sensitivity=3,
            post_speech_silence_duration=0.5,
            min_length_of_recording=0.2,
        )

        # Start the background thread for finalized transcriptions
        self._running = True
        self._process_thread = threading.Thread(
            target=self._transcription_loop, daemon=True
        )
        self._process_thread.start()

        logger.info("RealtimeSTT provider initialized successfully.")

    def _handle_realtime_update(self, text: str) -> None:
        """Called repeatedly by RealtimeSTT with partial fast transcripts."""
        if self._on_realtime_update and text.strip():
            self._on_realtime_update(text)

    def _transcription_loop(self) -> None:
        """Background thread calling recorder.text() for final transcripts."""
        while self._running:
            try:
                # This call blocks until an utterance ends (silence detected),
                # then runs the main 'large-v3-turbo' model.
                text = self._recorder.text()
                if text and text.strip() and self._on_transcription_complete:
                    result = STTResult(
                        text=text.strip(),
                        words=[],  # RealtimeSTT abstracts words away in text() mode
                        is_final=True,
                        language_detected=self._language,
                    )
                    self._on_transcription_complete(result)
            except Exception as e:
                logger.error("Error in final transcription loop: %s", e)
                if not self._running:
                    break

    def feed_audio(self, pcm_chunk: bytes) -> None:
        """
        Feed continuous mic recording directly into RealtimeSTT.
        It has its own internal VAD gating.
        """
        self._recorder.feed_audio(pcm_chunk)

    def set_task(self, task: Literal["transcribe", "translate"]) -> None:
        """Set the transcription task mode."""
        self._task = task

    def shutdown(self) -> None:
        """Cleanly shut down the recorder."""
        self._running = False
        try:
            self._recorder.shutdown()
        except:
            pass
