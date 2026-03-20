#!/usr/bin/env python3
"""
ECHO Phase 1 PoC — VAD + STT Pipeline

Usage:
    python pipeline/poc/test_vad_stt.py                  # live mic
    python pipeline/poc/test_vad_stt.py --file test.wav  # from file
    python pipeline/poc/test_vad_stt.py --debug           # show VAD probs

Raw mic audio → Silero VAD gate → faster-whisper transcript.
Standalone Python script, no Rust, no UI.
"""

import argparse
import datetime
import struct
import sys
import os
import time

# Ensure pipeline root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sounddevice as sd
import soundfile as sf

from shared.types import VADResult
from vad.base import VADProvider
from vad.silero import SileroVADProvider
from stt.base import STTProvider
from stt.faster_whisper_provider import FasterWhisperSTTProvider
from vad.thresholds import VAD_SAMPLE_RATE, WINDOW_SIZE_SAMPLES


def run_from_file(
    filepath: str,
    vad: VADProvider,
    stt: STTProvider,
    debug: bool = False,
    language: str = "en",
    language_secondary: str | None = None,
) -> None:
    """Process a WAV file through the VAD → STT pipeline."""
    print(f"[INFO] Processing file: {filepath}")

    audio_data, sample_rate = sf.read(filepath, dtype="int16")

    # If stereo, take first channel
    if audio_data.ndim == 2:
        audio_data = audio_data[:, 0]

    # Resample to 16kHz if needed
    if sample_rate != VAD_SAMPLE_RATE:
        from scipy.signal import resample as scipy_resample
        num_samples = int(len(audio_data) * VAD_SAMPLE_RATE / sample_rate)
        audio_data = scipy_resample(audio_data, num_samples).astype(np.int16)

    pcm_bytes = audio_data.tobytes()
    chunk_bytes = WINDOW_SIZE_SAMPLES * 2  # 16-bit = 2 bytes per sample

    # Accumulator for speech frames
    speech_buffer = bytearray()
    is_accumulating = False
    utterance_count = 0

    def on_speech_start():
        nonlocal is_accumulating
        is_accumulating = True
        if debug:
            print(f"  [VAD] speech_start")

    def on_speech_end():
        nonlocal is_accumulating, speech_buffer, utterance_count
        is_accumulating = False

        if len(speech_buffer) > 0:
            utterance_count += 1
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            start_t = time.time()
            result = stt.transcribe_chunk(
                bytes(speech_buffer), language, language_secondary
            )
            elapsed = time.time() - start_t
            print(f"[{ts}] (utterance {utterance_count}, {elapsed:.2f}s) {result.text}")
            speech_buffer = bytearray()
            stt.reset()
            vad.reset()

        if debug:
            print(f"  [VAD] speech_end")

    vad.on_speech_start(on_speech_start)
    vad.on_speech_end(on_speech_end)

    # Process chunk by chunk
    for offset in range(0, len(pcm_bytes), chunk_bytes):
        chunk = pcm_bytes[offset:offset + chunk_bytes]
        if len(chunk) < chunk_bytes:
            # Pad the last chunk with silence
            chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))

        result = vad.process_chunk(chunk)

        if debug:
            print(f"  [VAD] prob={result.speech_prob:.3f} speech={result.is_speech}")

        if is_accumulating:
            speech_buffer.extend(chunk)

    # Flush any remaining speech
    if len(speech_buffer) > 0:
        utterance_count += 1
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        result = stt.transcribe_chunk(
            bytes(speech_buffer), language, language_secondary
        )
        print(f"[{ts}] (utterance {utterance_count}, final flush) {result.text}")

    print(f"\n[INFO] Done. {utterance_count} utterance(s) detected.")


def run_live_mic(
    vad: VADProvider,
    stt: STTProvider,
    debug: bool = False,
    language: str = "en",
    language_secondary: str | None = None,
) -> None:
    """Capture from the default mic and pipe through VAD → STT."""
    print("[INFO] Listening on default microphone (16kHz, mono, 32ms chunks)")
    print("[INFO] Press Ctrl+C to stop.\n")

    speech_buffer = bytearray()
    is_accumulating = False
    utterance_count = 0

    def on_speech_start():
        nonlocal is_accumulating
        is_accumulating = True
        if debug:
            print(f"  [VAD] speech_start")

    def on_speech_end():
        nonlocal is_accumulating, speech_buffer, utterance_count
        is_accumulating = False

        if len(speech_buffer) > 0:
            utterance_count += 1
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            start_t = time.time()
            result = stt.transcribe_chunk(
                bytes(speech_buffer), language, language_secondary
            )
            elapsed = time.time() - start_t
            print(f"[{ts}] (utterance {utterance_count}, {elapsed:.2f}s) {result.text}")
            speech_buffer = bytearray()
            stt.reset()
            vad.reset()

        if debug:
            print(f"  [VAD] speech_end")

    vad.on_speech_start(on_speech_start)
    vad.on_speech_end(on_speech_end)

    def audio_callback(indata, frames, time_info, status):
        nonlocal speech_buffer, is_accumulating

        if status:
            print(f"  [WARN] sounddevice status: {status}", file=sys.stderr)

        # Convert float32 from sounddevice to int16 bytes
        audio_int16 = (indata[:, 0] * 32768).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()

        chunk_bytes = WINDOW_SIZE_SAMPLES * 2
        for offset in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[offset:offset + chunk_bytes]
            if len(chunk) < chunk_bytes:
                chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))

            result = vad.process_chunk(chunk)

            if debug:
                print(f"  [VAD] prob={result.speech_prob:.3f} speech={result.is_speech}")

            if is_accumulating:
                speech_buffer.extend(chunk)

    try:
        with sd.InputStream(
            samplerate=VAD_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=WINDOW_SIZE_SAMPLES,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped. {utterance_count} utterance(s) recognized.")

    # Flush remaining
    if len(speech_buffer) > 0:
        result = stt.transcribe_chunk(
            bytes(speech_buffer), language, language_secondary
        )
        print(f"[FINAL FLUSH] {result.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ECHO Phase 1 PoC: VAD + STT")
    parser.add_argument("--file", type=str, help="Path to a WAV file (instead of live mic)")
    parser.add_argument("--debug", action="store_true", help="Print VAD probabilities per chunk")
    parser.add_argument("--language", type=str, default="en", help="Primary language code")
    parser.add_argument("--language-secondary", type=str, default=None, help="Secondary language code")
    parser.add_argument(
        "--model", type=str, default="large-v3-turbo",
        help="Whisper model name (default: large-v3-turbo)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ECHO Phase 1 PoC — VAD + STT Pipeline")
    print("=" * 60)

    # Instantiate providers via interfaces (no direct library imports here)
    print("[INIT] Loading Silero VAD model...")
    vad_provider: VADProvider = SileroVADProvider()

    print(f"[INIT] Loading faster-whisper via RealtimeSTT (model={args.model})...")
    stt_provider: STTProvider = FasterWhisperSTTProvider(
        model=args.model,
        compute_type="int8",
        device="cpu",
        beam_size=1,
        language=args.language,
    )

    print("[INIT] Pipeline ready!\n")

    if args.file:
        run_from_file(
            args.file, vad_provider, stt_provider,
            debug=args.debug,
            language=args.language,
            language_secondary=args.language_secondary,
        )
    else:
        run_live_mic(
            vad_provider, stt_provider,
            debug=args.debug,
            language=args.language,
            language_secondary=args.language_secondary,
        )


if __name__ == "__main__":
    main()
