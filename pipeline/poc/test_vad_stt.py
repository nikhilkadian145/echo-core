#!/usr/bin/env python3
"""
ECHO Phase 1 PoC — Instant Realtime STT Pipeline

Usage:
    python pipeline/poc/test_vad_stt.py                  # live mic
    python pipeline/poc/test_vad_stt.py --file test.wav  # from file

Audio → STTProvider (RealtimeSTT) → Instant text callbacks.
Standalone Python script, no Rust, no UI.
"""

import argparse
import datetime
import sys
import os
import time

# Fix OpenMP duplicate library error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure pipeline root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.types import STTResult
from stt.base import STTProvider
from stt.faster_whisper_provider import FasterWhisperSTTProvider


def _print_realtime(text: str):
    """Callback for instant partial text (prints on same line)."""
    # Use carriage return to overwrite the current line
    sys.stdout.write(f"\r\033[K[Realtime] {text}")
    sys.stdout.flush()


def _print_final(result: STTResult):
    """Callback for final confirmed utterance."""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    # Print on a fresh line, erasing any current realtime text
    sys.stdout.write(f"\r\033[K[{ts}] (Final) {result.text}\n")
    sys.stdout.flush()


def run_from_file(
    filepath: str,
    stt: STTProvider,
) -> None:
    """Process a WAV file through the continuous STT pipeline."""
    import soundfile as sf
    import numpy as np
    from vad.thresholds import VAD_SAMPLE_RATE, WINDOW_SIZE_SAMPLES

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

    stt.on_realtime_update(_print_realtime)
    stt.on_transcription_complete(_print_final)

    # Process chunk by chunk to simulate streaming
    for offset in range(0, len(pcm_bytes), chunk_bytes):
        chunk = pcm_bytes[offset:offset + chunk_bytes]
        if len(chunk) < chunk_bytes:
            chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))

        stt.feed_audio(chunk)
        # Sleep exactly the duration of the chunk to simulate realtime
        time.sleep(WINDOW_SIZE_SAMPLES / VAD_SAMPLE_RATE)

    print("\n[INFO] End of file reached. Waiting for final transcription flush...")
    time.sleep(2.0)
    stt.shutdown()


def run_live_mic(
    stt: STTProvider,
) -> None:
    """Native RealtimeSTT mic capture (uses PyAudio internally for lowest latency)."""
    print("[INFO] Listening on default microphone (via RealtimeSTT native capture)")
    print("[INFO] Speak now! The text should appear instantly.")
    print("[INFO] Press Ctrl+C to stop.\n")

    stt.on_realtime_update(_print_realtime)
    stt.on_transcription_complete(_print_final)

    try:
        # The provider's internal threads handle all VAD and audio capture
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped. Shutting down...")
    
    finally:
        stt.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="ECHO Phase 1 PoC: Instant STT")
    parser.add_argument("--file", type=str, help="Path to a WAV file (instead of live mic)")
    parser.add_argument("--language", type=str, default="en", help="Primary language code")
    parser.add_argument(
        "--model", type=str, default="large-v3-turbo",
        help="Whisper model name (default: large-v3-turbo)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ECHO Phase 1 PoC — Instant Realtime STT Pipeline")
    print("=" * 60)

    use_mic = args.file is None

    print(f"[INIT] Loading RealtimeSTT (model={args.model}, native_mic={use_mic})...")
    stt_provider: STTProvider = FasterWhisperSTTProvider(
        model=args.model,
        compute_type="int8",
        device="cpu",
        language=args.language,
        use_microphone=use_mic,
    )
    print("[INIT] Pipeline ready!\n")

    if args.file:
        run_from_file(args.file, stt_provider)
    else:
        run_live_mic(stt_provider)


if __name__ == "__main__":
    main()
