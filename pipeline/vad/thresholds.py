"""VAD tuning parameters — named constants for easy Phase 28 tuning."""

# Silero VAD thresholds (from TRD Section 14)
VAD_THRESHOLD: float = 0.5
MIN_SPEECH_DURATION_MS: int = 250
MIN_SILENCE_DURATION_MS: int = 300
WINDOW_SIZE_SAMPLES: int = 512
SPEECH_PAD_MS: int = 100

# Sample rate for VAD processing
VAD_SAMPLE_RATE: int = 16000
