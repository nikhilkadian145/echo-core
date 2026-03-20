# Phase 1 — Language Pair Test Results

## Test Configuration

- **Model**: faster-whisper `large-v3-turbo` via RealtimeSTT
- **Compute**: int8, CPU-only
- **Beam size**: 1
- **VAD**: Silero VAD v5 (threshold=0.5)

## Results

| Test File | Primary Lang | Secondary Lang | Expected Content | Actual Transcript | WER | Language Detected |
|:---|:---|:---|:---|:---|:---|:---|
| `english_clear.wav` | en | — | TBD | TBD | TBD | TBD |
| `english_accented.wav` | en | — | TBD | TBD | TBD | TBD |
| `hindi_english_codeswitched.wav` | hi | en | TBD | TBD | TBD | TBD |
| `hindi_pure.wav` | hi | — | TBD | TBD | TBD | TBD |
| `korean_pure.wav` | ko | — | TBD | TBD | TBD | TBD |

## Notes

- Synthetic WAV files generated via `gTTS` are used as placeholders.
- Replace with real recordings for accurate WER benchmarks.
- WER target: `english_clear.wav` < 8%, `hindi_english_codeswitched.wav` < 15%.
