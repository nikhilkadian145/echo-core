#!/usr/bin/env python3
"""
Generate synthetic test WAV files for Phase 1 language pair testing.

Uses gTTS (Google Text-to-Speech) to create test audio files.
Run: python pipeline/poc/generate_test_wavs.py
"""

import os
import sys

def generate_with_gtts():
    """Generate test WAVs using gTTS + pydub."""
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        import io
    except ImportError:
        print("Install gTTS and pydub: pip install gTTS pydub")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), "test_assets")
    os.makedirs(output_dir, exist_ok=True)

    test_cases = [
        {
            "filename": "english_clear.wav",
            "text": "The quick brown fox jumps over the lazy dog. "
                    "Technology continues to shape how we communicate and work.",
            "lang": "en",
        },
        {
            "filename": "english_accented.wav",
            "text": "Good morning everyone. Today we will discuss the project timeline "
                    "and the upcoming milestones for the quarter.",
            "lang": "en",  # gTTS accent varies; this is a placeholder
        },
        {
            "filename": "hindi_english_codeswitched.wav",
            "text": "Hello, aaj hum discuss karenge project ke baare mein. "
                    "Timeline thoda tight hai but we can manage.",
            "lang": "hi",
        },
        {
            "filename": "hindi_pure.wav",
            "text": "नमस्ते, आज का मौसम बहुत अच्छा है। "
                    "हम सभी को मिलकर काम करना होगा।",
            "lang": "hi",
        },
        {
            "filename": "korean_pure.wav",
            "text": "안녕하세요. 오늘 회의에서 프로젝트 현황을 발표하겠습니다.",
            "lang": "ko",
        },
    ]

    for case in test_cases:
        filepath = os.path.join(output_dir, case["filename"])
        if os.path.exists(filepath):
            print(f"  [SKIP] {case['filename']} already exists")
            continue

        print(f"  [GEN]  {case['filename']} ({case['lang']})")
        tts = gTTS(text=case["text"], lang=case["lang"])
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)

        # Convert MP3 to 16kHz mono WAV
        audio = AudioSegment.from_mp3(mp3_buffer)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(filepath, format="wav")
        print(f"         → {filepath}")

    print("\nDone! Test WAV files are in:", output_dir)


if __name__ == "__main__":
    generate_with_gtts()
