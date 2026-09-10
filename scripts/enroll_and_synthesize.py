import time
from pathlib import Path

import librosa
import soundfile as sf

from resona_base import VoiceProfile, enroll, synthesize

SAMPLES_DIR = Path("vendor/neutts-air/samples")
PROFILES_DIR = Path("profiles")
OUTPUT_DIR = Path("m1_output")

VOICES = ["jo", "dave"]
TEXT = (
    "Hello there. This is the voice profile system talking. "
    "If you can understand all three sentences, enrollment and caching both worked."
)


def enroll_voice(name: str) -> VoiceProfile:
    wav_path = SAMPLES_DIR / f"{name}.wav"
    txt_path = SAMPLES_DIR / f"{name}.txt"

    ref_audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    ref_text = txt_path.read_text().strip()

    profile = enroll(ref_audio, ref_text)
    profile.name = name
    profile.save(PROFILES_DIR / f"{name}.pt")
    return profile


def synthesize_voice(name: str) -> None:
    profile = VoiceProfile.load(PROFILES_DIR / f"{name}.pt")

    gen = synthesize(TEXT, profile)
    t_start = time.perf_counter()
    chunks = []
    for i, chunk in enumerate(gen):
        t_elapsed = time.perf_counter() - t_start
        if i == 0:
            print(f"[{name}] first-chunk latency: {t_elapsed * 1000:.0f}ms")
        chunks.append(chunk)
        out_path = OUTPUT_DIR / f"{name}_{i}.wav"
        sf.write(out_path, chunk, 24000)

    t_total = time.perf_counter() - t_start
    print(f"[{name}] {len(chunks)} sentences, {t_total * 1000:.0f}ms total")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Enrolling voices...")
    for name in VOICES:
        enroll_voice(name)
        print(f"[{name}] enrolled, profile saved to {PROFILES_DIR / f'{name}.pt'}")

    print("\nSynthesizing from cached profiles...")
    for name in VOICES:
        synthesize_voice(name)


if __name__ == "__main__":
    main()
