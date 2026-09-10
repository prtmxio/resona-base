import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from resona_base import VoiceProfile, synthesize
from resona_base.tts import _get_tts
from resona_base.util import split_sentences

PROFILES_DIR = Path("profiles")
OUTPUT_DIR = Path("say_output")
SR = 24000


def _player(q: "queue.Queue[np.ndarray | None]") -> None:
    stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32")
    stream.start()
    while True:
        chunk = q.get()
        if chunk is None:
            break
        stream.write(chunk)  # blocks until the device drains this chunk
    stream.stop()
    stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Speak text through a cached VoiceProfile")
    parser.add_argument("--voice", required=True, help="profile name under profiles/ (e.g. jo)")
    parser.add_argument("--text", required=True)
    parser.add_argument("--save", action="store_true", help="also write wavs to say_output/")
    args = parser.parse_args()

    profile = VoiceProfile.load(PROFILES_DIR / f"{args.voice}.pt")
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"[{args.voice}] {len(split_sentences(args.text))} sentence(s)")
    t = time.perf_counter()
    _get_tts()
    print(f"  model load: {(time.perf_counter() - t) * 1000:.0f} ms")

    q: "queue.Queue[np.ndarray | None]" = queue.Queue()
    player = threading.Thread(target=_player, args=(q,), daemon=True)
    player.start()

    prev = time.perf_counter()
    for i, chunk in enumerate(synthesize(args.text, profile)):
        gen = time.perf_counter() - prev
        prev = time.perf_counter()
        secs = len(chunk) / SR
        print(f"  s{i + 1}: gen {gen * 1000:6.0f} ms  ->  {secs:4.1f}s audio  ({gen / secs:.2f}x RTF)")
        q.put(chunk.astype(np.float32))
        if args.save:
            sf.write(OUTPUT_DIR / f"{args.voice}_{i}.wav", chunk, SR)

    q.put(None)
    player.join()


if __name__ == "__main__":
    main()
