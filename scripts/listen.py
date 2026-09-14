import argparse
import time
from pathlib import Path

import soundfile as sf

from resona_base import is_turn_complete, transcribe

TEST_CLIPS_DIR = Path("test_clips")


def _run_one(path: Path) -> None:
    audio, sr = sf.read(path, dtype="float32")

    t0 = time.perf_counter()
    text = transcribe(audio, sr)
    t_asr = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    complete = is_turn_complete(audio, sr)
    t_turn = (time.perf_counter() - t0) * 1000

    expected = None
    if path.stem.startswith("complete_"):
        expected = True
    elif path.stem.startswith("incomplete_"):
        expected = False

    verdict = ""
    if expected is not None:
        verdict = "  OK" if complete == expected else f"  MISMATCH (expected {expected})"

    print(path.name)
    print(f'  text:          "{text}"  ({t_asr:.0f}ms)')
    print(f"  turn_complete: {complete}  ({t_turn:.0f}ms){verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test transcribe() and is_turn_complete() on wav clips")
    parser.add_argument("clips", nargs="*", help="wav files; defaults to everything under test_clips/")
    args = parser.parse_args()

    paths = [Path(p) for p in args.clips] if args.clips else sorted(TEST_CLIPS_DIR.glob("*.wav"))
    if not paths:
        parser.error(f"no clips given and nothing found under {TEST_CLIPS_DIR}/")

    for path in paths:
        _run_one(path)
        print()


if __name__ == "__main__":
    main()
