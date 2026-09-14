import os

import numpy as np
from faster_whisper import WhisperModel

from .util import to_mono_16k

_ASR_MODEL = os.environ.get("RESONA_ASR_PATH", "small")

# "small" is multilingual and auto-detects language per call, but a 4-5s clip
# with quiet leading/trailing audio gives it too little signal — it'll pin the
# wrong language outright rather than mis-transcribe within the right one.
# Pinned to English for now; M6's Hinglish work will need this adjustable
# again (set RESONA_ASR_LANGUAGE=auto to fall back to detection, or e.g. "hi").
_ASR_LANGUAGE = os.environ.get("RESONA_ASR_LANGUAGE", "en")

_asr: WhisperModel | None = None


def _get_asr() -> WhisperModel:
    global _asr
    if _asr is None:
        # int8_float16: CTranslate2's int8-quantized GPU path (weights int8,
        # accumulation fp16) — same "small" checkpoint as CPU, far less VRAM
        # and faster than running it fp16/fp32.
        _asr = WhisperModel(_ASR_MODEL, device="cuda", compute_type="int8_float16")
    return _asr


def transcribe(audio: np.ndarray, sr: int) -> str:
    model = _get_asr()
    audio = to_mono_16k(audio, sr)
    language = None if _ASR_LANGUAGE == "auto" else _ASR_LANGUAGE
    # condition_on_previous_text=False: distilled decoders (e.g. distil-large-v3)
    # are prone to repetition/hallucination when primed with prior-segment text;
    # our clips are short standalone turns, so cross-segment conditioning buys
    # nothing anyway.
    segments, _info = model.transcribe(
        audio, language=language, vad_filter=True, condition_on_previous_text=False
    )
    return " ".join(segment.text.strip() for segment in segments).strip()
