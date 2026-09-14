import os

import numpy as np
from faster_whisper import WhisperModel

from .util import to_mono_16k

_ASR_MODEL = os.environ.get("RESONA_ASR_PATH", "small")

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
    segments, _info = model.transcribe(audio, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()
