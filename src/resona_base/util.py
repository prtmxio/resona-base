import re

import librosa
import numpy as np

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)  # (T, C) -> (T,)
    return resample(audio, sr, 16000)
