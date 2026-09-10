import os
from typing import Iterator

import numpy as np
import torch

# torch>=2.6 defaults torch.load to weights_only=True, which breaks
# neucodec's own internal `torch.load(ckpt_path, map_location)` call on its
# pytorch_model.bin checkpoint. Patched here rather than in the installed
# package; safe because the checkpoint is an official neuphonic release.
_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _patched_load

from neutts import NeuTTS

from .util import split_sentences
from .voice_profile import VoiceProfile

_BACKBONE_REPO = os.environ.get("RESONA_BACKBONE_PATH", "neuphonic/neutts-air-q4-gguf")

_tts: NeuTTS | None = None


def _get_tts() -> NeuTTS:
    global _tts
    if _tts is None:
        # Backbone on GPU (llama.cpp, the token-by-token loop). Codec on
        # CPU: NeuCodec's semantic sub-model (facebook/w2v-bert-2.0, fp32)
        # alone is ~2.3GB, which doesn't fit alongside the backbone in the
        # RTX 2050's ~3.68GiB usable VRAM. Codec runs once per call, so CPU
        # cost is fine.
        _tts = NeuTTS(
            backbone_repo=_BACKBONE_REPO,
            backbone_device="gpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu",
            language="en-us",
        )
    return _tts


def enroll(ref_audio: np.ndarray, ref_text: str) -> VoiceProfile:
    tts = _get_tts()
    # [T] -> [1, 1, T]: NeuCodec's tensor-input path expects [B, 1, T] and,
    # unlike its path-based load, does not resample — ref_audio must
    # already be mono float32 @ 16kHz (see util.resample for callers).
    wav_tensor = torch.from_numpy(ref_audio).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        # [1, 1, T] -> [1, 1, F] -> [F]: squeeze batch and channel dims off
        # the codec's frame-indexed token sequence.
        ref_codes = tts.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
    return VoiceProfile(ref_codes=ref_codes, ref_text=ref_text)


def synthesize(text: str, profile: VoiceProfile) -> Iterator[np.ndarray]:
    tts = _get_tts()
    for sentence in split_sentences(text):
        yield tts.infer(sentence, profile.ref_codes, profile.ref_text)
