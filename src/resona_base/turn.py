import os

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import WhisperFeatureExtractor

from .util import to_mono_16k

_SMART_TURN_REPO = os.environ.get("RESONA_SMART_TURN_REPO", "pipecat-ai/smart-turn-v3")
_SR = 16000
_SMART_TURN_WINDOW_S = 8

# If the tail of the buffer still has speech in it, the speaker is mid-utterance
# and there's no point running Smart Turn at all.
_MIN_TRAILING_SILENCE_S = 0.8
_SMART_TURN_THRESHOLD = 0.5

_vad_model = None
_smart_turn_session: ort.InferenceSession | None = None
_feature_extractor: WhisperFeatureExtractor | None = None


def _get_vad():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad(onnx=True)
    return _vad_model


def _resolve_smart_turn_onnx() -> str:
    # The exact filename isn't documented on the model card (an fp32 build
    # and a smaller int8 one both live in the repo); pick the int8 one at
    # runtime instead of hardcoding a guess that could 404.
    files = [f for f in list_repo_files(_SMART_TURN_REPO) if f.endswith(".onnx")]
    quantized = [f for f in files if "int8" in f or "quant" in f]
    filename = quantized[0] if quantized else files[0]
    return hf_hub_download(_SMART_TURN_REPO, filename)


def _get_smart_turn() -> tuple[ort.InferenceSession, WhisperFeatureExtractor]:
    global _smart_turn_session, _feature_extractor
    if _smart_turn_session is None:
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _smart_turn_session = ort.InferenceSession(_resolve_smart_turn_onnx(), sess_options=options)
        _feature_extractor = WhisperFeatureExtractor(chunk_length=_SMART_TURN_WINDOW_S)
    return _smart_turn_session, _feature_extractor


def _smart_turn_prob(audio: np.ndarray) -> float:
    session, feature_extractor = _get_smart_turn()
    inputs = feature_extractor(
        audio,
        sampling_rate=_SR,
        return_tensors="np",
        padding="max_length",
        max_length=_SMART_TURN_WINDOW_S * _SR,
        truncation=True,
        do_normalize=True,
    )
    (logits,) = session.run(None, {"input_features": inputs["input_features"]})
    logit = float(logits.reshape(-1)[0])  # (1, 1) -> scalar
    return 1 / (1 + np.exp(-logit))


def is_turn_complete(audio: np.ndarray, sr: int) -> bool:
    audio = to_mono_16k(audio, sr)
    segments = get_speech_timestamps(
        torch.from_numpy(audio), _get_vad(), sampling_rate=_SR, return_seconds=True
    )
    if not segments:
        return False

    tail_start = max(0.0, len(audio) / _SR - _MIN_TRAILING_SILENCE_S)
    still_talking = segments[-1]["end"] > tail_start
    if still_talking:
        return False

    return _smart_turn_prob(audio) >= _SMART_TURN_THRESHOLD
