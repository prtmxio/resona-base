import os

import soundfile as sf
import torch

# torch>=2.6 defaults torch.load to weights_only=True, which breaks
# neucodec's own internal `torch.load(ckpt_path, map_location)` call on its
# pytorch_model.bin checkpoint. Patched here rather than in the installed
# package; safe because the checkpoint is an official neuphonic release we
# downloaded and sha256-verified ourselves.
_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _patched_load

from neutts import NeuTTS


def main(input_text, ref_audio_path, ref_text, backbone, output_path="output.wav"):
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    # GGUF Q4 backbone on GPU (llama.cpp) for the actual autoregressive
    # generation loop. NeuCodec on CPU: its semantic sub-model
    # (facebook/w2v-bert-2.0, fp32) alone is ~2.3GB, which doesn't fit
    # alongside the backbone in the RTX 2050's ~3.68GiB usable VRAM.
    # Codec encode/decode is a single forward pass, so CPU cost is fine.
    tts = NeuTTS(
        backbone_repo=backbone,
        backbone_device="gpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
        # local .gguf paths aren't in BACKBONE_LANGUAGE_MAP, so language
        # auto-detection fails unless we pass it explicitly.
        language="en-us",
    )

    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    ref_codes_path = ref_audio_path.replace(".wav", ".pt")
    if not os.path.exists(ref_codes_path):
        print("Encoding reference audio")
        ref_codes = tts.encode_reference(ref_audio_path)
        torch.save(ref_codes, ref_codes_path)
    else:
        print("Loading pre-encoded reference audio")
        ref_codes = torch.load(ref_codes_path)

    print(f"Generating audio for input text: {input_text}")
    wav = tts.infer(input_text, ref_codes, ref_text)

    print(f"Saving output to {output_path}")
    sf.write(output_path, wav, 24000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS voice-clone runner (M0)")
    parser.add_argument("--input_text", type=str, required=True)
    parser.add_argument(
        "--ref_audio",
        type=str,
        default="vendor/neutts-air/samples/jo.wav",
        help="Path to reference audio file",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="vendor/neutts-air/samples/jo.txt",
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument("--output_path", type=str, default="output.wav")
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-air-q4-gguf",
        help="Huggingface repo containing the GGUF backbone checkpoint",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        backbone=args.backbone,
        output_path=args.output_path,
    )
