from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class VoiceProfile:
    ref_codes: torch.Tensor
    ref_text: str
    sample_rate: int = 16000
    # not used by enroll()/synthesize() — the public API contract is
    # enroll(ref_audio, ref_text) with no name arg. Purely a caller-set
    # label for logging/filenames, e.g. profile.name = "jo".
    name: str = ""

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "ref_codes": self.ref_codes,
                "ref_text": self.ref_text,
                "sample_rate": self.sample_rate,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "VoiceProfile":
        # weights_only=False: this is a dict of plain Python values + one
        # tensor, not a raw state_dict, so torch's default restricted
        # unpickler (weights_only=True since torch 2.6) rejects it.
        data = torch.load(path, weights_only=False)
        return cls(**data)
