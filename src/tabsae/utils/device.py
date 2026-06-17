"""Device auto-detect. Never hard-code cuda anywhere else in the codebase."""
from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return a torch.device.

    prefer: 'auto' (cuda>mps>cpu), or an explicit 'cuda'/'cpu'/'mps'/'cuda:0'.
    Why: local dev is CPU-only; the GPU server has cuda. Same code, no edits.
    """
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
