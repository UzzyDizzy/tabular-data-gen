"""Save/load SAE checkpoints (state_dict + SAEConfig sidecar)."""
from __future__ import annotations

import json

import torch

from ..types import SAEConfig
from .models import BaseSAE, build_sae


def save_sae(sae: BaseSAE, path: str) -> None:
    torch.save(sae.state_dict(), path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(sae.cfg.__dict__, f)


def load_sae(path: str) -> BaseSAE:
    with open(path + ".json", encoding="utf-8") as f:
        cfg = SAEConfig(**json.load(f))
    sae = build_sae(cfg)
    sae.load_state_dict(torch.load(path, map_location="cpu"))
    sae.eval()
    return sae
