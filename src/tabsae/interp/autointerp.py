"""Lightweight, dependency-free latent naming (optional).

We avoid external LLM APIs here for reproducibility: a latent is described by the
ground-truth concept it best aligns with plus its firing statistics. Swap in an
LLM-based namer later if richer descriptions are needed.
"""
from __future__ import annotations

import numpy as np


def name_latents(alignment: list[dict]) -> dict[str, int]:
    """concept -> best latent (only for concepts that were scored)."""
    return {r["concept"]: r["best_latent"] for r in alignment if "best_latent" in r}


def describe_latent(latent: int, alignment: list[dict], latent_acts: np.ndarray) -> str:
    """Human-readable one-liner for a latent."""
    matches = [r for r in alignment if r.get("best_latent") == latent]
    freq = float((latent_acts[:, latent] > 0).mean()) if latent < latent_acts.shape[1] else float("nan")
    if matches:
        r = matches[0]
        return f"latent#{latent}: ~'{r['concept']}' (AUROC={r['sae_auroc']:.2f}, fires {freq:.1%})"
    return f"latent#{latent}: no clear concept (fires {freq:.1%})"
