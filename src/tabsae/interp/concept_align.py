"""Quantify which SAE latents correspond to which ground-truth concept."""
from __future__ import annotations

import numpy as np
import torch

from ..sae.models import BaseSAE
from ..types import CONCEPTS
from .baselines import auroc_score, lr_probe_concept, neuron_selectivity


def latent_activations(sae: BaseSAE, norm_acts: np.ndarray) -> np.ndarray:
    """Encode normalized activations -> [N_tokens, d_sae] latent activations."""
    sae.eval()
    with torch.no_grad():
        z = sae.encode(torch.tensor(norm_acts, dtype=torch.float32))
    return z.numpy()


def align_latents_to_concepts(
    latent_acts: np.ndarray,
    label_matrix: np.ndarray,
    concepts: list[str] = CONCEPTS,
    raw_acts: np.ndarray | None = None,
) -> list[dict]:
    """For each concept, find the most selective latent (sign-agnostic AUROC), and
    optionally compare to LR-probe / best-neuron baselines on raw activations."""
    rows = []
    for ci, concept in enumerate(concepts):
        y = label_matrix[:, ci].astype(int)
        n_pos = int(y.sum())
        if n_pos < 2 or n_pos > len(y) - 2:
            rows.append({"concept": concept, "status": "skipped", "n_pos": n_pos})
            continue
        aurocs = np.array([auroc_score(latent_acts[:, li], y) for li in range(latent_acts.shape[1])])
        aurocs = np.nan_to_num(aurocs, nan=0.0)
        best_latent = int(np.argmax(aurocs))
        row = {
            "concept": concept,
            "best_latent": best_latent,
            "sae_auroc": float(aurocs[best_latent]),
            "n_pos": n_pos,
        }
        if raw_acts is not None:
            row["lr_probe_auroc"] = lr_probe_concept(raw_acts, y)
            row["best_neuron_auroc"] = neuron_selectivity(raw_acts, y)["auroc"]
        rows.append(row)
    return rows


def purity_and_coverage(
    latent_acts: np.ndarray,
    label_matrix: np.ndarray,
    alignment: list[dict],
    threshold: float = 0.7,
) -> dict:
    """Coverage = fraction of concepts cleanly captured (best AUROC >= threshold).
    Purity = mean gap between a best-latent's AUROC for its own concept vs its best
    AUROC for any OTHER concept (high gap => monosemantic)."""
    scored = [r for r in alignment if "sae_auroc" in r]
    if not scored:
        return {"coverage": 0.0, "purity": float("nan")}
    coverage = float(np.mean([r["sae_auroc"] >= threshold for r in scored]))
    gaps = []
    for r in scored:
        li = r["best_latent"]
        own = r["sae_auroc"]
        others = []
        for ci, concept in enumerate(CONCEPTS):
            if concept == r["concept"]:
                continue
            y = label_matrix[:, ci].astype(int)
            if y.sum() < 2 or y.sum() > len(y) - 2:
                continue
            others.append(auroc_score(latent_acts[:, li], y))
        if others:
            gaps.append(own - max(np.nan_to_num(others, nan=0.0)))
    return {"coverage": coverage, "purity": float(np.mean(gaps)) if gaps else float("nan")}
