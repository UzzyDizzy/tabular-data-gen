"""SAE quality metrics: reconstruction, sparsity, frontier, cross-seed stability."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import BaseSAE


@torch.no_grad()
def _collect(sae: BaseSAE, loader: DataLoader, device: str):
    dev = torch.device(device)
    sae.to(dev).eval()
    xs, recons, zs = [], [], []
    for x in loader:
        x = x.to(dev)
        recon, z, _ = sae(x)
        xs.append(x.cpu())
        recons.append(recon.cpu())
        zs.append(z.cpu())
    if not xs:
        return None, None, None
    return torch.cat(xs), torch.cat(recons), torch.cat(zs)


@torch.no_grad()
def reconstruction_metrics(sae: BaseSAE, loader: DataLoader, device: str = "cpu") -> dict:
    x, recon, _ = _collect(sae, loader, device)
    if x is None:
        return {"mse": float("nan"), "fve": float("nan")}
    mse = torch.mean((x - recon) ** 2).item()
    var = torch.mean((x - x.mean(0)) ** 2).item() + 1e-12
    return {"mse": mse, "fve": 1.0 - mse / var}  # fraction of variance explained


@torch.no_grad()
def sparsity_metrics(sae: BaseSAE, loader: DataLoader, device: str = "cpu") -> dict:
    x, _, z = _collect(sae, loader, device)
    if x is None:
        return {"l0": float("nan"), "dead_frac": float("nan"), "ultralow_frac": float("nan")}
    l0 = (z > 0).float().sum(-1).mean().item()
    freq = (z > 0).float().mean(0)  # firing frequency per latent
    dead = (freq == 0).float().mean().item()
    ultralow = (freq < 1e-3).float().mean().item()
    return {"l0": l0, "dead_frac": dead, "ultralow_frac": ultralow}


def frontier(saes: list[BaseSAE], loader: DataLoader, device: str = "cpu"):
    """Return a list of dicts (l0, fve, variant) for a reconstruction-sparsity Pareto plot."""
    rows = []
    for sae in saes:
        r = reconstruction_metrics(sae, loader, device)
        s = sparsity_metrics(sae, loader, device)
        rows.append({"variant": sae.cfg.variant, "d_sae": sae.cfg.d_sae, "l0": s["l0"], "fve": r["fve"]})
    return rows


@torch.no_grad()
def cross_seed_stability(saes: list[BaseSAE]) -> dict:
    """Mean max-cosine matching between decoder dictionaries across seed pairs (in [0,1])."""
    if len(saes) < 2:
        return {"mean_max_cosine": float("nan"), "n_pairs": 0}
    dicts = [torch.nn.functional.normalize(s.W_dec.detach(), dim=1) for s in saes]
    scores = []
    for i in range(len(dicts)):
        for j in range(i + 1, len(dicts)):
            sim = dicts[i] @ dicts[j].T  # [d_sae_i, d_sae_j]
            scores.append(sim.max(dim=1).values.mean().item())
    return {"mean_max_cosine": float(np.mean(scores)), "n_pairs": len(scores)}
