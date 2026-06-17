"""Causal validation: necessity & sufficiency via activation patching.

This is what makes alignment causal (not correlational) — and is only possible because
we have SCM ground truth. We patch the backend's residual stream (in SAE/token space) and
measure the effect on the model's predictive distribution.
"""
from __future__ import annotations

import numpy as np
import torch

from ..scm.concepts import column_concept_matrix
from ..types import CONCEPTS, SCMDataset
from ..sae.models import BaseSAE


# -- patch builders -------------------------------------------------------------------
def make_sae_patch(sae: BaseSAE, latents: list[int], mean: np.ndarray, std: np.ndarray,
                   mode: str = "ablate", value: float = 0.0):
    """Return a PatchFn (numpy acts -> numpy acts) that ablates/clamps SAE latents.

    The backend's activations are raw; the SAE lives in normalized space, so we
    normalize -> intervene -> denormalize.
    """
    m = torch.tensor(mean, dtype=torch.float32)
    s = torch.tensor(std, dtype=torch.float32)

    def fn(a: np.ndarray) -> np.ndarray:
        at = torch.tensor(np.asarray(a), dtype=torch.float32)
        an = (at - m) / s
        if mode == "ablate":
            a2 = sae.ablate(an, latents)
        else:  # clamp one latent
            a2 = sae.clamp(an, latents[0], value)
        return (a2 * s + m).detach().numpy().astype(np.float32)

    return fn


def make_direction_patch(direction: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Ablate the component along a unit direction (baseline: LR/PCA direction removal)."""
    d = torch.tensor(direction, dtype=torch.float32)
    d = d / (d.norm() + 1e-8)
    m = torch.tensor(mean, dtype=torch.float32)
    s = torch.tensor(std, dtype=torch.float32)

    def fn(a: np.ndarray) -> np.ndarray:
        an = (torch.tensor(np.asarray(a), dtype=torch.float32) - m) / s
        comp = (an @ d).unsqueeze(-1) * d
        return ((an - comp) * s + m).detach().numpy().astype(np.float32)

    return fn


# -- helpers --------------------------------------------------------------------------
def datasets_with_concept(datasets: list[SCMDataset], concept: str) -> list[SCMDataset]:
    ci = CONCEPTS.index(concept)
    out = []
    for ds in datasets:
        if column_concept_matrix(ds.concept_labels)[:, ci].any():
            out.append(ds)
    return out


def _bernoulli_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return float(np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))))


def _accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs > 0.5).astype(int) == y.astype(int)))


def _query_y(ds: SCMDataset) -> np.ndarray:
    n_ctx = int(ds.meta.get("n_context", ds.n_rows // 2))
    return ds.y[n_ctx:]


# -- experiments ----------------------------------------------------------------------
def _random_controls(rng, d_sae: int, exclude: int, n: int) -> list[int]:
    pool = [i for i in range(d_sae) if i != exclude]
    n = min(n, len(pool))
    return [int(i) for i in rng.choice(pool, size=n, replace=False)]


def measure_necessity(
    backend,
    sae: BaseSAE,
    latent: int,
    datasets: list[SCMDataset],
    mean: np.ndarray,
    std: np.ndarray,
    layer: int = 0,
    n_controls: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Ablate `latent`; measure predictive degradation vs the MEAN over several random
    control latents (a fair, low-variance baseline).

    Necessity holds if ablating the concept latent degrades accuracy / shifts the
    predictive distribution far MORE than ablating random latents.
    """
    rng = rng or np.random.default_rng(0)
    controls = _random_controls(rng, sae.cfg.d_sae, latent, n_controls)
    patch = make_sae_patch(sae, [latent], mean, std, mode="ablate")
    ctrl_patches = [make_sae_patch(sae, [c], mean, std, mode="ablate") for c in controls]
    dacc, kls, dacc_c, kls_c = [], [], [], []
    for ds in datasets:
        base = backend.run(ds, patches=None)[0]
        abl = backend.run(ds, patches={layer: patch})[0]
        y = _query_y(ds)
        dacc.append(_accuracy(base, y) - _accuracy(abl, y))
        kls.append(_bernoulli_kl(base, abl))
        for cp in ctrl_patches:
            con = backend.run(ds, patches={layer: cp})[0]
            dacc_c.append(_accuracy(base, y) - _accuracy(con, y))
            kls_c.append(_bernoulli_kl(base, con))
    return {
        "latent": latent,
        "control_latents": controls,
        "delta_acc": float(np.mean(dacc)),
        "kl": float(np.mean(kls)),
        "delta_acc_control": float(np.mean(dacc_c)),
        "kl_control": float(np.mean(kls_c)),
        "n_datasets": len(datasets),
    }


def measure_sufficiency(
    backend,
    sae: BaseSAE,
    latent: int,
    datasets: list[SCMDataset],
    mean: np.ndarray,
    std: np.ndarray,
    value: float,
    layer: int = 0,
    n_controls: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Inject (clamp) `latent` on datasets LACKING the concept; measure behavior change
    vs clamping several random latents to the same value (mean control)."""
    rng = rng or np.random.default_rng(1)
    controls = _random_controls(rng, sae.cfg.d_sae, latent, n_controls)
    patch = make_sae_patch(sae, [latent], mean, std, mode="clamp", value=value)
    ctrl_patches = [make_sae_patch(sae, [c], mean, std, mode="clamp", value=value) for c in controls]
    eff, eff_c = [], []
    for ds in datasets:
        base = backend.run(ds, patches=None)[0]
        inj = backend.run(ds, patches={layer: patch})[0]
        eff.append(float(np.mean(np.abs(inj - base))))
        for cp in ctrl_patches:
            con = backend.run(ds, patches={layer: cp})[0]
            eff_c.append(float(np.mean(np.abs(con - base))))
    return {
        "latent": latent,
        "control_latents": controls,
        "effect": float(np.mean(eff)),
        "effect_control": float(np.mean(eff_c)),
        "value": value,
        "n_datasets": len(datasets),
    }
