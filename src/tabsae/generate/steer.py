"""Steer generation by clamping an SAE latent in the residual stream.

Clamping a concept-aligned latent modifies the energy model (via the backend's column
activations), which changes the distribution of SGLD-sampled rows. Controllability is
measured as on-target change vs off-target distortion (SAE-TS, 2411.02193).
"""
from __future__ import annotations

import numpy as np

from ..interp.causal import make_sae_patch
from ..sae.models import BaseSAE
from ..types import SCMDataset
from .energy import MockEnergyModel


def make_steered_energy(
    backend,
    context_ds: SCMDataset,
    sae: BaseSAE | None = None,
    latent: int | None = None,
    value: float = 0.0,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    target_col: int | None = None,
) -> MockEnergyModel:
    """Build an energy model whose residual stream has `latent` clamped.

    If target_col is given, the clamp is applied ONLY to that column's token, so steering
    is selective (on-target). Otherwise it is applied to all column tokens.
    """
    col_acts = backend.column_activations(context_ds)
    if sae is not None and latent is not None:
        patch = make_sae_patch(sae, [latent], mean, std, mode="clamp", value=value)
        if target_col is None:
            col_acts = patch(col_acts)
        else:
            row = patch(col_acts[target_col : target_col + 1])
            col_acts = col_acts.copy()
            col_acts[target_col] = row[0]
    return MockEnergyModel.from_backend(backend, context_ds, col_acts=col_acts)


def controllability_report(
    base_samples: np.ndarray,
    steered_samples: np.ndarray,
    target_col: int,
) -> dict:
    """On-target = shift in target column's mean; off-target = mean shift in other columns."""
    n_cols = base_samples.shape[1]
    on = float(abs(steered_samples[:, target_col].mean() - base_samples[:, target_col].mean()))
    others = [c for c in range(n_cols) if c != target_col]
    off = float(np.mean([abs(steered_samples[:, c].mean() - base_samples[:, c].mean()) for c in others])) if others else 0.0
    return {
        "target_col": target_col,
        "on_target_shift": on,
        "off_target_shift": off,
        "selectivity": on / (off + 1e-8),
    }
