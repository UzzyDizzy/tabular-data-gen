"""Stochastic Gradient Langevin Dynamics sampler for EnergyModel."""
from __future__ import annotations

import numpy as np

from .energy import EnergyModel


def sgld_sample(grad_fn, x0: np.ndarray, steps: int, eps: float, noise_scale: float,
                rng: np.random.Generator) -> np.ndarray:
    """x <- x - 0.5*eps*grad E(x) + noise_scale*sqrt(eps)*eta."""
    x = np.asarray(x0, dtype=np.float64).copy()
    for _ in range(steps):
        g = grad_fn(x)
        x = x - 0.5 * eps * g + noise_scale * np.sqrt(eps) * rng.normal(size=x.shape)
    return x


def generate(
    energy: EnergyModel,
    n: int,
    n_cols: int,
    target_class: int,
    rng: np.random.Generator,
    steps: int = 200,
    eps: float = 0.1,
    noise_scale: float = 1.0,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Sample `n` synthetic rows for `target_class` via SGLD."""
    if x0 is None:
        x0 = rng.normal(0, 1, size=(n, n_cols))
    return sgld_sample(lambda x: energy.grad_energy(x, target_class), x0, steps, eps, noise_scale, rng)
