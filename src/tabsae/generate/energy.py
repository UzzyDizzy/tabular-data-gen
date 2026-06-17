"""Turn a (frozen) tabular FM into a class-conditional energy model.

E(x) = -log p(y=c | x, context). SGLD then samples synthetic rows from the implied
conditional distribution. Steering modifies the energy via SAE-latent interventions
(see steer.py), so generation becomes interpretable + controllable.

The mock energy uses an analytic gradient (no autograd) so generation is fast on CPU.
A real-TabPFN energy would differentiate class logits w.r.t. x (discovery task D4).
"""
from __future__ import annotations

import numpy as np

from ..tabpfn_hooks import MONO_IDX
from ..types import SCMDataset


class EnergyModel:
    """Interface: per-row logit for class 1, energy, and dE/dx for a target class."""

    def logit(self, X: np.ndarray) -> np.ndarray:  # noqa
        raise NotImplementedError

    def proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    def energy(self, X: np.ndarray, target_class: int) -> np.ndarray:
        p = np.clip(self.proba(X), 1e-6, 1 - 1e-6)
        return -np.log(p) if target_class == 1 else -np.log(1 - p)

    def grad_energy(self, X: np.ndarray, target_class: int) -> np.ndarray:  # noqa
        raise NotImplementedError


class MockEnergyModel(EnergyModel):
    """Linear-readout energy derived from the mock backend's recovered per-column weights."""

    def __init__(self, w: np.ndarray, mu: np.ndarray, sd: np.ndarray):
        self.w = np.asarray(w, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64) + 1e-6

    @classmethod
    def from_backend(cls, backend, context_ds: SCMDataset, col_acts: np.ndarray | None = None) -> "MockEnergyModel":
        if col_acts is None:
            col_acts = backend.column_activations(context_ds)
        w = (backend.W_pinv @ col_acts.T).T[:, MONO_IDX]  # recovered monotone weight per column
        n_ctx = int(context_ds.meta.get("n_context", context_ds.n_rows // 2))
        Xc = context_ds.X[:n_ctx]
        return cls(w, Xc.mean(0), Xc.std(0))

    def logit(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.mu) / self.sd
        return z @ self.w

    def grad_energy(self, X: np.ndarray, target_class: int) -> np.ndarray:
        p = self.proba(X)
        # dE/dlogit = (p-1) for class 1, p for class 0 ; dlogit/dx = w/sd
        dl = (p - 1.0) if target_class == 1 else p
        return dl[:, None] * (self.w / self.sd)[None, :]
