"""SAE variants with a unified API.

Why three variants:
- TopK     : directly fixes L0 (clean sparsity control), no L1 shrinkage. (OpenAI 2406.04093)
- JumpReLU : learnable per-latent threshold; strong reconstruction at fixed sparsity. (2407.14435)
- Matryoshka: nested prefixes each reconstruct -> reduces feature ABSORPTION of hierarchical
              concepts (tabular roles are hierarchical: 'relevant' > 'monotone'). (2503.17547)

All share the standard SAE form with a tied pre-decoder bias:
    z      = activation( (x - b_dec) @ W_enc + b_enc )
    x_hat  = z @ W_dec + b_dec
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import SAEConfig


class BaseSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_in, d_sae = cfg.d_in, cfg.d_sae
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(self.W_enc.detach().clone().T.contiguous())
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if cfg.normalize_decoder:
            self.normalize_decoder()

    # -- core ---------------------------------------------------------------------
    def preact(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def _activate(self, pre: torch.Tensor) -> torch.Tensor:  # noqa
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._activate(self.preact(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z = self.encode(x)
        recon = self.decode(z)
        losses = {
            "recon": F.mse_loss(recon, x),
            "sparsity": self.sparsity_loss(z),
            "l0": (z > 0).float().sum(-1).mean().detach(),
        }
        return recon, z, losses

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=z.device)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norm = self.W_dec.norm(dim=1, keepdim=True) + 1e-8
        self.W_dec.div_(norm)

    def ablate(self, x: torch.Tensor, latents: list[int]) -> torch.Tensor:
        """Reconstruction with `latents` forced to 0, PLUS the SAE error term, so all
        non-targeted computation is preserved. This is the causal-ablation primitive."""
        z = self.encode(x)
        full = self.decode(z)
        z2 = z.clone()
        if latents:
            z2[:, latents] = 0.0
        return self.decode(z2) + (x - full)

    def clamp(self, x: torch.Tensor, latent: int, value: float) -> torch.Tensor:
        """Set one latent to `value` (steering), preserving the error term."""
        z = self.encode(x)
        full = self.decode(z)
        z2 = z.clone()
        z2[:, latent] = value
        return self.decode(z2) + (x - full)


class TopKSAE(BaseSAE):
    def _activate(self, pre: torch.Tensor) -> torch.Tensor:
        k = self.cfg.k or pre.shape[-1]
        relu = F.relu(pre)
        if k >= pre.shape[-1]:
            return relu
        topv, topi = relu.topk(k, dim=-1)
        return torch.zeros_like(relu).scatter_(-1, topi, topv)


class JumpReLUSAE(BaseSAE):
    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.log_theta = nn.Parameter(torch.full((cfg.d_sae,), -2.0))

    def _activate(self, pre: torch.Tensor) -> torch.Tensor:
        z = F.relu(pre)
        theta = F.softplus(self.log_theta)
        mask = (z > theta).float()
        # straight-through: hard gate forward, gradient flows through z
        return z * mask + (z * 0.0)  # mask has no grad; z keeps grad where mask=1

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        coeff = self.cfg.l1 if self.cfg.l1 is not None else 1e-3
        return coeff * (z > 0).float().sum(-1).mean()


class MatryoshkaSAE(TopKSAE):
    """TopK selection, but each nested prefix of latents must reconstruct on its own."""

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.sizes = cfg.matryoshka_sizes or [
            max(1, cfg.d_sae // 4),
            max(1, cfg.d_sae // 2),
            cfg.d_sae,
        ]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z = self.encode(x)
        recon_full = self.decode(z)
        recon_loss = torch.zeros((), device=x.device)
        for s in self.sizes:
            recon_s = z[:, :s] @ self.W_dec[:s] + self.b_dec
            recon_loss = recon_loss + F.mse_loss(recon_s, x)
        recon_loss = recon_loss / len(self.sizes)
        losses = {
            "recon": recon_loss,
            "sparsity": self.sparsity_loss(z),
            "l0": (z > 0).float().sum(-1).mean().detach(),
        }
        return recon_full, z, losses


def build_sae(cfg: SAEConfig) -> BaseSAE:
    variant = cfg.variant.lower()
    if variant == "topk":
        return TopKSAE(cfg)
    if variant == "jumprelu":
        return JumpReLUSAE(cfg)
    if variant == "matryoshka":
        return MatryoshkaSAE(cfg)
    raise ValueError(f"unknown SAE variant {cfg.variant!r}")
