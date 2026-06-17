"""SAE training loop with dead-latent resampling + multi-seed helper."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from ..types import SAEConfig
from ..utils.logging import get_logger
from ..utils.seed import set_global_seed
from .metrics import reconstruction_metrics, sparsity_metrics
from .models import BaseSAE, build_sae

log = get_logger(__name__)


@dataclass
class TrainReport:
    history: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)


@torch.no_grad()
def _resample_dead(sae: BaseSAE, fired: torch.Tensor, x_sample: torch.Tensor) -> int:
    """Reinitialize latents that never fired toward high-residual directions (AuxK spirit)."""
    dead = torch.where(fired == 0)[0]
    if dead.numel() == 0:
        return 0
    recon, _z, _ = sae(x_sample)
    resid = x_sample - recon
    # pick the highest-residual examples as new decoder directions
    norms = resid.norm(dim=1)
    idx = torch.topk(norms, min(dead.numel(), x_sample.shape[0])).indices
    dirs = torch.nn.functional.normalize(resid[idx], dim=1)
    for i, d in enumerate(dead[: dirs.shape[0]]):
        sae.W_dec.data[d] = dirs[i]
        sae.W_enc.data[:, d] = dirs[i] * 0.1
        sae.b_enc.data[d] = 0.0
    return int(dead.numel())


def train_sae(
    sae: BaseSAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: SAEConfig,
    device: str = "cpu",
    log_every: int = 200,
    resample_every: int = 0,
) -> TrainReport:
    dev = torch.device(device)
    sae.to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    report = TrainReport()
    fired = torch.zeros(cfg.d_sae)
    step = 0
    last_batch = None
    while step < cfg.steps:
        for x in train_loader:
            x = x.to(dev)
            last_batch = x
            recon, z, losses = sae(x)
            loss = losses["recon"] + losses["sparsity"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cfg.normalize_decoder:
                sae.normalize_decoder()
            fired += (z > 0).any(0).float().cpu()
            step += 1
            if step % log_every == 0:
                rec = float(losses["recon"].detach())
                report.history.append({"step": step, "recon": rec, "l0": float(losses["l0"])})
                log.info("step %d recon=%.4f l0=%.1f", step, rec, float(losses["l0"]))
            if resample_every and step % resample_every == 0 and last_batch is not None:
                n = _resample_dead(sae, fired, last_batch)
                if n:
                    log.info("resampled %d dead latents", n)
                fired.zero_()
            if step >= cfg.steps:
                break
    sae.eval()
    report.final = {**reconstruction_metrics(sae, val_loader, device), **sparsity_metrics(sae, val_loader, device)}
    log.info("SAE final: %s", report.final)
    return report


def train_sae_multi_seed(
    cfg: SAEConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    seeds: list[int],
    device: str = "cpu",
) -> list[BaseSAE]:
    """Train N seeds for cross-seed stability analysis (known SAE weakness)."""
    saes = []
    for s in seeds:
        set_global_seed(s)
        c = SAEConfig(**{**cfg.__dict__, "seed": s})
        sae = build_sae(c)
        train_sae(sae, train_loader, val_loader, c, device=device)
        saes.append(sae)
    return saes
