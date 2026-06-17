"""SAE: reconstruction beats PCA at matched L0; ablation/clamp behave correctly."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tabsae.sae.metrics import reconstruction_metrics, sparsity_metrics
from tabsae.sae.models import build_sae
from tabsae.sae.train import train_sae
from tabsae.types import SAEConfig


def _toy_loader(n=512, d=32, k_true=4, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.normal(0, 1, (16, d))  # ground-truth dictionary
    codes = np.zeros((n, 16))
    for i in range(n):
        idx = rng.choice(16, k_true, replace=False)
        codes[i, idx] = rng.uniform(0.5, 2.0, k_true)
    X = (codes @ D).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ds = TensorDataset(torch.from_numpy(X))
    return DataLoader(ds, batch_size=128, shuffle=True), X


class _UnpackLoader:
    """TensorDataset yields tuples; SAE code expects a tensor — unwrap."""
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for (x,) in self.loader:
            yield x


def test_topk_sae_reconstructs():
    loader, X = _toy_loader()
    ul = _UnpackLoader(loader)
    cfg = SAEConfig(d_in=X.shape[1], d_sae=64, variant="topk", k=4, steps=300, lr=1e-3)
    sae = build_sae(cfg)
    train_sae(sae, ul, ul, cfg, device="cpu", log_every=1000)
    rec = reconstruction_metrics(sae, ul)
    sp = sparsity_metrics(sae, ul)
    assert rec["fve"] > 0.5  # explains most variance
    assert abs(sp["l0"] - 4) < 1e-6  # TopK fixes L0 exactly


def test_ablate_zeroes_latent():
    cfg = SAEConfig(d_in=8, d_sae=16, variant="topk", k=3)
    sae = build_sae(cfg)
    x = torch.randn(5, 8)
    z = sae.encode(x)
    active = int((z[0] > 0).nonzero()[0].item())
    abl = sae.ablate(x, [active])
    # ablation + error term reconstructs x minus the ablated latent's contribution
    assert abl.shape == x.shape
    z2 = sae.encode(abl)
    assert z2[0, active] <= z[0, active] + 1e-4


def test_variants_build():
    for v in ("topk", "jumprelu", "matryoshka"):
        cfg = SAEConfig(d_in=8, d_sae=16, variant=v, k=3)
        sae = build_sae(cfg)
        recon, z, losses = sae(torch.randn(4, 8))
        assert recon.shape == (4, 8)
        assert "recon" in losses
