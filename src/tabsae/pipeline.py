"""End-to-end pipeline: extract -> train SAE -> interpret (align+causal) -> generate.

Stages are deterministic from (cfg, seed): the corpus and backend are regenerated from
the seed, while activation shards are cached to disk. ``run_all`` chains everything and
returns a results dict; CLI scripts and tests both call into here.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .activations import ActivationDataset, extract_corpus_activations, make_loaders
from .generate.energy import MockEnergyModel
from .generate.eval_gen import fidelity_suite
from .generate.sgld import generate
from .generate.steer import controllability_report, make_steered_energy
from .interp.causal import datasets_with_concept, measure_necessity, measure_sufficiency
from .interp.concept_align import align_latents_to_concepts, latent_activations, purity_and_coverage
from .sae.metrics import cross_seed_stability
from .sae.models import build_sae
from .sae.train import train_sae, train_sae_multi_seed
from .scm.generators import generate_corpus, make_controlled_dataset
from .tabpfn_hooks import load_backend
from .types import SAEConfig
from .utils.io import Manifest
from .utils.logging import get_logger, log_config
from .utils.seed import set_global_seed

log = get_logger(__name__)


@dataclass
class PipelineConfig:
    # general
    backend: str = "mock"  # 'mock' | 'tabpfn' | 'auto'
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "experiments/smoke"
    # backend (mock)
    d_model: int = 64
    # corpus
    n_datasets: int = 40
    n_cols: int = 8
    n_context: int = 128
    n_query: int = 64
    shift_frac: float = 0.2
    positive_effects: bool = True  # clean monotone sign for the mock/smoke; set False for realistic runs
    block_size: int = 1  # >1 assigns one role per contiguous block (use = features_per_group on TabPFN)
    # activations
    token_kind: str = "column"
    layer: int = 0
    normalize: str = "global"
    # sae
    variant: str = "topk"
    d_sae: int = 256
    k: int = 16
    lr: float = 3e-4
    steps: int = 400
    batch_size: int = 512
    seeds: tuple[int, ...] = (0,)  # >1 enables cross-seed stability
    # interp
    concept: str = "monotone"  # headline concept for causal validation
    # generate
    do_generate: bool = True
    gen_n: int = 256
    sgld_steps: int = 150
    sgld_eps: float = 0.1
    sgld_noise: float = 1.0
    steer_value: float = 4.0


# -- deterministic builders -----------------------------------------------------------
def build_corpus(cfg: PipelineConfig):
    rng = np.random.default_rng(cfg.seed)
    return generate_corpus(
        cfg.n_datasets, rng, n_cols=cfg.n_cols, n_context=cfg.n_context, n_query=cfg.n_query,
        shift_frac=cfg.shift_frac, positive_effects=cfg.positive_effects, block_size=cfg.block_size,
    )


def get_backend(cfg: PipelineConfig):
    return load_backend(cfg.backend, d_model=cfg.d_model, seed=cfg.seed, device=cfg.device)


# -- stages ---------------------------------------------------------------------------
def run_extract(cfg: PipelineConfig, datasets=None, backend=None) -> Manifest:
    datasets = datasets if datasets is not None else build_corpus(cfg)
    backend = backend if backend is not None else get_backend(cfg)
    acts_dir = os.path.join(cfg.out_dir, "acts")
    return extract_corpus_activations(backend, datasets, acts_dir, token_kind=cfg.token_kind)


def run_train(cfg: PipelineConfig, manifest: Manifest):
    train_loader, val_loader, full = make_loaders(
        manifest, cfg.layer, batch_size=cfg.batch_size, normalize=cfg.normalize, seed=cfg.seed
    )
    sae_cfg = SAEConfig(d_in=full.d_in, d_sae=cfg.d_sae, variant=cfg.variant, k=cfg.k,
                        lr=cfg.lr, steps=cfg.steps, batch_size=cfg.batch_size, seed=cfg.seed)
    set_global_seed(cfg.seed)
    sae = build_sae(sae_cfg)
    report = train_sae(sae, train_loader, val_loader, sae_cfg, device=cfg.device)
    stability = {}
    if len(cfg.seeds) > 1:
        saes = train_sae_multi_seed(sae_cfg, train_loader, val_loader, list(cfg.seeds), device=cfg.device)
        stability = cross_seed_stability(saes)
    return sae, full, report, stability


def run_interp(cfg: PipelineConfig, sae, full: ActivationDataset, backend, datasets) -> dict:
    z = latent_activations(sae, full.norm_acts)
    labels = full.concept_label_matrix()
    alignment = align_latents_to_concepts(z, labels, concepts=full.concepts, raw_acts=full.acts)
    pc = purity_and_coverage(z, labels, alignment)

    # causal validation on the headline concept
    causal = {}
    row = next((r for r in alignment if r.get("concept") == cfg.concept and "best_latent" in r), None)
    if row is not None:
        latent = row["best_latent"]
        present = datasets_with_concept(datasets, cfg.concept)
        present_ids = {d.dataset_id for d in present}
        absent = [d for d in datasets if d.dataset_id not in present_ids]
        if not absent:
            # synthesize clean concept-ABSENT datasets so sufficiency has somewhere to inject
            arng = np.random.default_rng(cfg.seed + 7)
            absent = [
                make_controlled_dataset("irrelevant", arng, n_cols=cfg.n_cols,
                                        n_context=cfg.n_context, n_query=cfg.n_query,
                                        dataset_id=f"absent-{i}")
                for i in range(5)
            ]
        rng = np.random.default_rng(cfg.seed)
        nec = measure_necessity(backend, sae, latent, present, full._mean, full._std,
                                layer=cfg.layer, rng=rng)
        # injection value = mean activation of the latent on positive tokens
        ci = full.concepts.index(cfg.concept)
        pos = labels[:, ci]
        val = float(z[pos, latent].mean()) if pos.any() else cfg.steer_value
        suf = measure_sufficiency(backend, sae, latent, absent or present, full._mean, full._std,
                                  value=val, layer=cfg.layer, rng=rng) if absent else {}
        causal = {"concept": cfg.concept, "latent": latent, "necessity": nec, "sufficiency": suf}
    return {"alignment": alignment, "purity_coverage": pc, "causal": causal}


def run_generate(cfg: PipelineConfig, sae, full: ActivationDataset, backend, interp: dict) -> dict:
    # clean controlled context with a single monotone column (target_col = 0)
    ctx = make_controlled_dataset("monotone", np.random.default_rng(cfg.seed + 2), n_cols=cfg.n_cols,
                                  n_context=cfg.n_context, n_query=cfg.n_query)
    base_energy = MockEnergyModel.from_backend(backend, ctx)
    # identical seed for base & steered -> same x0 and noise, so only steering differs
    gen_seed = cfg.seed + 1
    base = generate(base_energy, cfg.gen_n, cfg.n_cols, target_class=1,
                    rng=np.random.default_rng(gen_seed),
                    steps=cfg.sgld_steps, eps=cfg.sgld_eps, noise_scale=cfg.sgld_noise)
    fidelity = fidelity_suite(ctx.X, base)

    out = {"fidelity": fidelity}
    latent = (interp.get("causal") or {}).get("latent")
    if latent is not None:
        steered_energy = make_steered_energy(backend, ctx, sae=sae, latent=latent,
                                             value=cfg.steer_value, mean=full._mean, std=full._std,
                                             target_col=0)
        steered = generate(steered_energy, cfg.gen_n, cfg.n_cols, target_class=1,
                           rng=np.random.default_rng(gen_seed),
                           steps=cfg.sgld_steps, eps=cfg.sgld_eps, noise_scale=cfg.sgld_noise)
        out["controllability"] = controllability_report(base, steered, target_col=0)
    return out


def run_all(cfg: PipelineConfig) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_config(asdict(cfg), cfg.out_dir)
    set_global_seed(cfg.seed)
    datasets = build_corpus(cfg)
    backend = get_backend(cfg)
    manifest = run_extract(cfg, datasets=datasets, backend=backend)
    sae, full, report, stability = run_train(cfg, manifest)
    interp = run_interp(cfg, sae, full, backend, datasets)
    results = {
        "config": asdict(cfg),
        "sae_final": report.final,
        "cross_seed_stability": stability,
        "interp": interp,
    }
    if cfg.do_generate:
        results["generate"] = run_generate(cfg, sae, full, backend, interp)
    with open(os.path.join(cfg.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    log.info("Saved results -> %s", os.path.join(cfg.out_dir, "results.json"))
    return results
