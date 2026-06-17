# Gap A — SAE-TFM: Sparse Dictionaries of a Tabular Foundation Model — Mechanistic Interpretation and Interpretable Generation ⭐ (recommended)

## Context

**Why this work.** SAEs / dictionary learning are the default mechanistic-interpretability tool and have been ported beyond text LLMs — to vision/CLIP, protein LMs (InterPLM, *Nature Methods* 2025), single-cell genomics FMs, and recommenders. Yet (per the 114-paper survey) **no one has applied SAEs to a tabular foundation model**. The only SAE-on-tabular work, **XNNTab** (arXiv:2512.13442), decomposes a per-dataset MLP — not a foundation model, not generative. TabPFN interpretability is so far **purely correlational probing** (Gupta et al. 2026, arXiv:2601.08181; Ye et al. 2025, arXiv:2502.17361).

**The problems we solve.** (1) We don't know what reusable computational *primitives* a tabular FM uses for in-context Bayesian inference. (2) The whole SAE field validates features only *correlationally* (against annotations), never against ground truth — because text/protein/vision have none. (3) Tabular generation lacks *interpretable, feature-level* control.

**Outcome — an A\* paper with a moat.** TabPFN v2 is meta-trained on synthetic **structural-causal-model (SCM)** datasets, so we can build probe tables with **known ground-truth causal structure** → the **first SAE interpretability study with ground-truth concepts**, enabling **causal** validation that no host domain currently supports.

## Contributions (claims)

- **C1 (headline, load-bearing):** mechanistic decomposition of TabPFN v2 — monosemantic latents that align with *known* SCM primitives (monotone effects, interactions, covariate shift, feature irrelevance/redundancy), validated **causally** (ablation/patching: necessity + sufficiency), benchmarked vs LR-probe / PCA / neuron baselines.
- **C2:** interpretable, steerable generation — TabPFN-as-EBM (TabPFGen/TabEBM-style, SGLD) **steered by clamping SAE latents** → controllable synthetic data, on-target with bounded off-target distortion.
- **C3:** a ground-truth interpretability **benchmark + protocol** for tabular-model SAE features (SCM tables with known concepts) — releasable companion / D&B fallback.

If time is tight, **C1+C3** stand alone; **C2** is upside.

## Key design decisions

- **Open-weights TabPFN v2** (`tabpfn` package) — open weights are mandatory for reproducible interpretability; v2.5/TabICL/TabDPT are stretch comparisons.
- **No-fixed-semantics handling:** train SAEs across many SCM datasets with per-dataset activation normalization; define concepts as **structural roles**, not column identities; report per-dataset SAEs as a control.
- **Hook points (compared):** (a) per-query-row residual stream into the final layers; (b) per-column token embeddings; (c) per-cell activations. Start with (a)+(b).
- **SAE variants:** TopK (clean L0), JumpReLU (SOTA reconstruction), **Matryoshka** (fights absorption of hierarchical tabular concepts).

## Architecture

```
Real + SCM tables ─► TabPFN v2 (frozen) ─► forward hooks ─► residual-stream activations (cached)
                                                                  │
                                                                  ▼
                                          SAE (TopK / JumpReLU / Matryoshka), TabPFN frozen
                                                                  │
        ┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
        ▼                               ▼                              ▼
  C1 concept alignment vs SCM     C1 causal validation          C2 TabPFN-as-EBM generation (SGLD)
  ground truth (+ baselines,      (ablate/clamp latent →        with SAE-latent CLAMPING →
  cross-seed stability)           necessity/sufficiency)        controllable synthetic rows
```

## Repository structure (greenfield)

```
tabular-data-gen/
  pyproject.toml / requirements.txt   # torch, tabpfn, numpy, scikit-learn, sdmetrics/synthcity, hydra-core, einops
  configs/                            # smoke.yaml (CPU) + full.yaml (GPU) + sae/, scm/, gen/
  src/tabsae/
    tabpfn_hooks.py    # load TabPFN v2; register hooks; extract residual stream
    activations.py     # extract → cache (memmap); decouples GPU extraction from SAE training
    scm/generators.py  # SCM sampler with controllable LABELED structure
    scm/concepts.py    # ground-truth concept defs: monotone, interaction, covariate-shift, irrelevant, redundant
    sae/models.py      # TopK, JumpReLU, Matryoshka (wrap dictionary_learning / SAELens)
    sae/train.py       # training loop, AuxK dead-latent revival, seeds
    sae/metrics.py     # reconstruction, L0, dead-fraction, monosemanticity
    interp/concept_align.py  # latent ↔ ground-truth concept (precision/recall/F1)
    interp/causal.py         # ablation + activation patching → necessity & sufficiency
    interp/baselines.py      # LR probe on raw activations, PCA, neuron-level
    interp/autointerp.py     # (optional) feature naming
    generate/energy.py # TabPFN-as-EBM (class-conditional energy from logits)
    generate/sgld.py   # SGLD sampler
    generate/steer.py  # clamp/scale SAE latents during generation
    generate/eval_gen.py # TabSyn-suite metrics + controllability + optional Synth-MIA
    utils/             # device auto-detect, seeding, config logging
  scripts/             # run_extract.py, run_train_sae.py, run_interp.py, run_generate.py
  tests/               # CPU smoke tests per module + one end-to-end smoke
  experiments/  paper/  README.md
```

**Reuse:** `dictionary_learning`/`SAELens` (SAEs), official `tabpfn` (model), `sdmetrics`/`synthcity` (gen metrics), TabSyn repo protocol (eval), Synth-MIA (privacy, optional).

## Implementation phases

- **P0 Bootstrap (CPU):** repo/env, load TabPFN v2, forward pass on a toy table, register hooks, dump shapes. Exit: `pytest tests/test_hooks.py` green.
- **P1 SCM ground truth (CPU):** labeled SCM datasets (monotone / interaction / covariate-shift / irrelevant / redundant). Exit: 1k datasets with verified labels.
- **P2 Activation extraction + caching:** sharded memmap cache; `--smoke` extracts 1 dataset in seconds on CPU.
- **P3 SAEs + baselines:** TopK/JumpReLU/Matryoshka; reconstruction–sparsity frontier, dead-latent %, **cross-seed stability**; LR/PCA/neuron baselines. Exit: SAE beats PCA at matched L0; CPU smoke trains d_sae=256.
- **P4 C1 interpretability (headline):** concept alignment F1 vs baselines; causal ablation/patching (necessity + sufficiency). Exit: ≥1 concept high-F1 **and** significant causal effect baselines can't reproduce.
- **P5 C2 generation:** TabPFN-EBM (validate base fidelity), then SAE-latent clamping; controllability vs coherence (à la SAE-TS). Exit: clamping a concept latent makes a predicted on-target change with bounded off-target distortion.
- **P6 Real-data + full sweeps (GPU):** CC18 subset, Adult, California Housing, TabSyn 6-suite; full width/sparsity/seed sweeps + ablations.
- **P7 Writeup & release:** figures, C3 benchmark, paper.

## Datasets & evaluation

- **SCM-synthetic (ours):** primary for C1/C3 (only data with concept ground truth).
- **Real:** OpenML CC18 subset, Adult, California Housing; TabSyn 6-suite (Adult, Default, Shoppers, Magic, Beijing, News) for generation.
- **Interp metrics:** concept F1; causal necessity/sufficiency effect sizes; monosemanticity; reconstruction–sparsity; dead-latent %; cross-seed stability; **vs LR-probe / PCA / neuron** (non-negotiable).
- **Gen metrics:** TabSyn suite (Shape, Trend, α-precision/β-recall, C2ST, MLE, DCR) + controllability + optional Synth-MIA.

## CPU-dev → GPU-scale

Hydra `smoke.yaml` runs the whole pipeline on **CPU** in minutes; `full.yaml` scales on the GPU server. Device auto-detected. Activation caching decouples TabPFN forward passes from SAE training. README documents the exact scale-up diff.

## Verification

1. `pytest tests/` (CPU): hook shapes, SCM labels, SAE > PCA at matched L0, samplers run.
2. End-to-end CPU smoke: one tiny SCM dataset extract→train→align→ablate→generate; assert planted concept recovered + ablation changes output as predicted.
3. **Make-or-break:** plant a single monotone effect → a latent aligns (high F1) and is causally necessary; baselines do **not** reproduce the causal result.
4. Gen sanity: EBM matches marginals/correlations on a toy; clamping shifts the targeted statistic and little else.
5. Repro: fixed seeds, logged configs, multi-seed stability reported.

## Risks & mitigations

- No fixed semantics → SCM ground truth + per-dataset normalization + structural-role concepts; per-dataset SAEs as control.
- SAE cross-seed instability (known) → Matryoshka + multi-seed + meta-SAE; report honestly.
- "First but not enough" → moat = ground-truth causal validation + a working capability.
- Classifier-energy gen may trail SOTA fidelity → frame C2 as *interpretable/controllable* (benchmarked, not fidelity-SOTA).
- Compute → CPU smoke; GPU only for P6.
- Competition (XNNTab / looking-glass) → we differ by FM (not per-dataset MLP), causal+ground-truth (not correlational), and generation/steering.

## Venue & timeline (date 2026-06-14)

- **Primary: ICLR 2027** (≈ late Sep 2026) on C1+C3, C2 as upside.
- **Fallback: ICML 2027** (≈ late Jan 2027) for the full arc.
- **Companion:** C3 benchmark → NeurIPS D&B.
