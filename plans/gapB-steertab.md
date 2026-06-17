# Gap B — SteerTab: An Interpretable SAE Control Layer for Tabular Generative Models

## Context

**Why this work.** Tabular generation is now dominated by latent/diffusion models — **TabSyn** (ICLR 2024, arXiv:2310.09656) and **TabDiff** (ICLR 2025, arXiv:2410.20626) are SOTA on the standard benchmark (Shape, Trend, α-precision/β-recall, C2ST, MLE, DCR). But their control is coarse: classifier-free guidance conditions on a target column, not on *interpretable, disentangled* factors, and no work decomposes a tabular generator's latent into human-meaningful directions. In images/LLMs, SAEs and SAE-targeted steering (SAE-TS, arXiv:2411.02193) and post-hoc concept bottlenecks (CB-AE/CC, CVPR 2025, arXiv:2503.19377) deliver exactly this — but **never for tabular generators** (survey confirms).

**The problem we solve.** Practitioners need to *control* synthetic data — set attributes, enforce fairness, suppress memorized/leaky directions — without retraining the generator, and to *understand* what factors the generator encodes. No interpretable, plug-in control layer exists for tabular generation.

**Outcome.** A general **SAE control layer** that bolts onto a frozen SOTA tabular generator, exposing disentangled, steerable, human-interpretable generative factors — the "general architecture/pipeline" framing, with a strong standard-benchmark story.

## Contributions (claims)

- **C1:** first dictionary-learning decomposition of a tabular **generative** model's latent → disentangled, interpretable generative factors (vs PCA/GANSpace baselines and raw latent dims).
- **C2:** a **plug-in SAE control layer** for attribute steering + constraint/fairness control with **no generator retraining**; on-target controllability vs off-target coherence quantified (SAE-TS-style).
- **C3:** **privacy via feature ablation** — suppress memorization-correlated directions to cut MIA risk (Synth-MIA, arXiv:2509.18014) at bounded utility cost.

## Key design decisions

- **Host: TabSyn** (frozen after training) as the SOTA, latent-based, single-continuous-space generator — its VAE latent is the natural SAE substrate. TabDiff as a secondary host (no VAE; intervene in the diffusion bottleneck instead).
- **Intervention points (compared):** (a) on the VAE latent `z` before decoding (simplest, fast); (b) on the diffusion-bottleneck activation during reverse sampling (Asyrp/h-space-style, arXiv:2210.10960).
- **SAE variants:** TopK / JumpReLU / Matryoshka (Matryoshka for hierarchical attribute factors).

## Architecture

```
Train TabSyn (VAE + latent diffusion) ─► freeze ─► encode tables to latent z (cache)
                                                          │
                                                          ▼
                                  SAE (TopK / JumpReLU / Matryoshka) on z
                                                          │
   generation:  noise ─► latent diffusion ─► z ─► [SAE encode → CLAMP/scale latents → SAE decode] ─► VAE decode ─► row
                                                          │
                       interpret factors        steer attributes / enforce fairness / ablate memorized dirs
```

## Repository structure

Reuses the shared `src/tabsae/` core (SAE, metrics, baselines, gen-eval). Adds:

```
src/tabsae/
  hosts/tabsyn.py     # train/load TabSyn VAE+diffusion; encode/decode; hook the bottleneck
  hosts/tabdiff.py    # (secondary) TabDiff bottleneck hooks
  sae/...             # shared TopK/JumpReLU/Matryoshka
  interp/factor_id.py # discover & name disentangled generative factors; PCA/GANSpace baselines
  interp/disentangle.py # disentanglement diagnostics (acknowledging Locatello et al. 2019 limits)
  generate/steer.py   # clamp/scale SAE latents inside the sampling loop (intervention points a/b)
  generate/fairness.py# demographic-parity / counterfactual steering
  generate/privacy.py # ablate memorization-correlated directions; Synth-MIA harness
  generate/eval_gen.py# TabSyn suite + controllability + fairness + privacy
configs/  scripts/  tests/
```

**Reuse:** TabSyn official repo (training + eval protocol), `dictionary_learning`/`SAELens`, `sdmetrics`/`synthcity`, Synth-MIA.

## Implementation phases

- **P0 Bootstrap:** env; train/load TabSyn on one small dataset; verify encode/decode + bottleneck hooks. Exit: round-trip reconstruction sane; CPU smoke on a subset.
- **P1 Latent extraction + caching:** encode all training rows to latent; cache.
- **P2 SAE on latent + baselines:** train TopK/JumpReLU/Matryoshka; reconstruction–sparsity; PCA/GANSpace/raw-dim baselines; cross-seed stability.
- **P3 C1 factor discovery:** identify which latents correspond to interpretable attributes (correlate with held-out column values / known sensitive attrs); compare disentanglement vs baselines.
- **P4 C2 steering:** clamp a factor during sampling; measure on-target controllability vs off-target distortion (TabSyn-suite fidelity must hold); vs classifier-free-guidance and InterFaceGAN-style supervised-direction baselines.
- **P5 C3 fairness + privacy:** demographic-parity steering; ablate memorization directions → re-measure Synth-MIA + utility.
- **P6 Full sweeps (GPU):** TabSyn 6-suite; all hosts/intervention points; ablations.
- **P7 Writeup & release.**

## Datasets & evaluation

- **TabSyn 6-suite:** Adult, Default, Shoppers, Magic, Beijing, News (the de-facto standard).
- **Fidelity:** Shape, Trend, α-precision/β-recall, C2ST, MLE, DCR (must be preserved under steering).
- **Controllability:** on-target attribute change vs off-target column drift.
- **Fairness:** demographic-parity gap shift on Adult/Default.
- **Privacy:** Synth-MIA AUROC + DCR-share (mostlyai-qa, arXiv:2504.01908) before/after direction ablation.
- **Baselines:** classifier-free guidance (TabDiff), PCA/GANSpace directions, InterFaceGAN-style supervised directions, CB-AE concept bottleneck, raw latent-dim steering.

## CPU-dev → GPU-scale

TabSyn training is GPU-bound but moderate; develop on a **small dataset subset / few epochs** locally (CPU smoke), then full training + sweeps on the GPU server. SAE/steer/eval code runs on **cached latents** on CPU. Hydra `smoke.yaml` / `full.yaml`; device auto-detected.

> ⚠️ Heavier upfront compute than Gap A: you must **train the generator** (Gap A uses a frozen pretrained TabPFN). Budget GPU time for TabSyn training across 6 datasets.

## Verification

1. `pytest tests/`: TabSyn round-trip, SAE > PCA at matched L0, steering hook changes output.
2. End-to-end CPU smoke on a subset: train tiny TabSyn → SAE → steer one factor → fidelity preserved.
3. **Make-or-break:** a discovered latent maps to a known attribute (e.g., `sex`/`income` on Adult); clamping it shifts that attribute's marginal predictably while other columns barely move, beating the raw-dim and PCA baselines.
4. Privacy: ablating top memorization-correlated latents reduces Synth-MIA AUROC with small TSTR loss.
5. Repro: seeds, configs, multi-seed stability.

## Risks & mitigations

- **TabSyn latent may not host clean disentangled directions** (Locatello et al. 2019 impossibility) → use weak supervision (attribute labels) to find directions; GANSpace/PCA as honest baselines; report disentanglement limits.
- Steering brittleness (Tan et al. 2024) → quantify reliability across inputs; prefer SAE-targeted (SAE-TS) over raw directions.
- Lower scientific "moat" than A (no ground truth) → compensate with strong **standard-benchmark SOTA control** results + the fairness/privacy capabilities.
- Compute (must train generator) → small-subset CPU dev; GPU for full training.

## Venue & timeline (date 2026-06-14)

- **Primary: ICLR 2027 / NeurIPS 2027 main track** — controllable generation + interpretability is a strong fit.
- Strongest if C2 (controllability) and C3 (privacy/fairness) both land on the standard benchmark.
