# tabular-data-gen — SAE-TFM

**Sparse dictionaries of a tabular foundation model (TabPFN v2): mechanistic interpretation and interpretable generation.**

This repo implements **Gap A** from `plans/`: train sparse autoencoders (SAEs) on a tabular
foundation model's residual stream, validate the discovered features **causally against SCM
ground truth** (the moat — no other SAE domain has ground-truth concepts), and use the same
features to **steer interpretable tabular generation**.

- Research plans: [`plans/gapA-sae-tfm.md`](plans/gapA-sae-tfm.md) (+ the alternatives B/C/D and the comparison in [`plans/README.md`](plans/README.md))
- Implementation blueprint: [`plans/deep-implementation-guide.md`](plans/deep-implementation-guide.md)
- Literature: 92 referred papers in `referred-papers/` (see `referred-papers/INDEX.md`; re-fetch with `python download_papers.py`)

## Status

- ✅ Full pipeline (extract → SAE → align → **causal necessity/sufficiency** → steered generation) runs and is tested **end-to-end on CPU** via a deterministic **mock backend** that simulates in-context inference primitives.
- ⏳ The **real TabPFN backend** is wired (load → find transformer → hook residual stream → reshape to tokens) but needs (a) a one-time `TABPFN_TOKEN` to download weights and (b) confirmation of TabPFN's internal activation layout (discovery tasks D1–D3 in the guide). Run `TabPFNBackend.describe(ds)` to print activation shapes and confirm.

## Install

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\python -m pip install -e ".[dev]"
# Unix:     ./.venv/bin/python -m pip install -e ".[dev]"
```

## Quickstart (CPU)

```bash
# end-to-end smoke (mock backend) — seconds–minutes on CPU
python scripts/run_all_smoke.py                 # uses configs/smoke.yaml
python scripts/run_all_smoke.py steps=600 d_sae=512   # override any field

# tests (14, ~9s on CPU)
python -m pytest
```

Results (metrics + config) are written to `experiments/<out_dir>/results.json`.

## What the smoke run shows

On the mock backend it reports: SAE reconstruction (FVE), per-concept **alignment AUROC vs
LR-probe/neuron baselines**, **causal necessity & sufficiency** of the concept latent vs
random-latent controls, and **steering selectivity** (on-target vs off-target generation shift).

## CPU-dev → GPU-scale

Everything is config-driven. `configs/smoke.yaml` is tiny and CPU-only; `configs/full.yaml`
scales up (more datasets, `variant: matryoshka`, `d_sae: 4096`, multi-seed, real backend):

```bash
python scripts/run_all_smoke.py configs/full.yaml
```

Device is auto-detected (`device: auto` → cuda if present). Activations are cached to disk so
SAE/interp/generation code runs on cached tensors without re-running the model.

## Using the real TabPFN v2

1. Create an account at https://ux.priorlabs.ai, accept the license, copy your API key.
2. `setx TABPFN_TOKEN "<key>"` (Windows) or `export TABPFN_TOKEN=<key>` (Unix).
3. Set `backend: tabpfn` (or `auto`) and run. First call downloads weights.
4. Confirm activation layout: `TabPFNBackend(...).describe(ds)` logs raw shapes; adjust
   `_to_tokens` in `src/tabsae/tabpfn_hooks.py` if the heuristic mislabels the items/feature axes.

## Repository layout

```
src/tabsae/
  types.py            shared dataclasses (SCMDataset, ConceptLabels, ActivationBatch, SAEConfig)
  scm/                SCM data with KNOWN structural roles (the ground truth)
  tabpfn_hooks.py     TabPFNBackend (real) + MockTFMBackend (CPU sim), activation patching
  activations.py      extraction + caching + torch Dataset (split by whole datasets)
  sae/                TopK / JumpReLU / Matryoshka SAEs, training, metrics, save/load
  interp/             concept alignment, baselines (LR/PCA/neuron), causal necessity/sufficiency
  generate/           TFM-as-EBM, SGLD, SAE-latent steering, generation-fidelity metrics
  pipeline.py         end-to-end orchestration (run_all)
scripts/              CLI entry points (run_all_smoke.py, _common.py)
configs/              smoke.yaml (CPU) / full.yaml (GPU)
tests/                14 CPU tests (scm, backend, sae, io, end-to-end make-or-break)
plans/                research plans + deep implementation guide
referred-papers/      92 referred arXiv PDFs (+ INDEX.md)
```
