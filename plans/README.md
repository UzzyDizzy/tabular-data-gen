# Research plans — Tabular data generation × SAEs / mechanistic interpretability

Four candidate A\*-targeted research directions, all at the intersection of **tabular foundation models / tabular generation** and **sparse autoencoders (SAEs) / mechanistic interpretability**. Each has its own full plan; pick one to implement.

Grounding: a 114-paper literature survey (workflow `wp015u30q`). The organizing finding:

> **No published work applies SAEs / dictionary learning to a tabular *foundation* model (TabPFN v2 / TabICL / TabDPT) or to a tabular *generative* model.** The only SAE-on-tabular paper, **XNNTab** (arXiv:2512.13442, Dec 2025), decomposes a per-dataset MLP. TabPFN interpretability so far is **purely correlational probing** (arXiv:2601.08181; arXiv:2502.17361) — no sparse features, no causal validation, no steering.

## The four plans

| # | Plan | One-liner | Host model | Mech-interp role | Novelty moat | Main risk | Upfront compute |
|---|------|-----------|-----------|------------------|--------------|-----------|-----------------|
| **A** | [SAE-TFM](./gapA-sae-tfm.md) ⭐ | Interpret **TabPFN v2** with SAEs (SCM ground-truth + causal patching) **and** steer TabPFN-as-generator | TabPFN v2 (frozen, pretrained) | **Central** (scientific object) | **Ground-truth causal validation** — the thing the whole SAE field can't do | TabPFN "no fixed feature semantics across datasets" | **Low** (frozen pretrained model; SAEs are cheap) |
| **B** | [SteerTab](./gapB-steertab.md) | SAE-as-control-layer on a frozen SOTA **generator (TabSyn)**: disentangled control + fairness/privacy editing | TabSyn (must train) | Control tool | Strongest **standard-benchmark SOTA** generation + control story | Latent may lack clean disentangled directions | **Medium** (must train TabSyn VAE+diffusion) |
| **C** | [MemTrace](./gapC-memtrace.md) | **Mechanistic privacy**: locate & surgically ablate memorization directions in tabular diffusion | TabDDPM/TabSyn (train) | Central (privacy mechanism) | Mechanistic, auditable privacy beating DCR/MIA | Feature→specific-record linkage hardest to land | **Medium** |
| **D** | [ClinSteer](./gapD-clinsteer.md) | Auditable, concept-steerable synthetic **EHR** with per-record provenance (FDA RWD/RWE) | EHR generator (train) | Central (clinical concepts) | Regulatory-grade auditability + real impact | Needs credentialed MIMIC/eICU access | **Medium** + data-access delay |

⭐ = recommended (matches "mech-interp central + general architecture + confident novelty"); chosen as the working default in the harness plan file. **B** is the safer, more SOTA-benchmark-driven alternative; **A** and **B** compose naturally.

## Shared infrastructure (any plan reuses this)

All four share a `src/tabsae/` core: SAE implementations (TopK / JumpReLU / Matryoshka, wrapping `dictionary_learning` or `SAELens`), activation extraction + disk caching, interpretability metrics (concept alignment, causal ablation/patching, the LR-probe / PCA / neuron baselines), and generation-eval metrics (`sdmetrics` / `synthcity`, TabSyn-suite protocol; `Synth-MIA` for privacy). They differ in the **host model** and the **headline claim**.

## Compute reality (CPU-local dev → GPU-server scale)

- Local machine is **CPU-only**; real runs happen later on a **single-GPU-class server**.
- All plans are **Hydra config-driven** with a `smoke.yaml` (tiny) that runs end-to-end on **CPU**, and a `full.yaml` for the GPU server. Device is auto-detected; activations are cached to disk so SAE/interp/gen code is debuggable locally on tiny cached tensors.
- **A is the lightest** to develop: TabPFN v2 is small and runs on CPU for small tables, and we never train it. **B/C/D require training a generator** (more GPU-bound upfront).

## How to choose

- Want the strongest **science / interpretability** story with a built-in defense against the field's #1 criticism, and the lightest compute → **A**.
- Want the strongest **generation + controllability SOTA** story on standard benchmarks → **B**.
- Most excited by **privacy / safety** with high real-world stakes → **C**.
- Want **applied healthcare impact** and already have MIMIC access → **D**.

_Tell me which to implement and I'll move that plan into execution._
