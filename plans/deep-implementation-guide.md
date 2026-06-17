# Deep Implementation Guide — Gap A: SAE-TFM

> Blueprint for implementing **SAE-TFM** (interpret TabPFN v2 with sparse autoencoders, with SCM ground-truth + causal validation, then steer TabPFN-as-generator). This is a *spec*, not code: every module, class, and function is described at the level of signatures + "what / why / I/O" so that writing the real code later is mechanical. Companion to `plans/gapA-sae-tfm.md`.

---

## 0. Guiding principles (read first)

1. **TabPFN is frozen.** We never train it. We only run forward passes and read/patch its internal activations. This keeps compute tiny (CPU-dev viable) and makes interpretability reproducible.
2. **Activations are cached to disk.** Extraction (TabPFN forward) is decoupled from SAE training/interp/gen. After caching once, all SAE/interp/gen code runs on `.npy`/`.pt` tensors — fully debuggable on CPU.
3. **Everything is config-driven (Hydra).** A `smoke` config runs the *entire* pipeline end-to-end on CPU in minutes; a `full` config scales on the GPU server. No code path hard-codes `cuda`.
4. **Ground truth is the moat.** Because TabPFN's prior is SCM-based, we *generate* data with known structural roles per column. Every interpretability claim is checked against these labels and validated **causally** (ablation/patching), not just correlationally.
5. **Baselines are mandatory.** Every SAE claim is compared to logistic-regression probes on raw activations, PCA, and raw neurons. We are honest where SAEs don't beat baselines on probing; our value-add is causal steerability + monosemanticity.
6. **Determinism.** Global seeding, logged configs, multi-seed runs for any stability claim.

### Discovery tasks (resolve against the real `tabpfn` package before/while coding)
These are the only genuinely uncertain points; each has a fallback. Do them in Phase 0.
- **D1.** Exact attribute path to the underlying transformer (`clf.model_` vs nested). → `tabpfn_hooks.find_transformer()` prints the module tree.
- **D2.** Shape/semantics of the residual stream per block (per-cell `[items, features, dim]`? where the target/label token lives). → `tabpfn_hooks.describe_activations()` logs shapes for a toy table.
- **D3.** Whether categorical/continuous columns are tokenized differently (affects column-token labeling). → inspect the model's input encoder.
- **D4.** Differentiability of the forward w.r.t. input `X` (needed for SGLD generation). → `generate.energy.check_grad()`.

---

## 1. Project scaffolding

### 1.1 `pyproject.toml` / `requirements.txt`
**What:** declare dependencies + package metadata. **Why:** reproducible env on both laptop and server.
Core deps: `torch`, `tabpfn` (official, open v2 weights), `numpy`, `pandas`, `scikit-learn`, `scipy`, `hydra-core`, `omegaconf`, `einops`, `tqdm`. Eval/extra: `sdmetrics`, `synthcity` (gen metrics), `synth-mia` (optional privacy). SAE: `dictionary_learning` **or** `sae_lens` (we wrap one; see §6). Dev: `pytest`, `ruff`, `rich`.

### 1.2 Directory tree (target)
```
tabular-data-gen/
  pyproject.toml
  configs/
    config.yaml            # Hydra root (defaults list)
    smoke.yaml             # CPU tiny end-to-end
    full.yaml              # GPU server
    scm/{default,smoke}.yaml
    sae/{topk,jumprelu,matryoshka}.yaml
    gen/{default}.yaml
  src/tabsae/
    __init__.py
    types.py               # shared dataclasses (§2)
    utils/{device.py,seed.py,io.py,logging.py}
    scm/{generators.py,concepts.py}
    tabpfn_hooks.py
    activations.py
    sae/{models.py,train.py,metrics.py}
    interp/{concept_align.py,causal.py,baselines.py,autointerp.py}
    generate/{energy.py,sgld.py,steer.py,eval_gen.py}
  scripts/{run_extract.py,run_train_sae.py,run_interp.py,run_generate.py,run_all_smoke.py}
  tests/
  experiments/             # outputs: metrics json/csv, figures
  paper/
  README.md
```

---

## 2. `src/tabsae/types.py` — shared data structures
**Why:** one source of truth for the tensors passed between modules; prevents shape drift.

```python
@dataclass
class SCMDataset:
    """One synthetic table + its ground-truth structure."""
    X: np.ndarray            # [n_rows, n_cols] feature matrix
    y: np.ndarray            # [n_rows] target
    col_types: list[str]     # 'num' | 'cat' per column
    concept_labels: "ConceptLabels"  # ground-truth roles (see concepts.py)
    meta: dict               # scm graph, seed, functional forms, etc.

@dataclass
class ConceptLabels:
    """Per-column / per-pair structural roles known a priori."""
    monotone: list[bool]              # per column
    irrelevant: list[bool]            # per column
    covariate_shift: list[bool]       # per column (context vs query)
    interactions: list[tuple[int,int,str]]  # (j,k,'xor'|'and'|...)
    redundant: list[tuple[int,int]]   # (j duplicates k)
    def column_role(self, j) -> set[str]: ...   # convenience → {'monotone',...}

@dataclass
class ActivationBatch:
    """Activations pulled from one TabPFN forward pass at one hook."""
    acts: np.ndarray         # [n_tokens, dim] (token = cell or column or query-item)
    token_kind: str          # 'cell' | 'column' | 'query_item'
    col_index: np.ndarray    # [n_tokens] which column each token belongs to (-1 if n/a)
    is_query: np.ndarray     # [n_tokens] bool
    dataset_id: str
    layer: int
    hook: str

@dataclass
class SAEConfig:
    d_in: int; d_sae: int; variant: str   # 'topk'|'jumprelu'|'matryoshka'
    k: int | None; l1: float | None        # variant-specific
    matryoshka_sizes: list[int] | None
    lr: float; steps: int; seed: int
```

---

## 3. `src/tabsae/utils/`
- `device.py: get_device(prefer:str='auto') -> torch.device` — auto-detect cuda/cpu. **Why:** never hard-code.
- `seed.py: set_global_seed(seed:int)` — seed torch/numpy/random + cudnn deterministic. **Why:** reproducibility.
- `io.py: save_acts(path, ActivationBatch)`, `load_acts(path)`, `MemmapShardWriter` / `MemmapShardReader` — sharded memmap activation store. **Why:** activation caches can be large on the server; memmap keeps RAM flat and lets CPU dev stream.
- `logging.py: get_logger(name)`, `log_config(cfg)` — rich logging + dump resolved Hydra config to the run dir. **Why:** every result traceable to a config.

---

## 4. `src/tabsae/scm/` — ground-truth data (the moat)

### 4.1 `concepts.py`
**What:** define the structural roles and the functions that *label* a generated dataset. **Why:** these labels are the ground truth every interpretability metric is scored against — the thing no other SAE domain has.

```python
CONCEPTS = ['monotone','irrelevant','covariate_shift','interaction','redundant']

def label_dataset(scm_spec) -> ConceptLabels:
    """Derive ConceptLabels directly from the SCM spec used to generate the table.
    Why: labels come from the generative process, so they are exact, not inferred."""

def concept_indicator(labels: ConceptLabels, concept: str) -> np.ndarray:
    """Return a per-COLUMN boolean vector for `concept`.
    Used to label column-token activations for alignment scoring."""

def make_concept_dataframe(datasets: list[SCMDataset], acts: list[ActivationBatch]) -> pd.DataFrame:
    """Join column-token activations with their concept labels across many datasets →
    a tidy (token, concept-label, activation-vector) table for alignment + probing."""
```

### 4.2 `generators.py`
**What:** sample SCM datasets with *controllable* planted structure. **Why:** we need many datasets where we know exactly which column is monotone / irrelevant / shifted / interacting.

```python
def sample_scm_spec(rng, cfg) -> dict:
    """Sample a random causal graph + functional forms + noise, but RECORD every choice
    (which edges, which monotone links, which interaction, planted shift/redundancy)."""

def render_dataset(scm_spec, n_rows, rng, with_shift:bool) -> SCMDataset:
    """Realize X,y from the spec. If with_shift, draw context vs query rows from
    different feature distributions to plant covariate shift on chosen columns."""

def make_controlled_dataset(concept:str, **kw) -> SCMDataset:
    """Generate a dataset that plants EXACTLY ONE clean instance of `concept`
    (e.g., a single monotone column, everything else irrelevant).
    Why: needed for the make-or-break sanity test and for clean causal experiments."""

def generate_corpus(cfg) -> list[SCMDataset]:
    """Produce N labeled datasets per the config (sizes within TabPFN's sweet spot:
    <=10k rows, <=~100 cols). Persist specs + labels."""
```

Notes: keep `n_rows`, `n_cols` small in `smoke` (e.g., 64 rows, 5 cols, 20 datasets) so CPU runs in seconds.

---

## 5. `src/tabsae/tabpfn_hooks.py` — read TabPFN's mind
**What:** load TabPFN v2, locate its transformer, attach forward hooks, return activations. **Why:** SAEs need the residual-stream activations; this is the bridge.

```python
def load_tabpfn(task:str='classifier', device=None):
    """Load open-weights TabPFN v2 (TabPFNClassifier/Regressor). Return (estimator, model)."""

def find_transformer(model) -> torch.nn.Module:        # D1
    """Return the underlying per-feature transformer module. Print module tree on first call."""

def list_hookable_layers(model) -> list[str]:
    """Return names of residual-stream points per block (post-attn, post-MLP).
    Why: we sweep layers to find where concepts are most linearly represented."""

def describe_activations(model, toy_table) -> dict:    # D2/D3
    """Run one forward pass; log the shape + token semantics of each hook.
    Resolves where the query tokens / column tokens / target token live."""

class ActivationCollector:
    """Context manager that registers forward hooks at requested (layer, hook) points
    and collects ActivationBatch objects for one forward pass.
    Why: clean attach/detach so we never leak hooks across runs."""
    def __init__(self, model, points:list[tuple[int,str]], token_kind:str): ...
    def __enter__(self)->'ActivationCollector'; def __exit__(self,*a): ...
    def run(self, X_ctx, y_ctx, X_query) -> dict[tuple,ActivationBatch]:
        """Single TabPFN forward over (context set, query set); return activations per point."""

def patch_forward(model, point, patch_fn):
    """Register a hook that REPLACES the activation at `point` with patch_fn(act).
    Why: the core primitive for causal validation (ablate a latent) and for steering."""
```

Token-kind handling: for **column** tokens we tag each with its `col_index` so it can be labeled by `concept_indicator`. For **query_item** tokens we pool per query row (used for prediction-relevant primitives + generation).

---

## 6. `src/tabsae/activations.py` — extraction & caching
**What:** drive `ActivationCollector` over a corpus and persist a training set for SAEs. **Why:** decouples the (heavier) forward passes from (iterative) SAE training.

```python
def extract_corpus_activations(model, datasets, points, token_kind, out_dir, cfg) -> Manifest:
    """For each dataset: split into context/query, run ActivationCollector, normalize
    PER-DATASET (z-score) to remove cross-dataset scale (handles 'no fixed semantics'),
    write sharded memmap + a manifest (which shard holds which dataset/concept labels)."""

class ActivationDataset(torch.utils.data.Dataset):
    """Streams cached activations for SAE training. Returns (act_vector, meta) where
    meta carries dataset_id/col_index/concept labels for downstream alignment."""

def make_loaders(manifest, batch_size, split) -> tuple[DataLoader, DataLoader]:
    """Train/val split that holds out whole DATASETS (not rows) to test transfer."""
```
Why per-dataset normalization: TabPFN has no fixed feature semantics across tables; normalizing per dataset makes the SAE learn *dataset-invariant* primitives (the column's structural role), not its arbitrary scale.

---

## 7. `src/tabsae/sae/` — the dictionaries

### 7.1 `models.py`
**What:** the SAE variants. **Why:** different sparsity mechanisms; Matryoshka specifically fights feature *absorption* of hierarchical tabular concepts. Wrap `dictionary_learning`/`sae_lens` internals where possible; expose a unified API.

```python
class BaseSAE(nn.Module):
    def encode(self, x) -> Tensor          # [B, d_sae] latent activations
    def decode(self, z) -> Tensor          # [B, d_in] reconstruction
    def forward(self, x) -> (recon, z, loss_dict)
    def ablate(self, x, latents:list[int]) -> Tensor:
        """Return reconstruction with given latents forced to 0 PLUS the residual error
        term (so non-targeted computation is preserved). Core for causal ablation."""

class TopKSAE(BaseSAE):    # keep top-k latents; clean L0 control (OpenAI 2406.04093)
class JumpReLUSAE(BaseSAE): # learnable per-latent threshold; SOTA reconstruction (2407.14435)
class MatryoshkaSAE(BaseSAE): # nested prefixes reconstruct independently (2503.17547)

def build_sae(cfg: SAEConfig) -> BaseSAE: ...
```

### 7.2 `train.py`
```python
def train_sae(sae, train_loader, val_loader, cfg) -> TrainReport:
    """Standard reconstruction + sparsity training with AuxK dead-latent revival.
    Logs reconstruction/L0/dead-fraction each step. Why AuxK: large dicts develop dead
    latents that waste capacity and bias completeness."""

def train_sae_multi_seed(cfg, seeds:list[int]) -> list[BaseSAE]:
    """Train N seeds → enables cross-seed stability measurement (known SAE weakness)."""
```

### 7.3 `metrics.py`
```python
def reconstruction_metrics(sae, loader) -> dict     # MSE, fraction-of-variance-explained
def sparsity_metrics(sae, loader) -> dict           # mean L0, dead %, ultra-low-freq %
def frontier(saes:list, loader) -> pd.DataFrame     # reconstruction vs L0 Pareto curve
def cross_seed_stability(saes:list) -> dict         # max-cosine matching of dictionaries across seeds
def monosemanticity(sae, concept_df) -> dict        # per-latent selectivity for a single concept
```

---

## 8. `src/tabsae/interp/` — the headline (C1)

### 8.1 `concept_align.py`
**What:** quantify which SAE latents correspond to which ground-truth concept. **Why:** the central correlational claim, scored against exact labels.
```python
def latent_activation_table(sae, acts_dataset) -> np.ndarray:   # [n_tokens, d_sae]
def align_latents_to_concepts(latent_acts, concept_df) -> pd.DataFrame:
    """For each (latent, concept): selectivity = AUROC/F1 of 'latent fires' predicting
    'token's column has this concept'. Return best latent per concept + scores."""
def purity_and_coverage(alignment) -> dict:
    """Purity: does a latent fire ~only for one concept? Coverage: is each concept
    captured by some latent? Why: monosemanticity + completeness summary."""
```

### 8.2 `baselines.py`
**What:** the comparators reviewers demand. **Why:** SAEs must be shown to add value over trivial methods (per 'Are SAEs useful?', 2502.16681).
```python
def lr_probe_concept(raw_acts, concept_labels) -> dict   # logistic regression on RAW activations
def pca_directions(raw_acts, n) -> np.ndarray            # PCA baseline directions
def neuron_selectivity(raw_acts, concept_labels) -> dict # best single raw neuron per concept
def compare_to_baselines(sae_alignment, baselines) -> pd.DataFrame
```

### 8.3 `causal.py` — necessity & sufficiency (the moat)
**What:** verify alignment is causal, not correlational, via activation patching. **Why:** this is what makes the paper A\* and what no protein/vision SAE paper can do.
```python
def measure_necessity(model, sae, latent:int, datasets_with_concept, point) -> dict:
    """ABLATE `latent` in TabPFN's forward (patch_forward + sae.ablate) on datasets where
    the concept is present; measure degradation of TabPFN's ability to exploit that
    structure (Δ accuracy on the structural task AND KL between original vs patched PPD)."""
def measure_sufficiency(model, sae, latent:int, datasets_without_concept, point) -> dict:
    """INJECT the latent direction into datasets lacking the concept; test whether TabPFN
    behaves as if the structure were present (predicted directional effect)."""
def causal_vs_baseline(model, sae, lr_direction, ...) -> dict:
    """Show that ablating the SAE latent has a clean causal effect while ablating the
    matched LR/PCA direction does not (or has diffuse off-target effects)."""
```

### 8.4 `autointerp.py` (optional)
```python
def describe_latent(latent, top_examples) -> str:
    """Heuristic/LLM-assisted name for a latent from its top-activating columns + their
    SCM roles. Why: human-facing labels; kept optional to avoid external-API dependence."""
```

---

## 9. `src/tabsae/generate/` — interpretable generation (C2)

### 9.1 `energy.py` — TabPFN as an energy model
**What:** turn frozen TabPFN into a class-conditional generator (TabPFGen/TabEBM-style). **Why:** lets one model both *explain* and *generate*; steering then reuses the same hooks.
```python
def class_energy(model, X_ctx, y_ctx, x, c) -> Tensor:
    """E(x) = -log p(y=c | x, context) from TabPFN logits. Differentiable in x."""
def check_grad(model, ...) -> bool                       # D4: confirm ∇_x flows
```

### 9.2 `sgld.py`
```python
def sgld_sample(energy_fn, x0, steps, eps, noise) -> Tensor:
    """Langevin sampling: x ← x - (eps/2)∇_x E(x) + sqrt(eps)·η. Returns synthetic rows."""
def generate_class_conditional(model, X_ctx, y_ctx, n, c, cfg) -> np.ndarray
```

### 9.3 `steer.py`
**What:** steer generation by clamping a SAE latent during sampling. **Why:** the controllable, interpretable-generation capability that elevates the paper beyond an interpretability catalogue.
```python
def steer_during_sampling(model, sae, latent:int, value:float):
    """Return a patch_fn that clamps `latent` to `value` inside the residual stream so the
    energy (and its gradient) reflect the intervention at every SGLD step."""
def controllability_report(base_samples, steered_samples, concept) -> dict:
    """On-target effect (shift in the steered concept's statistic) vs off-target distortion
    (drift in other columns). Mirrors SAE-TS steering-vs-coherence (2411.02193)."""
```

### 9.4 `eval_gen.py`
```python
def fidelity_suite(real, synth) -> dict:   # Shape, Trend, alpha-precision/beta-recall, C2ST, MLE, DCR (sdmetrics/synthcity)
def privacy_suite(real, synth) -> dict:    # optional Synth-MIA AUROC, DCR-share
def controllability_suite(...) -> dict     # wraps controllability_report across concepts
```

---

## 10. `scripts/` — CLI entry points (Hydra)
Each is thin: parse config, call library, write `experiments/<run>/...`.
- `run_extract.py` — generate SCM corpus (or load real data) → extract+cache activations. Output: manifest + shards.
- `run_train_sae.py` — train SAE(s) (optionally multi-seed) on cached activations. Output: SAE checkpoints + frontier/stability metrics.
- `run_interp.py` — concept alignment + baselines + causal necessity/sufficiency. Output: the C1 tables/figures.
- `run_generate.py` — TabPFN-EBM generation + steering + eval. Output: C2 tables/figures.
- `run_all_smoke.py` — runs all four end-to-end on the `smoke` config (CPU). Output: a green/red end-to-end check.

---

## 11. Config schema (Hydra)
`config.yaml` defaults list pulls `scm`, `sae`, `gen` groups. Key fields:
```yaml
device: auto
seed: 0
scm:   {n_datasets, n_rows, n_cols, concepts, shift_prob}
extract: {points: [[L,'resid_post']...], token_kind: column|query_item, out_dir}
sae:   {variant, d_sae, k, l1, matryoshka_sizes, lr, steps, seeds}
interp:{concepts, causal: {ablate, inject, n_eval_datasets}}
gen:   {classes, sgld: {steps, eps, noise}, steer: {latent, values}}
```
`smoke.yaml` overrides everything tiny (CPU). `full.yaml` scales for the GPU server (more datasets, wider SAE, more seeds). README documents the exact diff.

---

## 12. `tests/` — all CPU-runnable
- `test_device.py` — auto-detect returns cpu when no cuda.
- `test_scm_labels.py` — planted concept → `ConceptLabels` correct (e.g., monotone column flagged).
- `test_hooks.py` — `ActivationCollector` returns expected shapes/token kinds on a toy table.
- `test_cache.py` — write→read memmap shards round-trips; dataset-level split holds out whole datasets.
- `test_sae.py` — SAE trains a few steps; reconstruction beats PCA at matched L0; `ablate` zeros the right latent.
- `test_align.py` — on a controlled single-monotone dataset, some latent aligns (high F1).
- `test_causal.py` — ablating that latent degrades TabPFN's use of the monotone structure; ablating a random direction does not.
- `test_generate.py` — `check_grad` true; SGLD runs; clamping a latent shifts the targeted statistic.
- `test_end_to_end_smoke.py` — `run_all_smoke` completes and asserts the make-or-break checks.

---

## 13. Results → artifact map (what produces each paper claim)
- **C1 alignment table/fig** ← `run_interp` (`align_latents_to_concepts`, `compare_to_baselines`).
- **C1 causal fig (necessity/sufficiency)** ← `run_interp` (`measure_necessity/sufficiency`, `causal_vs_baseline`).
- **Frontier + cross-seed stability** ← `run_train_sae` (`frontier`, `cross_seed_stability`).
- **C2 controllability + fidelity tables** ← `run_generate` (`controllability_suite`, `fidelity_suite`).
- **C3 benchmark release** ← the SCM corpus + `concept_align`/`causal` protocol packaged as a dataset+evaluator.

---

## 14. Execution checklist (milestone order)
1. **P0** scaffolding + `load_tabpfn` + D1–D4 discovery (`describe_activations`, `check_grad`). ✔ when `test_hooks` green on CPU.
2. **P1** `scm/` generators + concepts. ✔ when `test_scm_labels` green; corpus generates.
3. **P2** `activations.py` extraction+cache. ✔ when `test_cache` green; smoke extract in seconds.
4. **P3** `sae/` models+train+metrics + baselines. ✔ when SAE > PCA at matched L0; multi-seed runs.
5. **P4 (headline)** `interp/` alignment + causal. ✔ when controlled-monotone make-or-break passes and baselines can't reproduce the causal effect.
6. **P5** `generate/` energy+sgld+steer+eval. ✔ when clamping a concept latent makes a predicted on-target change, fidelity preserved.
7. **P6 (GPU)** real data (CC18/Adult/California Housing) + TabSyn-suite + full sweeps/ablations.
8. **P7** figures, C3 packaging, paper draft.

---

## 15. Reuse map (don't reinvent)
| Ours | Backed by |
|------|-----------|
| `sae/models.py` variants | `dictionary_learning` (Marks et al.) or `sae_lens` |
| `tabpfn_hooks.load_tabpfn` | official `tabpfn` package (open v2 weights) |
| `generate/energy.py` | TabPFGen (arXiv:2406.05216) / TabEBM (NeurIPS'24) recipe |
| `generate/eval_gen.fidelity_suite` | `sdmetrics` / `synthcity`; TabSyn-repo protocol (arXiv:2310.09656) |
| `generate/eval_gen.privacy_suite` | `synth-mia` (arXiv:2509.18014), mostlyai-qa (arXiv:2504.01908) |
| causal patching pattern | standard activation-patching from mech-interp |

## 16. Known unknowns to confirm while coding
- Exact TabPFN v2 internal module names/shapes (D1–D3) — fallbacks in `tabpfn_hooks`.
- Whether column-token or query-item hook gives cleaner concept alignment — decided empirically in P4 (sweep both).
- Energy/gradient stability of TabPFN-as-EBM on mixed types (D4) — if unstable, restrict C2 to continuous columns first, then extend.
- SAEs may not beat LR probes on alignment accuracy — expected; lead the C1 story with **causal** results + monosemanticity, not probe accuracy.
```
```

> When you're ready, say the word and I'll start implementing against this guide, beginning with Phase 0 scaffolding + the TabPFN hook discovery (D1–D4).
