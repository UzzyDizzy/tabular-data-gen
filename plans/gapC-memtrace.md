# Gap C — MemTrace: Mechanistic Privacy for Tabular Generative Models

## Context

**Why this work.** "Synthetic ⇒ private" is false. Tabular diffusion models memorize training rows — **heavy-tailed**, a few records leak disproportionately (ICML 2025, arXiv:2412.11044; arXiv:2505.22322) — and the ubiquitous **Distance-to-Closest-Record (DCR)** metric is a proven-weak proxy that strong membership-inference attacks defeat (**Synth-MIA**, arXiv:2509.18014; Systematic Assessment, CCS 2025). Defenses are blunt (DP-SGD wrecks utility) or heuristic. In LLMs, mechanistic interpretability can *localize* memorization (crosscoders / model-diffing, Anthropic 2024); **no one has done this for tabular generators**.

**The problem we solve.** We need to *find* and *surgically remove* the specific internal directions that drive memorization — cutting MIA risk with minimal utility loss, and providing an **auditable, mechanistic** privacy story regulators can inspect, rather than a black-box distance number.

**Outcome.** A method + audit protocol that (i) mechanistically characterizes memorization in tabular diffusion, (ii) ablates memorizing directions training-free, and (iii) attributes synthetic records to training records (provenance).

## Contributions (claims)

- **C1:** mechanistic characterization — dictionary features in a tabular generator's latent that encode/copy specific training rows; quantify the heavy-tailed structure.
- **C2:** a **surgical, training-free memorization-removal** method (ablate/edit memorizing directions) that lowers Synth-MIA AUROC at far smaller utility cost than DP baselines.
- **C3:** a **mechanistic privacy audit / provenance** — attribute a synthetic row to its influential training rows via features, stronger and more interpretable than DCR.

## Key design decisions

- **Hosts:** TabDDPM (arXiv:2209.15421) and TabSyn (arXiv:2310.09656) — both shown to memorize.
- **Supervision for "memorizing" features:** use Synth-MIA / known-memorized rows (and controlled duplicate-injection experiments) as ground-truth memorization labels to identify which latents fire on them.
- **Decomposition tools:** SAEs + **crosscoders** (cross-layer / base-vs-overfit model diffing) to localize memorization that emerges with training.
- **e2e/KL-trained SAEs** (arXiv:2405.12241) to ensure features capture *functional* (output-affecting) memorization, not just reconstruction.

## Architecture

```
Train generator (TabDDPM / TabSyn), optionally with injected duplicate rows (ground-truth memorization)
        │
        ├─► Synth-MIA / duplicate labels ──► which rows are memorized?
        ▼
   SAE / crosscoder on generator activations ──► latents that fire on memorized rows (C1)
        │
        ├─► ABLATE/edit those latents during sampling ──► re-run Synth-MIA + utility (C2)
        └─► feature-attribution from synthetic row → training rows ──► provenance audit (C3)
```

## Repository structure

Reuses shared `src/tabsae/` core. Adds:

```
src/tabsae/
  hosts/tabddpm.py, hosts/tabsyn.py   # train/load; activation hooks
  privacy/memorization.py  # duplicate-injection protocol; memorized-row labeling
  privacy/synth_mia.py     # Synth-MIA attack suite wrapper
  sae/crosscoder.py        # cross-layer / model-diffing crosscoder
  interp/mem_features.py    # identify memorization-correlated latents (C1)
  generate/ablate.py        # ablate/edit memorizing directions during sampling (C2)
  audit/provenance.py       # synthetic→training attribution (C3)
  generate/eval_gen.py      # fidelity/utility + DCR-share + Synth-MIA
```

**Reuse:** Synth-MIA repo, mostlyai-qa (DCR-share, arXiv:2504.01908), TabDDPM/TabSyn repos, `dictionary_learning`/`SAELens`.

## Implementation phases

- **P0 Bootstrap:** train/load TabDDPM + TabSyn on a small dataset; hooks; Synth-MIA runs.
- **P1 Memorization ground truth:** duplicate-injection protocol → known memorized rows; confirm models copy them; DCR vs MIA gap.
- **P2 SAE/crosscoder on activations:** train; cache.
- **P3 C1:** identify latents that fire selectively on memorized rows; characterize heavy tail.
- **P4 C2:** ablate/edit those latents during sampling → re-measure Synth-MIA AUROC + TSTR utility + fidelity; compare vs DP-SGD and naive nearest-neighbor rejection.
- **P5 C3:** provenance attribution; evaluate against injected-duplicate ground truth.
- **P6 Full sweeps (GPU):** TabSyn suite + a memorization-prone dataset; ablations.
- **P7 Writeup & release.**

## Datasets & evaluation

- **TabSyn 6-suite** + a dataset with rare unique rows (high memorization).
- **Privacy:** Synth-MIA AUROC (multiple attacks), DCR-share — before/after ablation.
- **Utility/fidelity:** TSTR, Shape/Trend/C2ST — must be largely preserved.
- **Provenance:** attribution precision/recall vs injected-duplicate ground truth.
- **Baselines:** DP-SGD generators (utility cost), nearest-neighbor rejection, no-defense.

## CPU-dev → GPU-scale

Generator training is GPU-bound (medium); develop on small-subset CPU smoke, scale on server. SAE/crosscoder/ablation/audit run on cached activations (CPU-debuggable). Hydra smoke/full; device auto-detected.

## Verification

1. `pytest tests/`: duplicate-injection labels correct, Synth-MIA runs, ablation hook works.
2. End-to-end CPU smoke: inject a duplicate → confirm it's memorized → find the latent → ablate → memorization drops.
3. **Make-or-break:** ablating identified latents reduces Synth-MIA AUROC meaningfully **with** small TSTR/fidelity loss, beating nearest-neighbor rejection at equal utility.
4. Provenance recovers injected-duplicate sources above chance.
5. Repro: seeds, configs.

## Risks & mitigations

- **Feature→specific-record linkage is the hardest claim** → start with controlled duplicate-injection (clean ground truth), then generalize; be honest where attribution is coarse.
- Ablation may hurt fidelity → quantify the privacy-utility frontier vs DP; frame as a better frontier, not free lunch.
- Memorization may be distributed (not low-rank) → crosscoder + e2e SAEs; report if it resists localization (still a finding).
- Compute (train generators) → small-subset CPU dev; GPU sweeps.

## Venue & timeline (date 2026-06-14)

- **Primary: NeurIPS / ICML** (privacy + interpretability) or a **security crossover** (USENIX Security / CCS) given the attack/defense framing.
- High impact; the riskiest of the four to land *conclusively* — strongest if the duplicate-injection ground-truth experiments are crisp.
