# Gap D — ClinSteer: Auditable, Concept-Steerable Synthetic EHR

## Context

**Why this work.** Synthetic EHR is a major use case (diffusion now SOTA: EHRDiff, arXiv:2303.05656; EHR-temporal diffusion, JAMIA, arXiv:2310.15290; EHR-Safe, *npj Digital Medicine*). But the healthcare-eval literature flags an unsolved blocker: **auditability / provenance / regulatory acceptance**. FDA RWD/RWE guidance expects audit trails and provenance, yet black-box deep generators can't say *which* real records or *which* clinical factors drove a synthetic record, and there's no certification that a synthetic dataset is simultaneously private, faithful, and fair (Yan et al., *Nature Communications* 2022; SynthEHRella, arXiv:2411.04281). Meanwhile concept-based interpretability for tabular clinical data exists (TabCBM, TMLR 2023) but is not married to generation, and **no one has SAE-decomposed an EHR generator**.

**The problem we solve.** Produce synthetic patient cohorts that are (i) **steerable by clinical concept** ("diabetic patients with renal complications"), and (ii) **auditable** — each synthetic record carries a per-feature/per-source provenance trail regulators can inspect.

**Outcome.** An interpretable EHR-generation pipeline: generator + SAE-derived clinical concept features + concept steering + provenance audit.

## Contributions (claims)

- **C1:** SAE-decomposed **clinical concept** features from an EHR generator's latent (map to ICD/phenotype/lab concepts), validated against clinical annotations + clinician sanity checks.
- **C2:** **concept-steerable cohort generation** — generate cohorts by clinical concept with controllable prevalence.
- **C3:** a **provenance / audit-trail** mechanism (which concepts + training records drove each synthetic record) targeting FDA RWD/RWE auditability.

## Key design decisions

- **Data:** MIMIC-III / MIMIC-IV / eICU (credentialed via PhysioNet) — **start with phenotype/code-matrix EHR** (simpler, well-benchmarked by SynthEHRella) before longitudinal time series.
- **Host generator:** an EHR diffusion model (EHRDiff-style) or TabSyn adapted to EHR; frozen after training.
- **Concept supervision:** ICD groupings / phenotype labels / lab-panel concepts as weak labels for concept alignment.
- **SAE variants:** TopK / JumpReLU / Matryoshka (Matryoshka for hierarchical clinical concepts).

> ⚠️ **Prerequisite:** credentialed MIMIC/eICU access (PhysioNet CITI training + DUA) — can take days–weeks if not already held. You selected *general-purpose*; this plan is the applied alternative and assumes you have or will obtain access.

## Architecture

```
MIMIC/eICU phenotype matrices ─► train EHR generator ─► freeze ─► encode patients to latent (cache)
                                                                        │
                                                                        ▼
                                          SAE (TopK / JumpReLU / Matryoshka) on latent
                                                                        │
        ┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
        ▼                               ▼                              ▼
  C1 clinical concept features    C2 concept-steered cohort        C3 per-record provenance /
  (map to ICD/phenotype/labs)     generation (controllable          audit trail (concepts + sources)
                                  prevalence)
```

## Repository structure

Reuses shared `src/tabsae/` core. Adds:

```
src/tabsae/
  ehr/data.py         # MIMIC/eICU loaders → phenotype/code matrices; cohort definitions
  ehr/concepts.py     # ICD/phenotype/lab concept labels for alignment
  hosts/ehr_gen.py    # EHRDiff/TabSyn-EHR train/load; hooks
  interp/clinical_align.py  # SAE latent ↔ clinical concept (F1) + clinician-review export
  generate/cohort_steer.py  # concept-steered cohort sampling with prevalence control
  audit/provenance.py       # concept + training-source attribution per synthetic record
  generate/eval_ehr.py      # fidelity/utility (TSTR) + privacy (Synth-MIA) + clinical realism checks
```

**Reuse:** SynthEHRella (benchmarking), EHRDiff/EHR-diffusion repos, Synth-MIA, `dictionary_learning`/`SAELens`, `sdmetrics`/`synthcity`.

## Implementation phases

- **P0 Access + bootstrap:** secure PhysioNet credentials; build phenotype matrices; train/load a small EHR generator; hooks.
- **P1 Latent extraction + caching.**
- **P2 SAE on latent + baselines** (PCA, raw dims, neuron).
- **P3 C1 clinical concepts:** align latents to ICD/phenotype/lab concepts; export top cohorts for clinician sanity-check.
- **P4 C2 cohort steering:** steer concept prevalence; verify generated cohorts match requested clinical profile; fidelity preserved.
- **P5 C3 provenance:** per-record concept + source attribution; build the audit-trail artifact.
- **P6 Full eval (GPU):** MIMIC-III/IV + eICU; privacy (Synth-MIA), utility (TSTR), fairness across demographics.
- **P7 Writeup & release** (code + audit-trail spec; no patient data).

## Datasets & evaluation

- **MIMIC-III/IV, eICU** phenotype matrices (then longitudinal as stretch).
- **Fidelity/utility:** SynthEHRella protocol, TSTR.
- **Privacy:** Synth-MIA, DCR-share.
- **Clinical realism:** concept-prevalence correctness, clinician spot-checks, plausibility of co-occurrences.
- **Fairness:** subgroup fidelity across demographics.
- **Baselines:** EHRDiff/EHR-Safe (no concept control), classifier-free-guidance conditioning, TabCBM-style concept models.

## CPU-dev → GPU-scale

Develop on a small de-identified subset / few epochs locally (CPU smoke); full training + eval on the GPU server. SAE/steer/audit run on cached latents (CPU-debuggable). Hydra smoke/full; device auto-detected. **Never** stage raw patient data outside the credentialed environment.

## Verification

1. `pytest tests/`: data loaders, concept labels, SAE > PCA, steering + provenance hooks (on synthetic toy EHR, no real data in tests).
2. End-to-end CPU smoke on a tiny synthetic EHR: train → SAE → steer one concept → provenance trail produced.
3. **Make-or-break:** a latent aligns with a known clinical concept (e.g., diabetes phenotype); steering its prevalence produces cohorts whose downstream phenotype rate matches the request, with fidelity preserved.
4. Privacy: Synth-MIA within acceptable bounds; provenance attribution above chance.
5. Repro: seeds, configs; clinician review logged.

## Risks & mitigations

- **Data-access delay** (PhysioNet) → start the CITI/DUA process immediately; develop on synthetic/toy EHR meanwhile.
- Clinical concept validation needs domain expertise → recruit a clinician collaborator; use established phenotype definitions.
- Correlational-not-causal features (as in single-cell FMs) → causal ablation where possible; be explicit about limits for high-stakes use.
- Compute (train generator) → small-subset CPU dev; GPU sweeps.

## Venue & timeline (date 2026-06-14)

- **Primary: ML4H / CHIL / NeurIPS Datasets & Benchmarks**; main-track ICML/NeurIPS if the method (concept steering + provenance) is strong enough beyond the clinical application.
- Highest real-world/regulatory impact; gated by data access and clinician involvement.
