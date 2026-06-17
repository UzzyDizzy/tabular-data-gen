# Real TabPFN integration — findings & next steps (D1–D3)

Status from the first real-backend integration session (model: **TabPFN v2.5**, loaded via
`TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)`, open weights downloaded after
accepting the 2.5 license + HF gate).

## What works
- ✅ **Loads & runs** on CPU. `PerFeatureTransformer`, ~10.7M params, **24 `PerFeatureEncoderLayer`** blocks at `model.transformer_encoder.layers`. Predict accuracy **1.0** on a controlled monotone dataset.
- ✅ **Residual stream extracted.** Encoder-layer output shape = `[batch=1, items, feature_groups(+target), emsize=192]`. So `d_model = 192`.
- ✅ **SAE trains on real activations** (FVE ~0.79–0.98 depending on layer).
- ✅ Hooks wired: `_transformer_blocks()` → `transformer_encoder.layers`; `_to_tokens()` handles the 4-D layout.

## The blocker (D3): feature-token ↔ column mapping is scrambled
- TabPFN groups features (`features_per_group = 3`, padded) and adds **thinking tokens** (items axis = 224 for 128+32 samples).
- **Critically: default `n_estimators = 8`.** TabPFN runs 8 forward passes ("prompts"), **each with a different feature/class permutation + preprocessing transform**. Our forward hook captures one pass whose feature axis is permuted vs the original columns.
- **Result:** probing per-feature(-group) residual tokens for SCM concepts is **at chance across all layers (L4/L10/L16/L22), for BOTH the SAE and the linear-probe baseline (~0.55–0.59)**, even with correct OR-over-group labels. Chance LR probe ⇒ the labels don't line up with the tokens (permutation), not that the SAE failed.

## Next steps to unlock real-backend interpretability
1. **Deterministic, identity feature mapping.** Configure the estimator with `n_estimators=1` and an ensemble/preprocessing config that uses **no feature shuffle / identity permutation** (control `EnsembleConfig` feature_shift + the preprocessing transforms). Then the residual feature axis maps to original columns. Re-probe: the LR baseline rising above chance is the green light.
   - Alternatively, **track each estimator's permutation** and un-permute captured activations back to original column order (keeps ensembling).
2. **Layer sweep** once mapping is correct (concepts likely peak mid-network).
3. **features_per_group:** it's baked into the checkpoint (=3), so tokens are per-group. For per-column resolution either keep group-level labels (OR over members) or investigate a config/checkpoint with fpg=1.
4. **Causal patching for the real backend.** `_forward_capture` currently captures but does not *apply* patches. Add a forward hook that modifies the encoder-layer output in token space (normalize → SAE ablate/clamp → denormalize → write back), per feature token across items. (Works already on the mock backend.)
5. **Generation for the real backend.** `MockEnergyModel` uses mock-only methods (`column_activations`, `W_pinv`). For real TabPFN implement TabPFGen/TabEBM-style energy = `-log p(y=c | x, context)` differentiated w.r.t. `x` (discovery task D4), then steer via SAE-latent clamping during SGLD.

## UPDATE (2026-06-15): step 1 done — deterministic mapping works
Added `TabPFNBackend(deterministic=True)`: loads with `n_estimators=1`, `auto_scale_n_estimators=False`,
and `inference_config={FEATURE_SHIFT_METHOD:None, CLASS_SHIFT_METHOD:None, FINGERPRINT_FEATURE:False,
POLYNOMIAL_FEATURES:"no"}`. This removes per-estimator feature permutation, so residual feature
group g ↔ original columns `[g*fpg:(g+1)*fpg]`.

Result (50 SCM datasets, n_cols=12, fpg=3, OR-over-group labels, mean predict acc ~1.0):
- Concepts are now recovered **above chance**, **peaking mid-network at L10**:
  `monotone SAE=0.61/LR=0.67`, `redundant 0.65/0.68`, `irrelevant 0.62/0.63`, `covariate_shift 0.60/0.63`.
- L22 (last layer) decays toward chance — concepts are mid-network, not at the output.
- SAE tracks the LR probe (slightly below), consistent with the field.

### Why numbers are still modest + next refinements
- **Group mixing (fpg=3)** dilutes labels (a group token blends 3 columns; OR-labeling is coarse).
  Fix: generate SCM datasets with **roles assigned in contiguous blocks of size fpg** so each group
  token is single-role → clean labels. (Biggest expected lift.)
- More datasets/tokens; focus on L10; sweep all layers.
- Then wire **causal patching** + **EBM generation** for the real backend (currently mock-only).

## UPDATE 2 (2026-06-17): fpg-aligned block-SCM — strong C1 signal
Added `block_size` to the SCM generator (`generate_corpus(..., block_size=3)`): one structural
role per contiguous block of `features_per_group` columns, so every TabPFN feature-token is
single-role → sharp labels. (Gotcha found & fixed: `_features_per_group()` returns 1 *before* a
forward populates the model — read it AFTER the first `run()`; the library's internal `_to_tokens`
already calls it post-fit so extraction was always correct, only ad-hoc probe labels were wrong.)

Result (60 datasets, n_cols=12, fpg=3, deterministic, mean predict acc 0.94, S=SAE / L=LR-probe AUROC):

| Layer | monotone | irrelevant | covariate_shift | interaction | redundant |
|------|----------|-----------|-----------------|-------------|-----------|
| 4    | .66/.72  | .63/.75   | .68/.82         | .61/.86     | .77/.89   |
| 10   | .68/.86  | .80/.95   | .77/.82         | .63/.87     | .65/.94   |
| 16   | .76/.88  | .92/.93   | .64/.81         | .64/.89     | .66/.90   |
| 22   | .64/.86  | .72/.90   | .59/.72         | .63/.85     | .68/.88   |

**Takeaways:** (1) all 5 concepts are linearly decodable from TabPFN's residual stream, **LR 0.81–0.95**,
**peaking mid-network (L10–L16)** and decaying by L22 — a clean mech-interp layer profile.
(2) SAE latents recover them too (irrelevant **0.92** at L16), lagging the probe on interaction
(.64 vs .89) — expected; the SAE's edge is causal steerability/monosemanticity, not probe accuracy.

## UPDATE 3 (2026-06-17): real-backend CAUSAL patching works
`TabPFNBackend._forward_capture` now applies `patches` via a forward hook that modifies the
encoder-layer output in token space (normalize → SAE ablate/clamp → denormalize, per feature-group
token across all items, excluding the target token), so the intervention propagates to the
prediction. Requires `deterministic=True` (single, identity-ordered pass).

Demo (24 block-SCM datasets, L16, monotone latent #88, AUROC 0.71/LR 0.78):
- **Necessity** (ablate monotone latent on monotone datasets): KL **0.00226** vs random-latent
  control **1.2e-5** → ~**196×**. Hard accuracy doesn't flip (TabPFN is redundant), so the signal
  is the predictive-distribution shift (KL), strongly above control.
- **Sufficiency** (inject latent on all-irrelevant datasets): effect **1.7×** control.

So the causal moat is demonstrated on the real model. Effects are modest in absolute terms (TabPFN
robustness) but show the predicted asymmetry vs controls — stronger demonstrations likely come from
better-isolated latents (JumpReLU/Matryoshka, more data) and ablating multiple monotone latents.

### Next
- Strengthen SAE side: JumpReLU/Matryoshka, wider dict, more datasets, focus L10–L16; monosemanticity + cross-seed stability on real activations; multi-latent ablation for larger causal effects.
- Wire **EBM generation** for the real backend (TabPFGen-style energy + SGLD + SAE-latent steering); currently mock-only.

## Note
The **mock backend validates the full method** (alignment + causal + steering, 14 tests green). Real TabPFN now shows a **strong C1 result** (LR up to 0.95, SAE up to 0.92, mid-network) with the deterministic mapping + block-SCM.
