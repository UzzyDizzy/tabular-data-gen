"""End-to-end smoke test (the make-or-break checks), CPU-only, tiny config.

Asserts the pipeline:
  1. trains an SAE that reconstructs (FVE high),
  2. recovers concepts against ground truth (alignment),
  3. CAUSALLY validates the headline concept (necessity > control, sufficiency > control),
  4. steers generation selectively (on-target >> off-target).
"""
import numpy as np

from tabsae.pipeline import PipelineConfig, run_all


def _cfg(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        backend="mock",
        device="cpu",
        seed=0,
        out_dir=str(tmp_path),
        d_model=64,
        n_datasets=24,
        n_cols=8,
        n_context=128,
        n_query=64,
        d_sae=128,
        k=16,
        steps=200,
        gen_n=128,
        sgld_steps=80,
    )


def test_end_to_end(tmp_path):
    res = run_all(_cfg(tmp_path))

    # 1. SAE reconstructs
    assert res["sae_final"]["fve"] > 0.5

    # 2. concept alignment vs ground truth
    align = {r["concept"]: r for r in res["interp"]["alignment"] if "sae_auroc" in r}
    assert align["monotone"]["sae_auroc"] > 0.65
    assert max(r["sae_auroc"] for r in align.values()) > 0.95  # interaction/redundant ~1.0
    assert res["interp"]["purity_coverage"]["coverage"] >= 0.6

    # 3. causal validation (the moat): necessity & sufficiency beat random-latent controls
    nec = res["interp"]["causal"]["necessity"]
    assert nec["kl"] > nec["kl_control"]
    assert nec["delta_acc"] >= nec["delta_acc_control"] - 1e-6
    suf = res["interp"]["causal"]["sufficiency"]
    assert suf["effect"] > suf["effect_control"]

    # 4. interpretable, selective steering of generation
    gen = res["generate"]
    assert np.isfinite(gen["fidelity"]["shape_error"])
    assert gen["controllability"]["selectivity"] > 1.0
