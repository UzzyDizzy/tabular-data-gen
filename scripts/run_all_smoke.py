"""Run the entire pipeline end-to-end (extract -> SAE -> interp -> generate).

    python scripts/run_all_smoke.py                 # uses configs/smoke.yaml
    python scripts/run_all_smoke.py configs/full.yaml
"""
from __future__ import annotations

import json

from _common import load_pipeline_config

from tabsae.pipeline import run_all


def main() -> None:
    cfg = load_pipeline_config("configs/smoke.yaml")
    res = run_all(cfg)
    interp = res["interp"]
    print("\n=== SUMMARY ===")
    print("SAE final:", res["sae_final"])
    print("purity/coverage:", interp["purity_coverage"])
    print("alignment:")
    for r in interp["alignment"]:
        print("  ", json.dumps(r))
    if interp.get("causal"):
        print("causal:", json.dumps(interp["causal"], default=float))
    if res.get("generate"):
        print("generate:", json.dumps(res["generate"], default=float))


if __name__ == "__main__":
    main()
