"""Shared CLI helper: load a PipelineConfig from a YAML + dotlist overrides.

Usage in scripts:
    python scripts/run_all_smoke.py configs/smoke.yaml steps=200 backend=mock
"""
from __future__ import annotations

import os
import sys

from omegaconf import OmegaConf

from tabsae.pipeline import PipelineConfig


def load_pipeline_config(default: str = "configs/smoke.yaml") -> PipelineConfig:
    args = sys.argv[1:]
    cfg_path = default
    overrides = []
    for a in args:
        if a.endswith((".yaml", ".yml")):
            cfg_path = a
        elif "=" in a:
            overrides.append(a)
    base = OmegaConf.structured(PipelineConfig())
    if os.path.exists(cfg_path):
        base = OmegaConf.merge(base, OmegaConf.load(cfg_path))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    d = OmegaConf.to_container(base, resolve=True)
    return PipelineConfig(**d)
