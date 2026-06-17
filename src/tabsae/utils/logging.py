"""Lightweight logging + config dumping so every result is traceable to a config."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

try:  # rich is nice but optional at import time
    from rich.logging import RichHandler

    _HANDLER: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
except Exception:  # noqa
    _HANDLER = logging.StreamHandler()
    _HANDLER.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

_CONFIGURED = False


def get_logger(name: str = "tabsae", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(level=level, handlers=[_HANDLER], force=True)
        _CONFIGURED = True
    return logging.getLogger(name)


def log_config(cfg: Any, run_dir: str) -> None:
    """Dump a resolved config (OmegaConf or dict) to <run_dir>/config.json."""
    os.makedirs(run_dir, exist_ok=True)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception:  # noqa
        pass
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
