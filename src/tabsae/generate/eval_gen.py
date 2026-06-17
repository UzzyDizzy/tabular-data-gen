"""Generation-quality metrics (TabSyn-style), implemented dependency-light.

- Shape error : mean per-column KS distance between real and synthetic marginals.
- Trend error : mean abs difference of pairwise correlation matrices.
- C2ST accuracy: classifier-two-sample-test accuracy (ideal ~0.5 = indistinguishable).

For the full paper, swap/augment with sdmetrics/synthcity (optional extras) to get
alpha-precision/beta-recall, MLE, DCR. This module keeps the smoke pipeline self-contained.
"""
from __future__ import annotations

import numpy as np


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / max(len(a), 1)
    cb = np.searchsorted(np.sort(b), grid, side="right") / max(len(b), 1)
    return float(np.max(np.abs(ca - cb)))


def _c2st(real: np.ndarray, synth: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.vstack([real, synth])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])
    n = min(len(real), len(synth))
    if n < 5:
        return float("nan")
    folds = min(5, n)
    try:
        scores = cross_val_score(LogisticRegression(max_iter=500), X, y, cv=folds, scoring="accuracy")
    except Exception:  # noqa
        return float("nan")
    return float(scores.mean())


def fidelity_suite(real: np.ndarray, synth: np.ndarray) -> dict:
    real = np.asarray(real, dtype=np.float64)
    synth = np.asarray(synth, dtype=np.float64)
    n_cols = real.shape[1]
    shape = float(np.mean([_ks(real[:, j], synth[:, j]) for j in range(n_cols)]))
    cr = np.corrcoef(real, rowvar=False)
    cs = np.corrcoef(synth, rowvar=False)
    iu = np.triu_indices(n_cols, k=1)
    trend = float(np.mean(np.abs(np.nan_to_num(cr[iu]) - np.nan_to_num(cs[iu])))) if n_cols > 1 else 0.0
    return {"shape_error": shape, "trend_error": trend, "c2st_acc": _c2st(real, synth)}
