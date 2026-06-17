"""Baselines the SAE must be compared against (per 'Are SAEs useful?', 2502.16681).

If a logistic-regression probe on RAW activations recovers a concept as well as the SAE,
the SAE adds no probing value — we report that honestly. The SAE's value is causal
steerability + monosemanticity, established in causal.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


def auroc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Sign-agnostic AUROC of a 1-D score predicting a boolean label."""
    from sklearn.metrics import roc_auc_score

    labels = labels.astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    a = roc_auc_score(labels, scores)
    return float(max(a, 1 - a))


def lr_probe_concept(raw_acts: np.ndarray, labels: np.ndarray, cv: int = 5) -> float:
    """Cross-validated AUROC of logistic regression on raw activations."""
    labels = labels.astype(int)
    pos = labels.sum()
    if pos < 2 or pos > len(labels) - 2:
        return float("nan")
    folds = min(cv, pos, len(labels) - pos)
    if folds < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    try:
        proba = cross_val_predict(clf, raw_acts, labels, cv=folds, method="predict_proba")[:, 1]
    except Exception:  # noqa
        return float("nan")
    return auroc_score(proba, labels)


def neuron_selectivity(raw_acts: np.ndarray, labels: np.ndarray) -> dict:
    """Best single raw neuron AUROC for the concept (the polysemantic-unit baseline)."""
    best_auroc, best_idx = 0.0, -1
    for d in range(raw_acts.shape[1]):
        a = auroc_score(raw_acts[:, d], labels)
        if not np.isnan(a) and a > best_auroc:
            best_auroc, best_idx = a, d
    return {"best_neuron": best_idx, "auroc": best_auroc}


def pca_directions(raw_acts: np.ndarray, n: int = 16) -> np.ndarray:
    n = min(n, raw_acts.shape[1], max(1, raw_acts.shape[0] - 1))
    pca = PCA(n_components=n)
    pca.fit(raw_acts)
    return pca.components_  # [n, dim]


def compare_to_baselines(raw_acts: np.ndarray, labels: np.ndarray, sae_auroc: float) -> dict:
    """One-stop comparison row for a concept."""
    return {
        "sae_auroc": sae_auroc,
        "lr_probe_auroc": lr_probe_concept(raw_acts, labels),
        "best_neuron_auroc": neuron_selectivity(raw_acts, labels)["auroc"],
    }
