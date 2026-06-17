"""Concept-label utilities: turn ground-truth ConceptLabels into per-token label matrices
that line up with extracted column-token activations.
"""
from __future__ import annotations

import numpy as np

from ..types import CONCEPTS, ConceptLabels


def concept_indicator(labels: ConceptLabels, concept: str) -> np.ndarray:
    """Per-COLUMN boolean vector for one concept."""
    return labels.column_indicator(concept)


def column_concept_matrix(labels: ConceptLabels) -> np.ndarray:
    """[n_cols, n_concepts] boolean matrix (concept order = CONCEPTS)."""
    return np.stack([labels.column_indicator(c) for c in CONCEPTS], axis=1)


def token_concept_labels(
    col_index: np.ndarray,
    dataset_ids: np.ndarray,
    labels_by_dataset: dict[str, ConceptLabels],
) -> np.ndarray:
    """Build a [n_tokens, n_concepts] boolean label matrix for column-token activations.

    Each token belongs to (dataset_id, col_index); we look up that column's structural
    roles. Tokens with col_index < 0 (non-column tokens) get all-False rows.
    """
    n = col_index.shape[0]
    out = np.zeros((n, len(CONCEPTS)), dtype=bool)
    for i in range(n):
        j = int(col_index[i])
        ds = str(dataset_ids[i])
        if j < 0 or ds not in labels_by_dataset:
            continue
        cm = column_concept_matrix(labels_by_dataset[ds])  # [n_cols, n_concepts]
        if j < cm.shape[0]:
            out[i] = cm[j]
    return out
