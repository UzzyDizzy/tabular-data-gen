"""Synthetic structural-causal-model (SCM) data with KNOWN ground-truth structure.

This is the moat: because TabPFN's prior is SCM-based, we can generate probe tables whose
column roles (monotone / interaction / covariate-shift / irrelevant / redundant) are known
exactly, then score SAE features against them and validate causally.
"""
from .generators import (
    generate_corpus,
    make_controlled_dataset,
    render_dataset,
    sample_scm_spec,
)
from .concepts import column_concept_matrix, concept_indicator, token_concept_labels

__all__ = [
    "sample_scm_spec",
    "render_dataset",
    "make_controlled_dataset",
    "generate_corpus",
    "concept_indicator",
    "column_concept_matrix",
    "token_concept_labels",
]
