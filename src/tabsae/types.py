"""Shared data structures — one source of truth for tensors passed between modules.

Keeping shapes/semantics here prevents drift across scm/ -> tabpfn_hooks -> activations
-> sae -> interp -> generate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Canonical concept names (structural roles known a-priori for SCM data).
CONCEPTS = ["monotone", "irrelevant", "covariate_shift", "interaction", "redundant"]


@dataclass
class ConceptLabels:
    """Per-column / per-pair structural roles known a-priori from the SCM spec.

    Lists are per-column (length = n_cols). Pair concepts are lists of index tuples.
    These are the GROUND TRUTH every interpretability metric is scored against.
    """

    n_cols: int
    monotone: list[bool] = field(default_factory=list)
    irrelevant: list[bool] = field(default_factory=list)
    covariate_shift: list[bool] = field(default_factory=list)
    interactions: list[tuple[int, int, str]] = field(default_factory=list)  # (j, k, 'xor'|'and')
    redundant: list[tuple[int, int]] = field(default_factory=list)  # (j duplicates k)

    def column_role(self, j: int) -> set[str]:
        """Return the set of single-column roles that apply to column j."""
        roles: set[str] = set()
        if self.monotone and self.monotone[j]:
            roles.add("monotone")
        if self.irrelevant and self.irrelevant[j]:
            roles.add("irrelevant")
        if self.covariate_shift and self.covariate_shift[j]:
            roles.add("covariate_shift")
        if any(j in (a, b) for a, b, _ in self.interactions):
            roles.add("interaction")
        if any(j in (a, b) for a, b in self.redundant):
            roles.add("redundant")
        return roles

    def column_indicator(self, concept: str) -> np.ndarray:
        """Per-COLUMN boolean vector for `concept` (used to label column-token activations)."""
        return np.array([concept in self.column_role(j) for j in range(self.n_cols)], dtype=bool)


@dataclass
class SCMDataset:
    """One synthetic table + its ground-truth structure."""

    X: np.ndarray  # [n_rows, n_cols] float features (categoricals integer-encoded)
    y: np.ndarray  # [n_rows] target (int for classification)
    col_types: list[str]  # 'num' | 'cat' per column
    concept_labels: ConceptLabels
    task: str = "classification"  # or 'regression'
    dataset_id: str = ""
    meta: dict = field(default_factory=dict)  # scm graph, seed, functional forms, ...

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.X.shape[1])


@dataclass
class ActivationBatch:
    """Activations pulled from one TabPFN forward pass at one hook point."""

    acts: np.ndarray  # [n_tokens, dim]
    token_kind: str  # 'cell' | 'column' | 'query_item'
    col_index: np.ndarray  # [n_tokens] which column each token belongs to (-1 if n/a)
    is_query: np.ndarray  # [n_tokens] bool
    dataset_id: str
    layer: int
    hook: str

    def __post_init__(self) -> None:
        n = self.acts.shape[0]
        assert self.col_index.shape[0] == n, "col_index length mismatch"
        assert self.is_query.shape[0] == n, "is_query length mismatch"

    @property
    def dim(self) -> int:
        return int(self.acts.shape[1])


@dataclass
class SAEConfig:
    """Configuration for a single sparse autoencoder."""

    d_in: int
    d_sae: int
    variant: str = "topk"  # 'topk' | 'jumprelu' | 'matryoshka'
    k: int | None = 32  # topk / matryoshka
    l1: float | None = None  # (kept for relu variant / aux penalties)
    matryoshka_sizes: list[int] | None = None
    lr: float = 3e-4
    steps: int = 2000
    batch_size: int = 4096
    seed: int = 0
    aux_k: int = 0  # AuxK dead-latent revival; 0 disables
    normalize_decoder: bool = True
