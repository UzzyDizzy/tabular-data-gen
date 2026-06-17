"""Interpretability: ground-truth concept alignment, baselines, causal validation."""
from .baselines import auroc_score, compare_to_baselines, lr_probe_concept, neuron_selectivity, pca_directions
from .causal import (
    datasets_with_concept,
    make_sae_patch,
    measure_necessity,
    measure_sufficiency,
)
from .concept_align import align_latents_to_concepts, latent_activations, purity_and_coverage

__all__ = [
    "auroc_score",
    "lr_probe_concept",
    "neuron_selectivity",
    "pca_directions",
    "compare_to_baselines",
    "latent_activations",
    "align_latents_to_concepts",
    "purity_and_coverage",
    "make_sae_patch",
    "measure_necessity",
    "measure_sufficiency",
    "datasets_with_concept",
]
