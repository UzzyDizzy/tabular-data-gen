"""Drive a backend over a corpus, cache activations, and serve them for SAE training.

Extraction (forward passes) is decoupled from SAE training: shards on disk are read by
``ActivationDataset`` so SAE/interp/gen code runs on cached tensors (CPU-debuggable).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .scm.concepts import token_concept_labels
from .types import CONCEPTS, ConceptLabels, SCMDataset
from .utils.io import Manifest, load_acts, save_acts
from .utils.logging import get_logger

log = get_logger(__name__)


def extract_corpus_activations(
    backend,
    datasets: list[SCMDataset],
    out_dir: str,
    token_kind: str = "column",
) -> Manifest:
    """Run the backend over every dataset; write one shard per (dataset, layer, hook)."""
    man = Manifest(out_dir)
    for ds in datasets:
        _probs, batches = backend.run(ds, token_kind=token_kind)
        for layer, batch in batches.items():
            path = save_acts(out_dir, batch, ds.concept_labels)
            man.add(ds.dataset_id, layer, batch.hook, path, batch.acts.shape[0], batch.dim)
    man.save()
    log.info("Extracted %d datasets -> %d shards in %s", len(datasets), len(man.entries), out_dir)
    return man


class ActivationDataset(Dataset):
    """Streams cached activations for one (layer, hook). Holds metadata for interp."""

    def __init__(self, manifest: Manifest, layer: int, hook: str = "resid", normalize: str = "global"):
        self.normalize = normalize
        acts_list, col_idx_list, ds_ids, labels_by_ds = [], [], [], {}
        for shard in manifest.iter_shards(layer=layer, hook=hook):
            batch, labels = load_acts(shard)
            acts_list.append(batch.acts)
            col_idx_list.append(batch.col_index)
            ds_ids.extend([batch.dataset_id] * batch.acts.shape[0])
            if labels is not None:
                labels_by_ds[batch.dataset_id] = labels
        self.acts = np.concatenate(acts_list, axis=0).astype(np.float32)
        self.col_index = np.concatenate(col_idx_list, axis=0)
        self.dataset_ids = np.array(ds_ids)
        self.labels_by_dataset: dict[str, ConceptLabels] = labels_by_ds
        self.d_in = self.acts.shape[1]
        self._mean = np.zeros(self.d_in, np.float32)
        self._std = np.ones(self.d_in, np.float32)
        if normalize == "global":
            self._mean = self.acts.mean(0)
            self._std = self.acts.std(0) + 1e-6
        elif normalize == "per_dataset":
            self._apply_per_dataset_norm()
        self.norm_acts = (self.acts - self._mean) / self._std

    def _apply_per_dataset_norm(self) -> None:
        for ds in np.unique(self.dataset_ids):
            m = self.dataset_ids == ds
            self.acts[m] = (self.acts[m] - self.acts[m].mean(0)) / (self.acts[m].std(0) + 1e-6)

    def concept_label_matrix(self) -> np.ndarray:
        """[N_tokens, n_concepts] boolean ground-truth labels aligned with self.acts."""
        return token_concept_labels(self.col_index, self.dataset_ids, self.labels_by_dataset)

    @property
    def concepts(self) -> list[str]:
        return CONCEPTS

    def __len__(self) -> int:
        return self.acts.shape[0]

    def __getitem__(self, i: int):
        return torch.from_numpy(self.norm_acts[i])


def make_loaders(
    manifest: Manifest,
    layer: int,
    hook: str = "resid",
    batch_size: int = 4096,
    val_frac: float = 0.2,
    normalize: str = "global",
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, ActivationDataset]:
    """Train/val split that holds out whole DATASETS (tests transfer, not row memorization)."""
    full = ActivationDataset(manifest, layer, hook, normalize=normalize)
    ids = np.unique(full.dataset_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    is_val = np.array([d in val_ids for d in full.dataset_ids])
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    from torch.utils.data import Subset

    bs = min(batch_size, max(1, len(train_idx)))
    train_loader = DataLoader(Subset(full, train_idx.tolist()), batch_size=bs, shuffle=True)
    val_loader = DataLoader(Subset(full, val_idx.tolist()), batch_size=bs, shuffle=False)
    return train_loader, val_loader, full
