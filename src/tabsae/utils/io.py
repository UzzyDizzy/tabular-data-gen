"""Sharded activation cache + (de)serialization of ActivationBatch and SCMDataset.

Design: one shard = one .npz file holding the activation matrix + token metadata for
one (dataset, layer, hook). A manifest.json indexes shards and carries concept labels,
so SAE/interp code can stream activations without re-running TabPFN. Keeps RAM flat on
the server and lets CPU dev load a single tiny shard.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Iterator

import numpy as np

from ..types import ActivationBatch, ConceptLabels


def _shard_name(dataset_id: str, layer: int, hook: str) -> str:
    safe = f"{dataset_id}__L{layer}__{hook}".replace("/", "-").replace(" ", "_")
    return safe + ".npz"


def save_acts(out_dir: str, batch: ActivationBatch, concept_labels: ConceptLabels | None = None) -> str:
    """Write one ActivationBatch to a shard; return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _shard_name(batch.dataset_id, batch.layer, batch.hook))
    payload = dict(
        acts=batch.acts.astype(np.float32),
        col_index=batch.col_index.astype(np.int64),
        is_query=batch.is_query.astype(bool),
        token_kind=np.array(batch.token_kind),
        dataset_id=np.array(batch.dataset_id),
        layer=np.array(batch.layer),
        hook=np.array(batch.hook),
    )
    if concept_labels is not None:
        payload["concept_labels"] = np.array(json.dumps(_labels_to_dict(concept_labels)))
    np.savez_compressed(path, **payload)
    return path


def load_acts(path: str) -> tuple[ActivationBatch, ConceptLabels | None]:
    z = np.load(path, allow_pickle=False)
    batch = ActivationBatch(
        acts=z["acts"],
        token_kind=str(z["token_kind"]),
        col_index=z["col_index"],
        is_query=z["is_query"],
        dataset_id=str(z["dataset_id"]),
        layer=int(z["layer"]),
        hook=str(z["hook"]),
    )
    labels = None
    if "concept_labels" in z.files:
        labels = _labels_from_dict(json.loads(str(z["concept_labels"])))
    return batch, labels


def _labels_to_dict(cl: ConceptLabels) -> dict:
    d = asdict(cl)
    # tuples -> lists for json
    d["interactions"] = [list(t) for t in cl.interactions]
    d["redundant"] = [list(t) for t in cl.redundant]
    return d


def _labels_from_dict(d: dict) -> ConceptLabels:
    return ConceptLabels(
        n_cols=d["n_cols"],
        monotone=d.get("monotone", []),
        irrelevant=d.get("irrelevant", []),
        covariate_shift=d.get("covariate_shift", []),
        interactions=[tuple(t) for t in d.get("interactions", [])],
        redundant=[tuple(t) for t in d.get("redundant", [])],
    )


class Manifest:
    """Index of shards for a corpus extraction run."""

    def __init__(self, root: str):
        self.root = root
        self.entries: list[dict] = []

    def add(self, dataset_id: str, layer: int, hook: str, shard_path: str, n_tokens: int, dim: int) -> None:
        self.entries.append(
            dict(
                dataset_id=dataset_id,
                layer=layer,
                hook=hook,
                shard=os.path.relpath(shard_path, self.root),
                n_tokens=n_tokens,
                dim=dim,
            )
        )

    def save(self) -> str:
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"root": self.root, "entries": self.entries}, f, indent=2)
        return path

    @classmethod
    def load(cls, root: str) -> "Manifest":
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as f:
            data = json.load(f)
        m = cls(root)
        m.entries = data["entries"]
        return m

    def iter_shards(self, layer: int | None = None, hook: str | None = None) -> Iterator[str]:
        for e in self.entries:
            if layer is not None and e["layer"] != layer:
                continue
            if hook is not None and e["hook"] != hook:
                continue
            yield os.path.join(self.root, e["shard"])
