"""Activation cache + manifest round-trip."""
import numpy as np

from tabsae.types import ActivationBatch, ConceptLabels
from tabsae.utils.io import Manifest, load_acts, save_acts


def test_acts_roundtrip(tmp_path):
    batch = ActivationBatch(
        acts=np.random.rand(6, 8).astype(np.float32),
        token_kind="column",
        col_index=np.arange(6),
        is_query=np.zeros(6, bool),
        dataset_id="scm-00001",
        layer=0,
        hook="resid",
    )
    labels = ConceptLabels(n_cols=6, monotone=[True] + [False] * 5, irrelevant=[False] + [True] * 5,
                           covariate_shift=[False] * 6, interactions=[(0, 1, "xor")], redundant=[(2, 0)])
    path = save_acts(str(tmp_path), batch, labels)
    b2, l2 = load_acts(path)
    assert np.allclose(b2.acts, batch.acts)
    assert b2.dataset_id == "scm-00001"
    assert l2.monotone[0] is True
    assert l2.interactions[0] == (0, 1, "xor")


def test_manifest(tmp_path):
    man = Manifest(str(tmp_path))
    man.add("scm-0", 0, "resid", str(tmp_path / "a.npz"), 6, 8)
    man.add("scm-1", 0, "resid", str(tmp_path / "b.npz"), 6, 8)
    man.save()
    loaded = Manifest.load(str(tmp_path))
    assert len(loaded.entries) == 2
    shards = list(loaded.iter_shards(layer=0))
    assert len(shards) == 2
