"""Mock backend: activations + prediction + patching primitives."""
import numpy as np

from tabsae.scm.generators import make_controlled_dataset
from tabsae.tabpfn_hooks import MockTFMBackend, load_backend


def test_activation_shapes_and_determinism():
    be = MockTFMBackend(d_model=32, seed=0)
    ds = make_controlled_dataset("monotone", np.random.default_rng(0), n_cols=6)
    probs, batches = be.run(ds, token_kind="column")
    batch = batches[0]
    assert batch.acts.shape == (6, 32)
    assert batch.token_kind == "column"
    # deterministic across calls
    a2 = be.column_activations(ds)
    assert np.allclose(batch.acts, a2)


def test_mock_predicts_monotone():
    be = MockTFMBackend(d_model=64, seed=0)
    ds = make_controlled_dataset("monotone", np.random.default_rng(0), n_cols=6,
                                 n_context=256, n_query=256)
    probs = be.predict_proba(ds)
    y = ds.y[ds.meta["n_context"]:]
    acc = np.mean((probs > 0.5).astype(int) == y)
    assert acc > 0.7  # mock recovers the monotone signal


def test_patch_changes_prediction():
    be = MockTFMBackend(d_model=64, seed=0)
    ds = make_controlled_dataset("monotone", np.random.default_rng(0), n_cols=6)
    base = be.run(ds)[0]
    zeroed = be.run(ds, patches={0: lambda a: a * 0.0})[0]
    assert not np.allclose(base, zeroed)  # patching the residual stream changes output


def test_auto_backend_loads_something():
    be = load_backend("mock", d_model=16)
    assert be.name == "mock"
