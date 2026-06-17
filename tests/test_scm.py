"""SCM ground-truth: labels must be derived exactly from the generative process."""
import numpy as np

from tabsae.scm.concepts import column_concept_matrix
from tabsae.scm.generators import generate_corpus, make_controlled_dataset, sample_scm_spec
from tabsae.types import CONCEPTS


def test_controlled_monotone_labels():
    ds = make_controlled_dataset("monotone", np.random.default_rng(0), n_cols=6)
    cl = ds.concept_labels
    assert cl.monotone[0] is True
    assert all(cl.irrelevant[j] for j in range(1, 6))
    assert "monotone" in cl.column_role(0)


def test_controlled_interaction_labels():
    ds = make_controlled_dataset("interaction", np.random.default_rng(0), n_cols=6)
    pairs = ds.concept_labels.interactions
    assert pairs and pairs[0][0] == 0 and pairs[0][1] == 1
    assert ds.col_types[0] == "cat" and ds.col_types[1] == "cat"


def test_corpus_labels_consistent():
    corpus = generate_corpus(8, np.random.default_rng(1), n_cols=8, n_context=64, n_query=32)
    assert len(corpus) == 8
    for ds in corpus:
        cm = column_concept_matrix(ds.concept_labels)
        assert cm.shape == (ds.n_cols, len(CONCEPTS))
        # every spec plants at least one monotone column
        assert cm[:, CONCEPTS.index("monotone")].any()
        # rows split into context + query
        assert ds.meta["n_context"] + ds.meta["n_query"] == ds.n_rows


def test_spec_partition_is_valid():
    spec = sample_scm_spec(np.random.default_rng(2), n_cols=10)
    used = set(spec["monotone_cols"]) | set(spec["irrelevant_cols"])
    for a, b, _ in spec["interaction_pairs"]:
        used |= {a, b}
    for d, s in spec["redundant_pairs"]:
        used |= {d, s}
    assert used.issubset(set(range(10)))
