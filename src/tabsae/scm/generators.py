"""SCM dataset sampler with controllable, LABELED structure.

Each generated table records exactly which columns play which structural role, so the
labels are derived from the generative process (exact, not inferred). Datasets are kept
within TabPFN's sweet spot (small rows/cols).

Roles
-----
- monotone   : numeric column with a monotone (linear) effect on the target.
- interaction: a pair of binary columns whose XOR drives the target (pure interaction).
- irrelevant : column independent of the target.
- redundant  : column that duplicates a (relevant) source column + tiny noise.
- covariate_shift: column whose distribution differs between context and query rows.
"""
from __future__ import annotations

import numpy as np

from ..types import ConceptLabels, SCMDataset


def _counts(rng: np.random.Generator, n_cols: int, frac: dict) -> dict:
    """Turn role fractions into integer counts that fit within n_cols."""
    n_mono = max(1, int(round(frac.get("monotone", 0.35) * n_cols)))
    n_int_pairs = int(round(frac.get("interaction", 0.15) * n_cols / 2))
    n_redund = int(round(frac.get("redundant", 0.10) * n_cols))
    # leave at least one irrelevant column
    while n_mono + 2 * n_int_pairs + n_redund > n_cols - 1:
        if n_int_pairs > 0:
            n_int_pairs -= 1
        elif n_redund > 0:
            n_redund -= 1
        elif n_mono > 1:
            n_mono -= 1
        else:
            break
    return dict(n_mono=n_mono, n_int_pairs=n_int_pairs, n_redund=n_redund)


def _sample_block_spec(
    rng: np.random.Generator,
    n_cols: int,
    block_size: int,
    n_classes: int,
    task: str,
    shift_frac: float,
    noise_std: float,
    positive_effects: bool,
) -> dict:
    """Assign one structural role to each contiguous block of `block_size` columns.

    With block_size == features_per_group, every TabPFN feature-token maps to columns of a
    SINGLE role -> sharp, uncontaminated concept labels.
    """
    n_blocks = max(1, n_cols // block_size)
    n_cols = n_blocks * block_size  # drop any remainder so groups align exactly
    roles = list(rng.choice(["monotone", "irrelevant", "interaction", "redundant"],
                            size=n_blocks, p=[0.4, 0.3, 0.15, 0.15]))
    if "monotone" not in roles:
        roles[0] = "monotone"  # need a source for redundancy + a relevant signal
    if "redundant" in roles and "monotone" not in roles:
        roles[roles.index("redundant")] = "monotone"

    col_types = ["num"] * n_cols
    monotone_cols, irrelevant_cols, interaction_pairs, redundant_pairs, shift_cols = [], [], [], [], []
    mono_w, int_w = {}, {}
    sign = (lambda: 1.0) if positive_effects else (lambda: float(rng.choice([-1, 1])))

    def block_cols(b):
        return list(range(b * block_size, (b + 1) * block_size))

    mono_source = []
    for b, role in enumerate(roles):
        cols = block_cols(b)
        if role == "monotone":
            for c in cols:
                monotone_cols.append(c)
                mono_w[c] = sign() * float(rng.uniform(0.8, 2.0))
            mono_source += cols
        elif role == "irrelevant":
            irrelevant_cols += cols
        elif role == "interaction":
            for c in cols:
                col_types[c] = "cat"
            for i in range(len(cols) - 1):  # chain pairs so every col is in an interaction
                interaction_pairs.append((cols[i], cols[i + 1], "xor"))
                int_w[len(interaction_pairs) - 1] = float(rng.uniform(1.0, 2.5))
            if len(cols) == 1:  # degenerate block_size=1: pair with itself-shifted is impossible
                irrelevant_cols += cols
    # redundant blocks duplicate a monotone source column
    for b, role in enumerate(roles):
        if role == "redundant":
            src_pool = mono_source or monotone_cols or [0]
            for c in block_cols(b):
                redundant_pairs.append((c, int(rng.choice(src_pool))))

    # covariate shift assigned at BLOCK granularity (keeps the shift label group-uniform too)
    n_shift_blocks = int(round(shift_frac * n_blocks))
    for b in rng.choice(n_blocks, size=min(n_shift_blocks, n_blocks), replace=False) if n_shift_blocks else []:
        shift_cols += block_cols(int(b))

    return dict(
        n_cols=n_cols, n_classes=n_classes, task=task, col_types=col_types,
        monotone_cols=[int(j) for j in monotone_cols],
        interaction_pairs=[(int(a), int(b), t) for a, b, t in interaction_pairs],
        irrelevant_cols=[int(j) for j in irrelevant_cols],
        redundant_pairs=[(int(a), int(b)) for a, b in redundant_pairs],
        shift_cols=sorted(set(int(j) for j in shift_cols)),
        mono_w=mono_w, int_w=int_w, noise_std=noise_std,
        shift_delta=float(rng.uniform(1.0, 2.0)),
    )


def sample_scm_spec(
    rng: np.random.Generator,
    n_cols: int = 8,
    n_classes: int = 2,
    task: str = "classification",
    frac: dict | None = None,
    shift_frac: float = 0.2,
    noise_std: float = 0.3,
    positive_effects: bool = False,
    block_size: int = 1,
) -> dict:
    """Sample a causal spec and RECORD every choice (so labels are exact).

    block_size > 1 assigns one role per contiguous block of columns (use = features_per_group
    for clean per-token labels on TabPFN).
    """
    if block_size > 1:
        return _sample_block_spec(rng, n_cols, block_size, n_classes, task, shift_frac,
                                  noise_std, positive_effects)
    frac = frac or {}
    c = _counts(rng, n_cols, frac)
    idx = list(rng.permutation(n_cols))
    cur = 0

    def take(k: int) -> list[int]:
        nonlocal cur
        out = idx[cur : cur + k]
        cur += k
        return out

    monotone_cols = take(c["n_mono"])
    interaction_cols = take(2 * c["n_int_pairs"])
    interaction_pairs = [
        (interaction_cols[2 * i], interaction_cols[2 * i + 1], "xor") for i in range(c["n_int_pairs"])
    ]
    redundant_cols = take(c["n_redund"])
    # each redundant col duplicates a randomly chosen monotone (relevant) source
    redundant_pairs = [(d, int(rng.choice(monotone_cols))) for d in redundant_cols] if monotone_cols else []
    irrelevant_cols = idx[cur:]  # everything left over

    # column types: interaction cols are binary 'cat'; redundant inherit source type (num here)
    col_types = ["num"] * n_cols
    for j in interaction_cols:
        col_types[j] = "cat"

    # covariate shift on a random subset of columns
    n_shift = int(round(shift_frac * n_cols))
    shift_cols = list(rng.choice(n_cols, size=min(n_shift, n_cols), replace=False)) if n_shift else []

    # effect weights (positive_effects keeps signs +1 so a single ReLU latent can capture
    # the monotone concept cleanly — used for the mock/smoke; realistic runs use random signs)
    sign = (lambda: 1.0) if positive_effects else (lambda: float(rng.choice([-1, 1])))
    mono_w = {j: sign() * float(rng.uniform(0.8, 2.0)) for j in monotone_cols}
    int_w = {i: float(rng.uniform(1.0, 2.5)) for i in range(len(interaction_pairs))}

    return dict(
        n_cols=n_cols,
        n_classes=n_classes,
        task=task,
        col_types=col_types,
        monotone_cols=[int(j) for j in monotone_cols],
        interaction_pairs=[(int(a), int(b), t) for a, b, t in interaction_pairs],
        irrelevant_cols=[int(j) for j in irrelevant_cols],
        redundant_pairs=[(int(a), int(b)) for a, b in redundant_pairs],
        shift_cols=[int(j) for j in shift_cols],
        mono_w=mono_w,
        int_w=int_w,
        noise_std=noise_std,
        shift_delta=float(rng.uniform(1.0, 2.0)),
    )


def labels_from_spec(spec: dict) -> ConceptLabels:
    """Derive exact ConceptLabels from a spec."""
    n = spec["n_cols"]
    mono = set(spec["monotone_cols"])
    irr = set(spec["irrelevant_cols"])
    shift = set(spec["shift_cols"])
    return ConceptLabels(
        n_cols=n,
        monotone=[j in mono for j in range(n)],
        irrelevant=[j in irr for j in range(n)],
        covariate_shift=[j in shift for j in range(n)],
        interactions=[tuple(t) for t in spec["interaction_pairs"]],
        redundant=[tuple(t) for t in spec["redundant_pairs"]],
    )


def _draw_columns(spec: dict, n_rows: int, rng: np.random.Generator, query: bool) -> np.ndarray:
    n = spec["n_cols"]
    X = np.zeros((n_rows, n), dtype=np.float64)
    shift = set(spec["shift_cols"]) if query else set()
    for j in range(n):
        if spec["col_types"][j] == "cat":
            p = 0.5 + (0.25 if j in shift else 0.0)
            X[:, j] = (rng.random(n_rows) < p).astype(np.float64)
        else:
            mu = spec["shift_delta"] if j in shift else 0.0
            X[:, j] = rng.normal(mu, 1.0, size=n_rows)
    # redundant duplicates (after base draw)
    for dup, src in spec["redundant_pairs"]:
        noise = rng.normal(0, 0.01, n_rows) if spec["col_types"][src] == "num" else 0.0
        X[:, dup] = X[:, src] + noise
    return X


def _target(spec: dict, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_rows = X.shape[0]
    logit = rng.normal(0, spec["noise_std"], n_rows)
    for j in spec["monotone_cols"]:
        logit += spec["mono_w"][j] * X[:, j]
    for i, (a, b, _t) in enumerate(spec["interaction_pairs"]):
        xor = (X[:, a].astype(int) ^ X[:, b].astype(int)).astype(np.float64)
        logit += spec["int_w"][i] * (2 * xor - 1)  # centered XOR
    if spec["task"] == "regression":
        return logit.astype(np.float64)
    if spec["n_classes"] <= 2:
        return (logit > 0.0).astype(np.int64)
    qs = np.quantile(logit, np.linspace(0, 1, spec["n_classes"] + 1)[1:-1])
    return np.digitize(logit, qs).astype(np.int64)


def render_dataset(
    spec: dict,
    n_context: int,
    n_query: int,
    rng: np.random.Generator,
    dataset_id: str = "",
) -> SCMDataset:
    """Realize an SCMDataset. Rows 0:n_context are context (train), the rest are query.

    Covariate shift is applied to query rows on shift_cols, so the model must adapt.
    """
    Xc = _draw_columns(spec, n_context, rng, query=False)
    Xq = _draw_columns(spec, n_query, rng, query=True)
    X = np.vstack([Xc, Xq])
    y = _target(spec, X, rng)
    meta = dict(spec=spec, n_context=n_context, n_query=n_query)
    return SCMDataset(
        X=X.astype(np.float32),
        y=y,
        col_types=list(spec["col_types"]),
        concept_labels=labels_from_spec(spec),
        task=spec["task"],
        dataset_id=dataset_id,
        meta=meta,
    )


def make_controlled_dataset(
    concept: str,
    rng: np.random.Generator,
    n_cols: int = 6,
    n_context: int = 256,
    n_query: int = 128,
    dataset_id: str = "",
) -> SCMDataset:
    """Generate a dataset that plants EXACTLY ONE clean instance of `concept`,
    everything else irrelevant. Used for the make-or-break causal test."""
    spec = dict(
        n_cols=n_cols,
        n_classes=2,
        task="classification",
        col_types=["num"] * n_cols,
        monotone_cols=[],
        interaction_pairs=[],
        irrelevant_cols=list(range(n_cols)),
        redundant_pairs=[],
        shift_cols=[],
        mono_w={},
        int_w={},
        noise_std=0.2,
        shift_delta=1.5,
    )
    if concept == "monotone":
        spec["monotone_cols"] = [0]
        spec["mono_w"] = {0: 2.0}
        spec["irrelevant_cols"] = list(range(1, n_cols))
    elif concept == "interaction":
        spec["col_types"][0] = spec["col_types"][1] = "cat"
        spec["interaction_pairs"] = [(0, 1, "xor")]
        spec["int_w"] = {0: 2.5}
        spec["irrelevant_cols"] = list(range(2, n_cols))
    elif concept == "redundant":
        spec["monotone_cols"] = [0]
        spec["mono_w"] = {0: 2.0}
        spec["redundant_pairs"] = [(1, 0)]
        spec["irrelevant_cols"] = list(range(2, n_cols))
    elif concept == "covariate_shift":
        spec["monotone_cols"] = [0]
        spec["mono_w"] = {0: 2.0}
        spec["shift_cols"] = [0]
        spec["irrelevant_cols"] = list(range(1, n_cols))
    elif concept == "irrelevant":
        pass  # all irrelevant; target is pure noise
    else:
        raise ValueError(f"unknown concept {concept!r}")
    return render_dataset(spec, n_context, n_query, rng, dataset_id=dataset_id or f"ctrl-{concept}")


def generate_corpus(
    n_datasets: int,
    rng: np.random.Generator,
    n_cols: int = 8,
    n_context: int = 256,
    n_query: int = 128,
    n_classes: int = 2,
    task: str = "classification",
    frac: dict | None = None,
    shift_frac: float = 0.2,
    positive_effects: bool = False,
    block_size: int = 1,
) -> list[SCMDataset]:
    """Produce N labeled datasets (mixed structure) for SAE training / evaluation."""
    out: list[SCMDataset] = []
    for i in range(n_datasets):
        spec = sample_scm_spec(
            rng, n_cols=n_cols, n_classes=n_classes, task=task, frac=frac, shift_frac=shift_frac,
            positive_effects=positive_effects, block_size=block_size,
        )
        out.append(render_dataset(spec, n_context, n_query, rng, dataset_id=f"scm-{i:05d}"))
    return out
