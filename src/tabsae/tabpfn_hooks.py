"""Backends that expose a tabular foundation model's residual-stream activations.

Two interchangeable backends implement the same protocol:

* ``TabPFNBackend`` — wraps the real, frozen, open-weights TabPFN v2. Loads the model,
  finds the transformer, hooks the residual stream, and reshapes activations to tokens.
  Reshaping depends on TabPFN's internal tensor layout (discovery tasks D1-D3); a helper
  ``describe()`` prints shapes so the heuristic can be confirmed/adjusted.

* ``MockTFMBackend`` — a deterministic, CPU-only simulation of a tabular FM that performs
  *legitimate in-context inference primitives* (computed from data, not from labels):
  per-column monotone strength, interaction strength, covariate shift, redundancy and an
  irrelevance proxy, then mixes them into an overcomplete (superposed) activation. This
  lets the WHOLE pipeline (SAE -> alignment -> causal -> generation) run and be tested on
  CPU before the real TabPFN internals are wired. The science (paper) uses TabPFNBackend.

Both return :class:`ActivationBatch` objects and support activation *patching* (in token
space) — the primitive behind causal ablation and steering.
"""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np

from .types import ActivationBatch, SCMDataset
from .utils.logging import get_logger

log = get_logger(__name__)

# Order of the interpretable "primitive" statistics the mock computes per column.
STAT_NAMES = ["monotone", "interaction", "covariate_shift", "redundant", "irrelevant", "bias"]
MONO_IDX = STAT_NAMES.index("monotone")

PatchFn = Callable[[np.ndarray], np.ndarray]  # acts[n_tokens, dim] -> modified acts


# --------------------------------------------------------------------------------------
# Mock backend
# --------------------------------------------------------------------------------------
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order.astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_corr(_rankdata(a), _rankdata(b))


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / max(len(a), 1)
    cb = np.searchsorted(np.sort(b), grid, side="right") / max(len(b), 1)
    return float(np.max(np.abs(ca - cb)))


class MockTFMBackend:
    """Deterministic surrogate tabular FM for end-to-end CPU testing."""

    name = "mock"

    def __init__(self, d_model: int = 64, seed: int = 0, noise: float = 0.02):
        self.d_model = d_model
        self.n_stats = len(STAT_NAMES)
        self.noise = noise
        rng = np.random.default_rng(seed)
        # overcomplete mixing -> superposition (the thing SAEs must disentangle)
        self.W = rng.normal(0, 1, (d_model, self.n_stats)) / np.sqrt(self.n_stats)
        self.W_pinv = np.linalg.pinv(self.W)  # [n_stats, d_model] recover stats from activation
        self.layers = [0]

    # -- primitives -------------------------------------------------------------------
    def column_stats(self, ds: SCMDataset) -> np.ndarray:
        """[n_cols, n_stats] primitives computed from DATA only (never from labels)."""
        n_ctx = int(ds.meta.get("n_context", ds.n_rows // 2))
        Xc, yc = ds.X[:n_ctx], ds.y[:n_ctx]
        Xq = ds.X[n_ctx:]
        n_cols = ds.n_cols
        stats = np.zeros((n_cols, self.n_stats), dtype=np.float64)
        binar = [np.unique(Xc[:, j]).size <= 2 for j in range(n_cols)]
        for j in range(n_cols):
            mono = _spearman(Xc[:, j], yc.astype(np.float64))
            # interaction: best XOR partner among binary columns
            inter = 0.0
            if binar[j]:
                for k in range(n_cols):
                    if k == j or not binar[k]:
                        continue
                    xor = (Xc[:, j].astype(int) ^ Xc[:, k].astype(int)).astype(np.float64)
                    inter = max(inter, abs(_safe_corr(xor, yc.astype(np.float64))))
            shift = _ks_stat(Xc[:, j], Xq[:, j]) if Xq.shape[0] else 0.0
            redund = 0.0
            for k in range(n_cols):
                if k != j:
                    redund = max(redund, abs(_safe_corr(Xc[:, j], Xc[:, k])))
            irr = max(0.0, 1.0 - max(abs(mono), inter, redund))
            stats[j] = [mono, inter, shift, redund, irr, 1.0]
        return stats

    def column_activations(self, ds: SCMDataset, rng: np.random.Generator | None = None) -> np.ndarray:
        stats = self.column_stats(ds)
        acts = stats @ self.W.T  # [n_cols, d_model]
        if self.noise:
            seed = int(hashlib.md5(ds.dataset_id.encode()).hexdigest()[:8], 16)
            rng = rng or np.random.default_rng(seed)
            acts = acts + rng.normal(0, self.noise, acts.shape)
        return acts.astype(np.float32)

    # -- prediction (used for causal necessity/sufficiency) ---------------------------
    def predict_from_activations(self, col_acts: np.ndarray, ds: SCMDataset) -> np.ndarray:
        """Recover per-column monotone weight from activations, apply linear readout to
        query rows -> P(y=1). Because the readout reads ACTIVATIONS, patching them changes
        the prediction (enabling causal tests)."""
        n_ctx = int(ds.meta.get("n_context", ds.n_rows // 2))
        Xc = ds.X[:n_ctx]
        Xq = ds.X[n_ctx:]
        recovered = (self.W_pinv @ col_acts.T).T  # [n_cols, n_stats]
        w = recovered[:, MONO_IDX]  # signed monotone strength per column
        mu = Xc.mean(0)
        sd = Xc.std(0) + 1e-6
        z = (Xq - mu) / sd
        logit = z @ w
        return 1.0 / (1.0 + np.exp(-logit))

    # -- unified forward --------------------------------------------------------------
    def run(
        self,
        ds: SCMDataset,
        token_kind: str = "column",
        patches: dict[int, PatchFn] | None = None,
    ) -> tuple[np.ndarray, dict[int, ActivationBatch]]:
        """Single forward. Returns (P(y=1) over query rows, {layer: ActivationBatch}).

        patches maps layer -> fn applied to token-space activations (for causal/steer).
        """
        acts = self.column_activations(ds)
        if patches and 0 in patches:
            acts = np.asarray(patches[0](acts), dtype=np.float32)
        probs = self.predict_from_activations(acts, ds)
        if token_kind == "column":
            col_index = np.arange(ds.n_cols)
            is_query = np.zeros(ds.n_cols, dtype=bool)
            batch = ActivationBatch(acts, "column", col_index, is_query, ds.dataset_id, 0, "resid")
        elif token_kind == "query_item":
            # per query row embedding = column activations weighted by |feature value|
            n_ctx = int(ds.meta.get("n_context", ds.n_rows // 2))
            Xq = ds.X[n_ctx:]
            emb = np.abs(Xq) @ acts / max(ds.n_cols, 1)
            n = emb.shape[0]
            batch = ActivationBatch(
                emb.astype(np.float32), "query_item", -np.ones(n, int), np.ones(n, bool),
                ds.dataset_id, 0, "resid",
            )
        else:
            raise ValueError(f"unknown token_kind {token_kind!r}")
        return probs, {0: batch}

    def predict_proba(self, ds: SCMDataset) -> np.ndarray:
        return self.run(ds, token_kind="column")[0]


# --------------------------------------------------------------------------------------
# Real TabPFN backend
# --------------------------------------------------------------------------------------
class TabPFNBackend:
    """Wraps the real, frozen TabPFN v2. Activation reshape is best-effort (see D1-D3)."""

    name = "tabpfn"

    def __init__(self, task: str = "classification", device: str = "auto",
                 layers: list[int] | None = None, model_version: str = "v2.5",
                 deterministic: bool = True):
        self.task = task
        self.device = device
        self.layers = layers or [-1]
        self.model_version = model_version  # 'v2' | 'v2.5' | 'v2.6' | 'v3'
        # deterministic=True => single estimator, NO feature/class shuffle, no fingerprint,
        # so the residual-stream feature axis maps 1:1 to original columns (group g -> cols
        # [g*fpg:(g+1)*fpg]). Required for per-column interpretability. Default n_estimators=8
        # shuffles features per estimator and scrambles that mapping.
        self.deterministic = deterministic
        self._est = None
        self._model = None
        self._captured: dict[str, np.ndarray] = {}

    # -- loading / discovery ----------------------------------------------------------
    def load(self):
        import torch  # noqa
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        from .utils.device import get_device

        dev = get_device(self.device)
        Est = TabPFNClassifier if self.task == "classification" else TabPFNRegressor
        kw = dict(device=str(dev))
        if self.deterministic:
            kw.update(
                n_estimators=1,
                auto_scale_n_estimators=False,
                inference_config={
                    "FEATURE_SHIFT_METHOD": None,   # <- stop per-estimator feature permutation
                    "CLASS_SHIFT_METHOD": None,
                    "FINGERPRINT_FEATURE": False,   # <- no prepended fingerprint column
                    "POLYNOMIAL_FEATURES": "no",
                },
            )
        # create_default_for_version picks the right repo/license per version (v2's accept
        # page is currently broken upstream; v2.5/2.6/3 work). Fall back to plain ctor for v2.
        try:
            from tabpfn.constants import ModelVersion

            self._est = Est.create_default_for_version(ModelVersion(self.model_version), **kw)
        except Exception:  # noqa
            self._est = Est(**kw)
        log.info("Loaded TabPFN %s (%s, deterministic=%s) on %s",
                 self.task, self.model_version, self.deterministic, dev)
        return self

    def find_transformer(self):
        """Return the underlying torch transformer module (D1). Requires a prior fit:
        TabPFN exposes the model via the ``model_`` property only after .fit()."""
        import torch.nn as nn

        if self._est is None:
            self.load()
        model = None
        for attr in ("model_", "model"):
            try:
                cand = getattr(self._est, attr)
            except Exception:  # noqa  (model_ raises before fit)
                cand = None
            if isinstance(cand, nn.Module):
                model = cand
                break
        if model is not None:
            self._model = model
            n = sum(p.numel() for p in model.parameters())
            log.info("Transformer %s ~%.1fM params", type(model).__name__, n / 1e6)
        else:
            log.warning("TabPFN model not initialized yet — call after .fit() / via run().")
        return model

    def describe(self, ds: SCMDataset) -> dict:
        """Fit+predict on a dataset and log captured activation shapes (D2/D3)."""
        probs, acts = self._forward_capture(ds)
        shapes = {k: tuple(v.shape) for k, v in self._captured.items()}
        log.info("Captured raw activation shapes: %s", shapes)
        return shapes

    # -- forward ----------------------------------------------------------------------
    def _split(self, ds: SCMDataset):
        n_ctx = int(ds.meta.get("n_context", ds.n_rows // 2))
        return ds.X[:n_ctx], ds.y[:n_ctx], ds.X[n_ctx:]

    def _forward_capture(self, ds: SCMDataset, patches: dict[int, PatchFn] | None = None):
        """Fit on context, register hooks, run predict_proba on query.

        Capture hooks record each requested layer's residual stream. If `patches` is given,
        a patch hook MODIFIES that layer's output in token space (normalize -> SAE ablate/
        clamp -> denormalize, applied per feature-group token across all items, excluding the
        target token), so the intervention propagates to the prediction. This is the causal-
        patching primitive for the real backend (mirrors MockTFMBackend's token-space patch).
        Requires deterministic=True (n_estimators=1) so there is a single, identity-ordered pass.
        """
        import torch

        if self._est is None:
            self.load()
        Xc, yc, Xq = self._split(ds)
        self._est.fit(Xc, yc)
        self.find_transformer()  # model_ is available only after fit
        self._captured = {}
        handles = []
        blocks = self._transformer_blocks()

        def mk_capture(name: str):
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                self._captured[name] = t.detach().float().cpu().numpy()
            return hook

        def mk_patch(fn: PatchFn):
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                arr = t.detach().float().cpu().numpy()
                if arr.ndim == 4:  # [batch, items, feat(+target), emsize]
                    b, items, feat, emsize = arr.shape
                    ng = max(1, feat - 1)  # exclude trailing target token
                    flat = arr[:, :, :ng, :].reshape(-1, emsize)
                    arr[:, :, :ng, :] = fn(flat).reshape(b, items, ng, emsize)
                else:
                    arr = fn(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)
                new_t = torch.as_tensor(arr, dtype=t.dtype, device=t.device)
                return (new_t, *tuple(out[1:])) if isinstance(out, (tuple, list)) else new_t
            return hook

        for li in self.layers:
            handles.append(blocks[li].register_forward_hook(mk_capture(f"L{li}")))
        if patches:
            for li, fn in patches.items():
                handles.append(blocks[li].register_forward_hook(mk_patch(fn)))
        try:
            with torch.no_grad():
                proba = self._est.predict_proba(Xq)
        finally:
            for h in handles:
                h.remove()
        return np.asarray(proba), self._captured

    def _transformer_blocks(self) -> list:
        """Residual-stream blocks = the PerFeatureEncoderLayer stack (TabPFN v2/2.5/3)."""
        if self._model is None:
            self.find_transformer()
        te = getattr(self._model, "transformer_encoder", None)
        layers = getattr(te, "layers", None) if te is not None else None
        if layers is not None and len(layers) > 0:
            return list(layers)
        # fallback: any module whose class looks like an encoder layer
        blocks = [m for m in self._model.modules() if "encoderlayer" in type(m).__name__.lower()]
        return blocks or [self._model]

    def _features_per_group(self) -> int:
        return int(getattr(self._model, "features_per_group", 1) or 1)

    def _to_tokens(self, raw: np.ndarray, ds: SCMDataset, token_kind: str, layer: int) -> ActivationBatch:
        """Reshape a TabPFN encoder-layer output to tokens.

        TabPFN residual stream is [batch, items(samples+extras), feature_groups(+target), emsize].
        Features are grouped (features_per_group) and padded; the last feature index is the
        target token. 'column' tokens are per-feature-GROUP (averaged over all items); each
        group's representative original column = group_id * features_per_group.
        """
        arr = np.asarray(raw)
        if arr.ndim == 4:
            arr = arr[0]  # drop batch -> [items, feat, emsize]
        if arr.ndim == 2:  # already [tokens, emsize] (defensive)
            n = arr.shape[0]
            return ActivationBatch(arr.astype(np.float32), token_kind,
                                   -np.ones(n, int), np.zeros(n, bool), ds.dataset_id, layer, "resid")
        items, feat, dim = arr.shape
        fpg = self._features_per_group()
        if token_kind == "column":
            pooled = arr.mean(axis=0)  # [feat, emsize]; averaging over items avoids the
            n_groups = max(1, feat - 1)  # drop the trailing target token
            acts = pooled[:n_groups]
            col_index = np.clip(np.arange(n_groups) * fpg, 0, ds.n_cols - 1)
            is_query = np.zeros(n_groups, bool)
        else:  # query_item: pool over feature axis; take the last n_query items (best-effort)
            n_query = ds.n_rows - int(ds.meta.get("n_context", ds.n_rows // 2))
            pooled = arr.mean(axis=1)  # [items, emsize]
            acts = pooled[-n_query:]
            col_index = -np.ones(acts.shape[0], int)
            is_query = np.ones(acts.shape[0], bool)
        return ActivationBatch(acts.astype(np.float32), token_kind, col_index, is_query,
                               ds.dataset_id, layer, "resid")

    def run(self, ds: SCMDataset, token_kind: str = "column", patches=None):
        proba, captured = self._forward_capture(ds, patches=patches)
        probs = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        out = {}
        for name, raw in captured.items():
            li = int(name[1:])
            out[li] = self._to_tokens(raw, ds, token_kind, li)
        return probs, out

    def predict_proba(self, ds: SCMDataset) -> np.ndarray:
        return self.run(ds)[0]


def load_backend(name: str = "auto", **kw):
    """Factory. 'auto' tries TabPFN, falls back to mock with a warning."""
    if name == "mock":
        return MockTFMBackend(**{k: v for k, v in kw.items() if k in ("d_model", "seed", "noise")})
    tk = ("task", "device", "layers", "model_version", "deterministic")
    if name == "tabpfn":
        return TabPFNBackend(**{k: v for k, v in kw.items() if k in tk}).load()
    # auto
    try:
        import tabpfn  # noqa

        return TabPFNBackend(**{k: v for k, v in kw.items() if k in tk}).load()
    except Exception as e:  # noqa
        log.warning("TabPFN unavailable (%s); using MockTFMBackend.", e)
        return MockTFMBackend(**{k: v for k, v in kw.items() if k in ("d_model", "seed", "noise")})
