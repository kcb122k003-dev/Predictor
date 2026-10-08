"""Richer supervised and latent-state models: random forest, gradient boosting, pooled HMM.

None of these has a minimum number of exams. They run on every backtest fold and their
influence is decided by the evidence-aware ensemble: their reliability prior is
exams / (exams + df) with a large df (many parameters estimated from one course), and their
weight grows only when earlier folds show they predict well. With few papers they are
therefore present but carry almost no weight; with many papers they can earn it.

The only fallback is mathematical: a classifier cannot be fitted when every training label
is the same (for example one earlier paper). Then the general ranking model's score is used
for that fold and the fold is reported.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .base import BaseModel, ModelContext, ModelOutput
from .components import general_fallback, has_both_classes


def _rows(t: int, ctx: ModelContext):
    return ctx.store.training_rows(t, max(1, int(ctx.settings.models.min_history)))


class RandomForestModel(BaseModel):
    name, display, family, complexity, role = "random_forest", "Random forest", "ml", 6, "component"
    df = 40.0
    description = "Bagged decision trees on all features. Earns ensemble weight only when backtests support it."

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        X, y, _ = _rows(t, ctx)
        if not has_both_classes(y):
            return general_fallback(t, ctx, "training rows contain a single class")
        cfg = ctx.settings.models
        clf = RandomForestClassifier(n_estimators=int(cfg.forest_trees), min_samples_leaf=int(cfg.forest_min_leaf),
                                     random_state=ctx.seed, n_jobs=1)
        clf.fit(X, y)
        return ModelOutput(clf.predict_proba(ctx.store.at(t).X)[:, 1], {"train_rows": len(y)})


class GradientBoostingModel(BaseModel):
    name, display, family, complexity, role = "gradient_boosting", "Gradient boosting", "ml", 6, "component"
    df = 40.0
    description = "Shallow boosted trees (histogram gradient boosting). Earns ensemble weight only when backtests support it."

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        X, y, _ = _rows(t, ctx)
        if not has_both_classes(y):
            return general_fallback(t, ctx, "training rows contain a single class")
        cfg = ctx.settings.models
        clf = HistGradientBoostingClassifier(max_iter=int(cfg.gbm_max_iter), learning_rate=float(cfg.gbm_learning_rate),
                                             max_depth=int(cfg.gbm_max_depth), l2_regularization=1.0,
                                             random_state=ctx.seed)
        clf.fit(X, y)
        return ModelOutput(clf.predict_proba(ctx.store.at(t).X)[:, 1], {"train_rows": len(y)})


class PooledHMMModel(BaseModel):
    name, display, family, complexity, role = "hmm", "Pooled hot/cold HMM", "temporal", 6, "component"
    df = 5.0
    description = ("Two hidden states ('hot' and 'cold') shared by all topics, fitted with EM (smoothed, so it runs "
                   "on short histories). Each topic's next-exam probability comes from its filtered state.")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        Y = ctx.panel.until(t).Y
        if Y.shape[0] == 0:
            return ModelOutput(np.full(ctx.panel.K, 0.5), {"available": False,
                                                           "unavailable_reason": "No earlier paper to filter on."})
        params = fit_pooled_hmm(Y, iterations=int(ctx.settings.models.hmm_iterations))
        return ModelOutput(hmm_next_probability(Y, *params), {"emission": params[2].round(3).tolist()})


def _emissions(Y: np.ndarray, e: np.ndarray) -> np.ndarray:
    """(K, T, 2) emission likelihoods for all topics at once."""
    seq = Y.T[:, :, None] > 0
    return np.where(seq, e[None, None, :], 1 - e[None, None, :])


def _forward_all(Y: np.ndarray, pi: np.ndarray, A: np.ndarray, e: np.ndarray):
    """Scaled forward pass for every topic (topics share the parameters), vectorised over topics."""
    K, T = Y.shape[1], Y.shape[0]
    emis = _emissions(Y, e)
    alpha = np.zeros((K, T, 2))
    scale = np.zeros((K, T))
    alpha[:, 0] = pi[None, :] * emis[:, 0]
    scale[:, 0] = np.maximum(alpha[:, 0].sum(axis=1), 1e-12)
    alpha[:, 0] /= scale[:, 0, None]
    for i in range(1, T):
        alpha[:, i] = (alpha[:, i - 1] @ A) * emis[:, i]
        scale[:, i] = np.maximum(alpha[:, i].sum(axis=1), 1e-12)
        alpha[:, i] /= scale[:, i, None]
    return alpha, scale, emis


def fit_pooled_hmm(Y: np.ndarray, iterations: int = 60, tol: float = 1e-5):
    pi = np.array([0.6, 0.4])
    A = np.array([[0.8, 0.2], [0.3, 0.7]])
    e = np.array([0.15, 0.75])
    prev_ll = -np.inf
    T = Y.shape[0]
    for _ in range(iterations):
        alpha, scale, emis = _forward_all(Y, pi, A, e)
        K = alpha.shape[0]
        beta = np.ones((K, T, 2))
        for i in range(T - 2, -1, -1):
            beta[:, i] = ((emis[:, i + 1] * beta[:, i + 1]) @ A.T) / scale[:, i + 1, None]
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=2, keepdims=True), 1e-300)
        A_num = np.zeros((2, 2))
        for i in range(T - 1):
            xi = alpha[:, i, :, None] * A[None] * (emis[:, i + 1] * beta[:, i + 1])[:, None, :] / scale[:, i + 1, None, None]
            A_num += (xi / np.maximum(xi.sum(axis=(1, 2), keepdims=True), 1e-12)).sum(axis=0)
        pi_acc = gamma[:, 0].sum(axis=0)
        seq = Y.T[:, :, None]
        e_num = (gamma * seq).sum(axis=(0, 1))
        e_den = gamma.sum(axis=(0, 1))
        ll = float(np.log(scale).sum())
        pi = (pi_acc + 1) / (pi_acc.sum() + 2)
        A = (A_num + 1) / (A_num.sum(axis=1, keepdims=True) + 2)
        e = np.clip((e_num + 0.5) / (e_den + 1), 0.01, 0.99)
        if e[0] > e[1]:  # keep state 1 as the "hot" state
            e, pi, A = e[::-1].copy(), pi[::-1].copy(), A[::-1, ::-1].copy()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return pi, A, e


def hmm_next_probability(Y: np.ndarray, pi: np.ndarray, A: np.ndarray, e: np.ndarray) -> np.ndarray:
    alpha, _, _ = _forward_all(Y, pi, A, e)
    return (alpha[:, -1] @ A) @ e
