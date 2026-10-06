"""Supervised models trained on pooled (exam, topic) rows, plus the pooled hot/cold HMM.

Every learned model is gated by data-sufficiency rules (spec sections 22 and 68) and must
then beat the simpler models in the backtest to be selected. When a target early in the
backtest has too few training rows, the model falls back to recency-weighted frequency
for that target and says so (all models are compared on the same folds).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..features.builder import FEATURE_GROUPS, FEATURE_NAMES
from ..temporal import dynamics as dyn
from .base import BaseModel, ModelContext, ModelOutput

MIN_TRAIN_ROWS = 20
MIN_TRAIN_POSITIVES = 5


def _columns(exclude_groups: tuple[str, ...]) -> list[int]:
    excluded = {f for g in exclude_groups for f in FEATURE_GROUPS.get(g, [])}
    return [i for i, name in enumerate(FEATURE_NAMES) if name not in excluded]


def _fallback(t: int, ctx: ModelContext, reason: str) -> ModelOutput:
    Y = ctx.panel.until(t).Y
    h = float(ctx.settings.temporal.default_half_life)
    scores = dyn.ewma(Y, h) if Y.shape[0] else np.zeros(ctx.panel.K)
    return ModelOutput(scores, {"fallback": True, "fallback_reason": reason})


def _training_ok(y: np.ndarray) -> tuple[bool, str]:
    if len(y) < MIN_TRAIN_ROWS:
        return False, f"only {len(y)} training rows"
    pos = int(y.sum())
    if pos < MIN_TRAIN_POSITIVES or pos == len(y):
        return False, f"{pos} positive rows out of {len(y)}"
    return True, ""


class LogisticModel(BaseModel):
    family, complexity = "ml", 4
    description = ("L2-regularised logistic regression on the engineered features, trained on pooled "
                   "(exam, topic) rows from earlier exams only. Its weights give additive explanations.")

    def __init__(self, exclude_groups: tuple[str, ...] = (), name: str = "logistic", display: str = "Logistic regression"):
        self.exclude_groups = tuple(exclude_groups)
        self.name = name
        self.display = display
        self.hidden = bool(exclude_groups)
        self.cols = _columns(self.exclude_groups)

    def gate(self, panel, ctx):
        suff = ctx.settings.models.sufficiency
        need = int(suff.min_exams_logistic)
        if panel.T < need:
            return False, (f"Logistic regression needs at least {need} exams; you have {panel.T}. "
                           f"With fewer exams it would mostly fit noise.")
        _, y, _ = ctx.store.training_rows(panel.T, int(ctx.settings.models.min_history), self.cols)
        need_pos = int(suff.min_positive_rows_logistic)
        if y.sum() < need_pos:
            return False, f"Only {int(y.sum())} positive training rows (needs {need_pos})."
        return True, ""

    def fit(self, t: int, ctx: ModelContext):
        X, y, _ = ctx.store.training_rows(t, int(ctx.settings.models.min_history), self.cols)
        ok, why = _training_ok(y)
        if not ok:
            return None, None, why, len(y)
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(C=float(ctx.settings.models.logistic_C), max_iter=2000)
        clf.fit(scaler.transform(X), y)
        return scaler, clf, "", len(y)

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        scaler, clf, why, n_rows = self.fit(t, ctx)
        if clf is None:
            return _fallback(t, ctx, why)
        x = ctx.store.at(t).X[:, self.cols]
        z = scaler.transform(x)
        prob = clf.predict_proba(z)[:, 1]
        contrib = z * clf.coef_[0][None, :]
        names = [FEATURE_NAMES[i] for i in self.cols]
        info = {"train_rows": n_rows, "intercept": float(clf.intercept_[0]),
                "coefficients": {n: round(float(c), 4) for n, c in zip(names, clf.coef_[0])}}
        return ModelOutput(prob, info, contrib, names)


class RandomForestModel(BaseModel):
    name, display, family, complexity = "random_forest", "Random forest", "ml", 6
    description = "Bagged decision trees on the same features. Only used with enough exams and rows."

    def gate(self, panel, ctx):
        return _tree_gate(panel, ctx, "Random forest")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        X, y, _ = ctx.store.training_rows(t, int(ctx.settings.models.min_history))
        ok, why = _training_ok(y)
        if not ok:
            return _fallback(t, ctx, why)
        cfg = ctx.settings.models
        clf = RandomForestClassifier(n_estimators=int(cfg.forest_trees), min_samples_leaf=int(cfg.forest_min_leaf),
                                     random_state=ctx.seed, n_jobs=1)
        clf.fit(X, y)
        return ModelOutput(clf.predict_proba(ctx.store.at(t).X)[:, 1], {"train_rows": len(y)})


class GradientBoostingModel(BaseModel):
    name, display, family, complexity = "gradient_boosting", "Gradient boosting", "ml", 6
    description = "Shallow boosted trees (histogram gradient boosting). Only used with enough exams and rows."

    def gate(self, panel, ctx):
        return _tree_gate(panel, ctx, "Gradient boosting")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        X, y, _ = ctx.store.training_rows(t, int(ctx.settings.models.min_history))
        ok, why = _training_ok(y)
        if not ok:
            return _fallback(t, ctx, why)
        cfg = ctx.settings.models
        clf = HistGradientBoostingClassifier(max_iter=int(cfg.gbm_max_iter), learning_rate=float(cfg.gbm_learning_rate),
                                             max_depth=int(cfg.gbm_max_depth), l2_regularization=1.0,
                                             random_state=ctx.seed)
        clf.fit(X, y)
        return ModelOutput(clf.predict_proba(ctx.store.at(t).X)[:, 1], {"train_rows": len(y)})


def _tree_gate(panel, ctx, label: str) -> tuple[bool, str]:
    suff = ctx.settings.models.sufficiency
    need_exams, need_rows = int(suff.min_exams_trees), int(suff.min_rows_trees)
    if panel.T < need_exams:
        return False, f"{label} needs at least {need_exams} exams; you have {panel.T}."
    _, y, _ = ctx.store.training_rows(panel.T, int(ctx.settings.models.min_history))
    if len(y) < need_rows:
        return False, f"{label} needs at least {need_rows} training rows; the history gives {len(y)}."
    return True, ""


class PooledHMMModel(BaseModel):
    name, display, family, complexity = "hmm", "Pooled hot/cold HMM", "temporal", 6
    description = ("Two hidden states ('hot' and 'cold') shared by all topics, fitted with EM. Each topic's "
                   "next-exam probability comes from its filtered state.")

    def gate(self, panel, ctx):
        need = int(ctx.settings.models.sufficiency.min_exams_hmm)
        if panel.T < need:
            return False, (f"A hidden Markov model needs at least {need} exams to separate hidden states from "
                           f"noise; you have {panel.T}.")
        return True, ""

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        Y = ctx.panel.until(t).Y
        if Y.shape[0] < 3:
            return _fallback(t, ctx, "fewer than 3 exams of history")
        params = fit_pooled_hmm(Y, iterations=int(ctx.settings.models.hmm_iterations))
        return ModelOutput(hmm_next_probability(Y, *params), {"emission": params[2].round(3).tolist()})


def _forward(seq: np.ndarray, pi: np.ndarray, A: np.ndarray, e: np.ndarray):
    T = len(seq)
    alpha = np.zeros((T, 2))
    scale = np.zeros(T)
    emis = np.where(seq[:, None] > 0, e[None, :], 1 - e[None, :])
    alpha[0] = pi * emis[0]
    scale[0] = alpha[0].sum() or 1e-12
    alpha[0] /= scale[0]
    for i in range(1, T):
        alpha[i] = (alpha[i - 1] @ A) * emis[i]
        scale[i] = alpha[i].sum() or 1e-12
        alpha[i] /= scale[i]
    return alpha, scale, emis


def fit_pooled_hmm(Y: np.ndarray, iterations: int = 60, tol: float = 1e-5):
    pi = np.array([0.6, 0.4])
    A = np.array([[0.8, 0.2], [0.3, 0.7]])
    e = np.array([0.15, 0.75])
    prev_ll = -np.inf
    for _ in range(iterations):
        pi_acc = np.zeros(2)
        A_num = np.zeros((2, 2))
        e_num = np.zeros(2)
        e_den = np.zeros(2)
        ll = 0.0
        for k in range(Y.shape[1]):
            seq = Y[:, k]
            alpha, scale, emis = _forward(seq, pi, A, e)
            T = len(seq)
            beta = np.ones((T, 2))
            for i in range(T - 2, -1, -1):
                beta[i] = (A @ (emis[i + 1] * beta[i + 1])) / scale[i + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            for i in range(T - 1):
                xi = alpha[i][:, None] * A * (emis[i + 1] * beta[i + 1])[None, :] / scale[i + 1]
                A_num += xi / max(xi.sum(), 1e-12)
            pi_acc += gamma[0]
            e_num += (gamma * seq[:, None]).sum(axis=0)
            e_den += gamma.sum(axis=0)
            ll += np.log(scale).sum()
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
    out = np.zeros(Y.shape[1])
    for k in range(Y.shape[1]):
        alpha, _, _ = _forward(Y[:, k], pi, A, e)
        out[k] = float((alpha[-1] @ A) @ e)
    return out
