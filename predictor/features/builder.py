"""Feature engineering at a cutoff (spec section 20).

``compute_features(history)`` receives a panel already truncated at the cutoff and returns
one row per item. Features are grouped so the ablation study can remove whole signal
families (frequency, recency, temporal dynamics, semantic, marks, question type,
co-occurrence, syllabus weight, question recurrence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.settings import Settings
from ..inference.bayes import fit_recurrence
from ..temporal import dynamics as dyn
from ..temporal.panel import FORMATS, Panel

FEATURE_GROUPS: dict[str, list[str]] = {
    "frequency": ["freq_all", "count_log"],
    "recency": ["last1", "freq_last3", "freq_last5", "ewma_short", "ewma_long", "linear_decay"],
    "temporal": ["since_last", "never_seen", "streak_present", "streak_absent", "mean_gap", "gap_cv",
                 "due_z", "hazard", "markov_next", "trend"],
    "semantic": ["soft_ewma", "soft_mean", "sem_ewma", "sem_neighbors"],
    "bayesian": ["bayes_mean", "bayes_width"],
    "history": ["hist_log", "base_rate"],
    "marks": ["marks_share", "marks_share_recent", "high_mark_rate"],
    "question_type": ["numerical_share", "derivation_share", "theory_share", "type_entropy"],
    "cooccurrence": ["cooc_lift_last", "unit_ewma"],
    "syllabus": ["hours_share", "marks_weight_share", "breadth"],
    "recurrence": ["exact_repeat_rate", "para_repeat_rate"],
}
FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
GROUP_LABELS = {
    "frequency": "Historical frequency", "recency": "Recency", "temporal": "Recurrence dynamics",
    "semantic": "Semantic similarity", "marks": "Marks weight", "question_type": "Question type",
    "cooccurrence": "Topic co-occurrence", "syllabus": "Syllabus weight", "recurrence": "Question recurrence",
    "bayesian": "Bayesian recurrence", "history": "History length",
}
# Scale-free features that mean the same thing in every course. The general ranking model is
# trained on these (simulated sequences plus other real courses) and the course-specific model
# is pulled toward its weights. Text, marks and syllabus features are course-specific.
GENERIC_FEATURES = ["freq_all", "last1", "freq_last3", "ewma_short", "ewma_long", "linear_decay", "since_last",
                    "never_seen", "streak_present", "streak_absent", "mean_gap", "gap_cv", "due_z", "hazard",
                    "markov_next", "trend", "cooc_lift_last", "unit_ewma", "bayes_mean", "bayes_width", "hist_log",
                    "base_rate"]


@dataclass
class FeatureMatrix:
    names: list[str]
    X: np.ndarray  # (K, F)
    n_history: int
    extras: dict = field(default_factory=dict)

    def column(self, name: str) -> np.ndarray:
        return self.X[:, self.names.index(name)]

    def as_dicts(self) -> list[dict[str, float]]:
        return [{n: float(v) for n, v in zip(self.names, row)} for row in self.X]


def compute_features(history: Panel, settings: Settings) -> FeatureMatrix:
    cfg = settings.temporal
    Y = history.Y
    T, K = Y.shape
    half_lives = sorted(float(h) for h in cfg.half_lives)
    short_h, long_h = half_lives[min(1, len(half_lives) - 1)], half_lives[-1]
    f: dict[str, np.ndarray] = {}

    post = fit_recurrence(Y, history.item_unit if history.item_unit is not None and len(history.item_unit) == K
                          else None)
    if T == 0:
        X = np.zeros((K, len(FEATURE_NAMES)))
        names = list(FEATURE_NAMES)
        X[:, names.index("never_seen")] = 1.0
        X[:, names.index("gap_cv")] = 1.0
        X[:, names.index("hazard")] = 0.0
        X[:, names.index("bayes_mean")] = post.mean
        X[:, names.index("bayes_width")] = post.high - post.low
        static = _static_features(history, K)
        for name, values in static.items():
            X[:, names.index(name)] = values
        return FeatureMatrix(names, X, 0, {"posterior": post})

    counts = Y.sum(axis=0)
    f["freq_all"] = counts / T
    f["count_log"] = np.log1p(counts)
    f["last1"] = Y[-1]
    f["freq_last3"] = Y[-3:].mean(axis=0)
    f["freq_last5"] = Y[-5:].mean(axis=0)
    f["ewma_short"] = dyn.ewma(Y, short_h)
    f["ewma_long"] = dyn.ewma(Y, long_h)
    f["linear_decay"] = dyn.linear_decay(Y)

    since = dyn.since_last(Y)
    seen = counts > 0
    f["since_last"] = np.minimum(since, 10.0) / 10.0
    f["never_seen"] = (~seen).astype(float)
    present, absent = dyn.streaks(Y)
    f["streak_present"] = np.minimum(present, 6.0) / 6.0
    f["streak_absent"] = np.minimum(absent, 10.0) / 10.0
    mean_gap, std_gap, n_gaps = dyn.gap_stats(Y)
    pooled_gap = np.nanmean(mean_gap) if np.isfinite(mean_gap).any() else max(T / 2.0, 1.0)
    mg = np.where(np.isfinite(mean_gap), mean_gap, np.where(seen, T / np.maximum(counts, 1), pooled_gap))
    f["mean_gap"] = np.minimum(mg, 10.0) / 10.0
    sd = np.where(np.isfinite(std_gap), std_gap, mg)
    f["gap_cv"] = np.clip(np.where(mg > 0, sd / mg, 1.0), 0.0, 3.0)
    # Overdue score: positive when the next exam would exceed the typical gap.
    next_gap = np.where(seen, since, T + 1)
    f["due_z"] = np.clip((next_gap - mg) / (sd + 1.0), -3.0, 3.0)
    haz = dyn.pooled_hazard(Y, float(cfg.hazard_prior_strength))
    f["hazard"] = haz.at(np.where(seen, since, 0), seen)
    n00, n01, n10, n11 = dyn.transition_counts(Y)
    m = float(cfg.rate_prior_strength)
    p01 = (n01.sum() + 1e-9) / (n00.sum() + n01.sum() + 2e-9)
    p11 = (n11.sum() + 1e-9) / (n10.sum() + n11.sum() + 2e-9)
    f["markov_next"] = np.where(Y[-1] > 0, (n11 + m * p11) / (n11 + n10 + m), (n01 + m * p01) / (n01 + n00 + m))
    f["trend"] = np.clip(dyn.trend_slope(Y), -0.5, 0.5)

    f["soft_ewma"] = dyn.ewma(history.soft, long_h)
    f["soft_mean"] = history.soft.mean(axis=0)
    sem = history.semantic_matrix()
    f["sem_ewma"] = dyn.ewma(sem, long_h)
    if history.item_sim is not None and history.item_sim.shape == (K, K):
        S = np.clip(history.item_sim, 0.0, None).copy()
        np.fill_diagonal(S, 0.0)
        rows = S.sum(axis=1)
        f["sem_neighbors"] = np.where(rows > 0, (S @ f["ewma_short"]) / np.maximum(rows, 1e-9), 0.0)
    else:
        f["sem_neighbors"] = np.zeros(K)
    f["bayes_mean"] = post.mean
    f["bayes_width"] = post.high - post.low
    f["hist_log"] = np.full(K, np.log1p(T))
    f["base_rate"] = np.full(K, float(Y.mean()))

    total = history.total_marks()[:, None]
    share = history.marks / total
    appeared = Y > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f["marks_share"] = np.where(counts > 0, (share * appeared).sum(axis=0) / np.maximum(counts, 1), 0.0)
    f["marks_share_recent"] = dyn.ewma(share, short_h)
    high = (share >= 0.1) & appeared
    f["high_mark_rate"] = np.where(counts > 0, high.sum(axis=0) / np.maximum(counts, 1), 0.0)

    fmt = history.formats.sum(axis=0)  # (K, F)
    fmt_total = fmt.sum(axis=1)
    safe = np.maximum(fmt_total, 1)
    f["numerical_share"] = fmt[:, FORMATS.index("numerical")] / safe
    f["derivation_share"] = fmt[:, FORMATS.index("derivation")] / safe
    f["theory_share"] = (fmt[:, FORMATS.index("theory")] + fmt[:, FORMATS.index("definition")]) / safe
    probs = fmt / safe[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    f["type_entropy"] = ent / np.log(len(FORMATS))

    lift = dyn.sequential_lift(Y, m)
    last_present = np.flatnonzero(Y[-1])
    if last_present.size:
        f["cooc_lift_last"] = np.log(np.clip(lift[last_present].mean(axis=0), 1e-3, None))
    else:
        f["cooc_lift_last"] = np.zeros(K)
    if history.item_unit is not None and len(history.item_unit) == K:
        units = history.item_unit
        unit_Y = np.zeros((T, int(units.max()) + 1 if K else 0))
        for k in range(K):
            unit_Y[:, units[k]] = np.maximum(unit_Y[:, units[k]], Y[:, k])
        f["unit_ewma"] = dyn.ewma(unit_Y, short_h)[units] if unit_Y.size else np.zeros(K)
    else:
        f["unit_ewma"] = f["ewma_short"]

    nq = history.n_questions.sum(axis=0)
    f["exact_repeat_rate"] = (history.exact_repeat.sum(axis=0) + 0.0) / np.maximum(nq, 1)
    f["para_repeat_rate"] = (history.para_repeat.sum(axis=0) + 0.0) / np.maximum(nq, 1)

    f.update(_static_features(history, K))
    X = np.column_stack([np.nan_to_num(f[name], nan=0.0, posinf=0.0, neginf=0.0) for name in FEATURE_NAMES])
    extras = {"since_last_raw": since, "counts": counts, "mean_gap_raw": mg, "std_gap_raw": sd,
              "n_gaps": n_gaps, "hazard_table_base": np.array([haz.base_rate]), "posterior": post}
    return FeatureMatrix(list(FEATURE_NAMES), X, T, extras)


def _static_features(history: Panel, K: int) -> dict[str, np.ndarray]:
    st = history.static
    out = {}
    for key in ("hours_share", "marks_weight_share", "breadth"):
        values = st.get(key)
        out[key] = np.asarray(values, dtype=float) if values is not None and len(values) == K else np.zeros(K)
    return out


class FeatureStore:
    """Caches features for every cutoff of one panel (each computed from ``panel.until(t)``)."""

    def __init__(self, panel: Panel, settings: Settings):
        self.panel = panel
        self.settings = settings
        self._cache: dict[int, FeatureMatrix] = {}

    def at(self, t: int) -> FeatureMatrix:
        if t not in self._cache:
            self._cache[t] = compute_features(self.panel.until(t), self.settings)
        return self._cache[t]

    def training_rows(self, t: int, min_history: int, columns: list[int] | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows (features at cutoff s, label Y[s]) for every s < t with at least ``min_history`` exams."""
        Xs, ys, groups = [], [], []
        for s in range(max(min_history, 1), t):
            fm = self.at(s)
            X = fm.X if columns is None else fm.X[:, columns]
            Xs.append(X)
            ys.append(self.panel.Y[s])
            groups.append(np.full(X.shape[0], s))
        if not Xs:
            n_cols = len(FEATURE_NAMES) if columns is None else len(columns)
            return np.zeros((0, n_cols)), np.zeros(0), np.zeros(0)
        return np.vstack(Xs), np.concatenate(ys), np.concatenate(groups)
