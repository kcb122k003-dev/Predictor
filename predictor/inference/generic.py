"""General ranking model: what transfers between courses.

The model is a logistic regression on scale-free features (``GENERIC_FEATURES``: recency
rates, gaps, streaks, hazard, Markov persistence, Bayesian posterior, history length). It
never sees topic names or a course's own frequencies as identities, so it cannot carry one
course's topic counts into another course. It carries only patterns such as "a topic that
appeared in each of the last three papers is likely again" or "after one paper of history,
recency is a weak signal".

Two sources of knowledge:

1. ``generic_prior.json`` ships with the app. It is trained by ``train_simulated`` on
   simulated exam sequences drawn from broad families of examiner behaviour (stable rates,
   persistence and avoidance, rotations, trends, cool-downs, hot/cold phases) with random
   course sizes and history lengths. Simulated sequences are training data for this model
   only; they are never stored as exams and never count as any course's history.
2. Real courses in the local repository. After each analysis of a course that is not marked
   synthetic, its feature rows are stored (``course_feature_set``). The general model used
   for course C is the shipped prior updated by maximum a posteriori fitting on the rows of
   every other real course (never C itself).

Run ``python -m predictor.inference.generic`` to rebuild ``generic_prior.json``; it is
deterministic for a given seed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize

from ..config.settings import Settings, defaults
from ..features.builder import FEATURE_NAMES, GENERIC_FEATURES, compute_features
from ..temporal.panel import FORMATS, ExamInfo, Panel

VERSION = 1
PRIOR_FILE = Path(__file__).with_name("generic_prior.json")
DEFAULT_SEED = 20241
DEFAULT_COURSES = 400
# Prior precision (per standardised coefficient) when real courses update the simulated prior.
REPOSITORY_PRIOR_PRECISION = 5.0


def generic_columns() -> list[int]:
    return [FEATURE_NAMES.index(f) for f in GENERIC_FEATURES]


@dataclass
class GeneralModel:
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    source: str = "simulation"
    summary: dict[str, Any] = field(default_factory=dict)
    version: int = VERSION

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.standardize(X) @ self.coef + self.intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision(X), -30, 30)))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        return self.standardize(X) * self.coef[None, :]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "features": self.features, "mean": self.mean.round(6).tolist(),
                "std": self.std.round(6).tolist(), "coef": self.coef.round(6).tolist(),
                "intercept": round(float(self.intercept), 6), "source": self.source, "summary": self.summary}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneralModel":
        return cls(list(d["features"]), np.asarray(d["mean"], dtype=float), np.asarray(d["std"], dtype=float),
                   np.asarray(d["coef"], dtype=float), float(d["intercept"]), d.get("source", "simulation"),
                   dict(d.get("summary", {})), int(d.get("version", VERSION)))


# ------------------------------------------------------------------------------ MAP logistic
def fit_map_logistic(Z: np.ndarray, y: np.ndarray, prior_coef: np.ndarray, prior_intercept: float,
                     precision: float | np.ndarray, intercept_precision: float = 0.01,
                     sample_weight: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Logistic regression with a Gaussian prior centred on ``prior_coef`` (not on zero).

    With no rows the answer is the prior. With many rows the data dominate. ``precision`` is
    the inverse prior variance of each standardised coefficient.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = Z.shape
    if n == 0:
        return np.asarray(prior_coef, dtype=float).copy(), float(prior_intercept)
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    lam = np.broadcast_to(np.asarray(precision, dtype=float), (p,))
    x0 = np.concatenate([prior_coef, [prior_intercept]])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coef, b = theta[:p], theta[p]
        z = Z @ coef + b
        # log(1 + e^z) - y z, computed stably
        loss = np.sum(w * (np.logaddexp(0.0, z) - y * z))
        r = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35))) - y
        d = coef - prior_coef
        loss += 0.5 * float(np.sum(lam * d * d)) + 0.5 * intercept_precision * (b - prior_intercept) ** 2
        grad = np.concatenate([Z.T @ (w * r) + lam * d, [np.sum(w * r) + intercept_precision * (b - prior_intercept)]])
        return float(loss), grad

    res = optimize.minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x[:p].copy(), float(res.x[p])


# ------------------------------------------------------------------------------ simulation
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_panel(rng: np.random.Generator, K: int | None = None, T: int | None = None) -> Panel:
    """One simulated course: an exam x topic appearance matrix from a random examiner process."""
    K = int(K or rng.integers(8, 37))
    T = int(T or rng.integers(3, 25))
    U = int(rng.integers(2, 8))
    units = rng.integers(0, U, size=K)
    base = rng.uniform(0.15, 0.65)
    het = rng.uniform(0.3, 2.0)
    unit_eff = rng.normal(0.0, 0.5, size=U)
    logit0 = math.log(base / (1 - base)) + rng.normal(0.0, het, size=K) + unit_eff[units]
    mix = rng.dirichlet(np.full(6, 0.7))
    kind = rng.choice(6, size=K, p=mix)
    rho = rng.uniform(-2.0, 2.0, size=K)
    period = rng.choice([2, 3, 4], size=K)
    phase = rng.integers(0, 4, size=K)
    amp = rng.uniform(1.0, 3.0, size=K)
    slope = rng.normal(0.0, 0.25, size=K)
    cool = rng.uniform(1.0, 3.0, size=K)
    switch = rng.uniform(0.05, 0.3, size=K)
    hot_shift = rng.uniform(1.0, 2.5, size=K)
    hot = rng.random(K) < 0.5
    Y = np.zeros((T, K))
    absent = np.zeros(K)
    for t in range(T):
        lg = logit0.copy()
        last = Y[t - 1] if t else np.zeros(K)
        for k in range(K):
            if kind[k] == 1 and t:
                lg[k] += rho[k] * (1 if last[k] else -0.3)
            elif kind[k] == 2:
                lg[k] += amp[k] * (1.0 if (t + phase[k]) % period[k] == 0 else -1.0)
            elif kind[k] == 3:
                lg[k] += slope[k] * (t - T / 2.0)
            elif kind[k] == 4 and t:
                lg[k] += -cool[k] if last[k] else (0.5 * cool[k] if absent[k] >= 2 else 0.0)
            elif kind[k] == 5:
                if rng.random() < switch[k]:
                    hot[k] = not hot[k]
                lg[k] += hot_shift[k] if hot[k] else -hot_shift[k]
        Y[t] = (rng.random(K) < _sigmoid(lg)).astype(float)
        if Y[t].sum() == 0:
            Y[t, int(np.argmax(lg))] = 1.0
        absent = np.where(Y[t] > 0, 0, absent + 1)
    exams = [ExamInfo(t, float(t), f"S{t}", 100.0, None) for t in range(T)]
    zeros = np.zeros((T, K))
    return Panel(exams=exams, item_ids=list(range(K)), item_labels=[f"t{k}" for k in range(K)], Y=Y,
                 marks=zeros.copy(), soft=zeros.copy(), formats=np.zeros((T, K, len(FORMATS))),
                 n_questions=Y.copy(), exact_repeat=zeros.copy(), para_repeat=zeros.copy(), item_unit=units,
                 unit_ids=list(range(U)), layer="simulated")


def panel_rows(panel: Panel, settings: Settings, min_cutoff: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic feature rows for every cutoff s >= ``min_cutoff``: (X, y, cutoff)."""
    cols = generic_columns()
    Xs, ys, cs = [], [], []
    for s in range(max(1, min_cutoff), panel.T):
        fm = compute_features(panel.until(s), settings)
        Xs.append(fm.X[:, cols])
        ys.append(panel.Y[s])
        cs.append(np.full(panel.K, s))
    if not Xs:
        return np.zeros((0, len(cols))), np.zeros(0), np.zeros(0)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(cs)


def train_simulated(settings: Settings | None = None, n_courses: int = DEFAULT_COURSES,
                    seed: int = DEFAULT_SEED) -> GeneralModel:
    from sklearn.linear_model import LogisticRegression

    settings = settings or defaults()
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n_courses):
        X, y, _ = panel_rows(simulate_panel(rng), settings)
        Xs.append(X)
        ys.append(y)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit((X - mean) / std, y)
    return GeneralModel(list(GENERIC_FEATURES), mean, std, clf.coef_[0].copy(), float(clf.intercept_[0]),
                        "simulation", {"simulated_courses": n_courses, "rows": int(len(y)),
                                       "positive_rate": round(float(y.mean()), 4), "seed": seed,
                                       "real_courses": 0, "real_rows": 0})


@lru_cache(maxsize=1)
def load_prior() -> GeneralModel:
    """The shipped general model (trained on simulated sequences only)."""
    if PRIOR_FILE.exists():
        data = json.loads(PRIOR_FILE.read_text(encoding="utf-8"))
        if data.get("features") == GENERIC_FEATURES and int(data.get("version", 0)) == VERSION:
            return GeneralModel.from_dict(data)
    # Feature set changed or file missing: retrain a smaller prior on the fly (a few seconds).
    return train_simulated(n_courses=120)


def flat_prior() -> GeneralModel:
    """No simulated knowledge: zero weights at the generic base rate (the general component then reports itself
    unavailable, and the course logistic is centred on zero)."""
    base = load_prior()
    return GeneralModel(base.features, base.mean.copy(), base.std.copy(), np.zeros_like(base.coef),
                        float(np.log(0.4 / 0.6)), "none", {"simulated_courses": 0, "rows": 0, "real_courses": 0,
                                                           "real_rows": 0})


def prior_fingerprint() -> str:
    import hashlib

    data = PRIOR_FILE.read_bytes() if PRIOR_FILE.exists() else b"retrained"
    return f"{VERSION}:prior:{hashlib.sha256(data).hexdigest()[:16]}"


def update_with_rows(prior: GeneralModel, X: np.ndarray, y: np.ndarray, n_courses: int,
                     precision: float = REPOSITORY_PRIOR_PRECISION) -> GeneralModel:
    """Update the general model with real rows from other courses (MAP around the prior)."""
    if len(y) == 0 or n_courses == 0:
        return prior
    Z = prior.standardize(X)
    coef, b = fit_map_logistic(Z, y, prior.coef, prior.intercept, precision)
    summary = {**prior.summary, "real_courses": int(n_courses), "real_rows": int(len(y))}
    source = "simulation+repository" if prior.source != "none" else "repository"
    return GeneralModel(prior.features, prior.mean, prior.std, coef, b, source, summary)


def main() -> None:  # pragma: no cover - maintenance entry point
    model = train_simulated()
    PRIOR_FILE.write_text(json.dumps(model.to_dict(), indent=1), encoding="utf-8")
    print(f"Wrote {PRIOR_FILE} ({model.summary})")


if __name__ == "__main__":  # pragma: no cover
    main()
