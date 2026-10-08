"""Probability calibration, applied only when held-out papers show it works.

Scores become probabilities through a calibrator on the *percentile rank* of each topic
within its exam, fitted only on out-of-sample backtest predictions. Two calibrators compete:
Platt scaling (two parameters) and isotonic regression (monotone, more flexible). For every
held-out paper t, each calibrator is fitted on the papers before t and scored on t, so both
the choice and the reported quality are out-of-sample.

Probabilities are labelled as validated only when the nested Brier score beats the base
rate (the share of topics that appeared in earlier papers) by more than one standard error
of the per-paper difference. There is no fixed row count: with few papers the standard error
is large and the ranking is shown as relative scores with an evidence band instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..config.settings import Settings
from ..evaluation.metrics import brier, log_loss, mean_and_se, reliability
from .base import percentile_rank

EPS = 0.01


@dataclass
class Platt:
    a: float
    b: float
    name: str = "platt"

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(x, dtype=float) + self.b)))

    def params(self) -> dict:
        return {"a": round(self.a, 4), "b": round(self.b, 4)}


@dataclass
class Isotonic:
    model: IsotonicRegression
    name: str = "isotonic"

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(np.asarray(x, dtype=float)), EPS, 1 - EPS)

    def params(self) -> dict:
        return {"knots": int(len(self.model.X_thresholds_))}


def fit_platt(x: np.ndarray, y: np.ndarray, C: float = 100.0) -> Platt:
    """Two-parameter logistic fit by Newton's method (same objective as scikit-learn with C=100:
    a light L2 penalty on the slope, none on the intercept)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a, b = 0.0, float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    lam = 1.0 / C
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        w = p * (1 - p)
        g = np.array([np.sum((p - y) * x) + lam * a, np.sum(p - y)])
        H = np.array([[np.sum(w * x * x) + lam, np.sum(w * x)], [np.sum(w * x), np.sum(w) + 1e-9]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-9:
            break
    return Platt(float(a), float(b))


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> Isotonic:
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip", increasing=True)
    iso.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return Isotonic(iso)


FITTERS = {"platt": fit_platt, "isotonic": fit_isotonic}


@dataclass
class CalibrationReport:
    valid: bool
    reason: str
    method: str = "platt"
    n_rows: int = 0
    n_positive: int = 0
    nested_folds: int = 0
    nested_brier: float = math.nan
    climatology_brier: float = math.nan
    brier_skill: float = math.nan
    brier_gain: float = math.nan
    brier_gain_se: float = math.nan
    nested_log_loss: float = math.nan
    ece: float = math.nan
    reliability: list[dict] = field(default_factory=list)
    comparison: dict = field(default_factory=dict)
    final: object | None = None
    bootstrap: list = field(default_factory=list)
    band: tuple[float, float] = (0.10, 0.90)

    def probabilities(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self.final is None:
            return None
        x = percentile_rank(scores)
        p = self.final.predict(x)
        if self.bootstrap:
            sims = np.vstack([c.predict(x) for c in self.bootstrap])
            low, high = np.percentile(sims, [100 * self.band[0], 100 * self.band[1]], axis=0)
            low, high = np.minimum(low, p), np.maximum(high, p)
        else:
            low, high = p, p
        return p, low, high

    def as_dict(self) -> dict:
        return {
            "valid": self.valid, "reason": self.reason, "method": self.method, "rows": self.n_rows,
            "positives": self.n_positive, "nested_folds": self.nested_folds,
            "nested_brier": _r(self.nested_brier), "climatology_brier": _r(self.climatology_brier),
            "brier_skill": _r(self.brier_skill), "brier_gain": _r(self.brier_gain), "brier_gain_se": _r(self.brier_gain_se),
            "nested_log_loss": _r(self.nested_log_loss), "ece": _r(self.ece), "reliability": self.reliability,
            "comparison": self.comparison, "params": self.final.params() if self.final is not None else None,
        }


def _r(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def _both(y: np.ndarray) -> bool:
    return len(y) > 0 and 0 < float(y.sum()) < len(y)


def calibrate(predictions: dict[int, np.ndarray], Y: np.ndarray, targets: list[int], settings: Settings,
              seed: int = 13) -> CalibrationReport:
    cfg = settings.calibration
    methods = [m for m in cfg.methods if m in FITTERS] or ["platt"]
    xs = {t: percentile_rank(predictions[t]) for t in targets if t in predictions}
    ys = {t: Y[t] for t in xs}
    ordered = sorted(xs)
    all_x = np.concatenate([xs[t] for t in ordered]) if ordered else np.zeros(0)
    all_y = np.concatenate([ys[t] for t in ordered]) if ordered else np.zeros(0)
    report = CalibrationReport(valid=False, reason="", n_rows=int(len(all_y)), n_positive=int(all_y.sum()),
                               band=(float(cfg.band_low), float(cfg.band_high)))
    if not ordered:
        report.reason = ("No held-out paper exists yet, so probabilities cannot be checked. The ranking is shown as "
                         "relative scores with an evidence band.")
        return report

    # Nested evaluation: calibrator for paper t is fitted on held-out papers before t.
    nested: dict[str, dict[int, np.ndarray]] = {m: {} for m in methods}
    clim: dict[int, np.ndarray] = {}
    for i, t in enumerate(ordered):
        prev = ordered[:i]
        if not prev:
            continue
        px = np.concatenate([xs[s] for s in prev])
        py = np.concatenate([ys[s] for s in prev])
        if not _both(py):
            continue
        clim[t] = np.full(len(ys[t]), py.mean())
        for m in methods:
            nested[m][t] = FITTERS[m](px, py).predict(xs[t])
    folds = sorted(clim)
    report.nested_folds = len(folds)
    if folds:
        losses = {m: log_loss(np.concatenate([nested[m][t] for t in folds]), np.concatenate([ys[t] for t in folds]))
                  for m in methods}
        report.method = min(methods, key=lambda m: (losses[m], methods.index(m)))
        report.comparison = {m: {"nested_log_loss": _r(losses[m])} for m in methods}
        p = np.concatenate([nested[report.method][t] for t in folds])
        y = np.concatenate([ys[t] for t in folds])
        c = np.concatenate([clim[t] for t in folds])
        report.nested_brier = brier(p, y)
        report.climatology_brier = brier(c, y)
        report.brier_skill = 1.0 - report.nested_brier / report.climatology_brier if report.climatology_brier else math.nan
        report.nested_log_loss = losses[report.method]
        report.reliability, report.ece = reliability(p, y, int(cfg.reliability_bins))
        gains = [brier(clim[t], ys[t]) - brier(nested[report.method][t], ys[t]) for t in folds]
        report.brier_gain, report.brier_gain_se = mean_and_se(gains)

    if _both(all_y):
        report.final = FITTERS[report.method](all_x, all_y)
        rng = np.random.default_rng(seed)
        for _ in range(int(cfg.bootstrap_samples)):
            pick = rng.choice(ordered, size=len(ordered), replace=True)
            bx = np.concatenate([xs[s] for s in pick])
            by = np.concatenate([ys[s] for s in pick])
            if _both(by):
                report.bootstrap.append(FITTERS[report.method](bx, by))

    n = report.nested_folds
    if n == 0:
        report.reason = (f"{len(ordered)} held-out paper(s) is not enough to fit a calibrator on earlier papers and test "
                         f"it on a later one. Scores are relative, shown with an evidence band.")
    elif math.isnan(report.brier_gain_se) or not (report.brier_gain > report.brier_gain_se):
        se = "n/a" if math.isnan(report.brier_gain_se) else f"{report.brier_gain_se:.3f}"
        report.reason = (f"Calibration was tested on {n} later paper(s): Brier {report.nested_brier:.3f} vs "
                         f"{report.climatology_brier:.3f} for the base rate (gain {report.brier_gain:+.3f}, standard "
                         f"error {se}). That is not a reliable improvement, so scores are shown as relative scores "
                         f"with an evidence band, not as probabilities.")
    else:
        report.valid = True
        report.reason = (f"Probabilities validated on {n} later papers ({report.method} calibration): Brier "
                         f"{report.nested_brier:.3f} vs {report.climatology_brier:.3f} for the base rate (skill "
                         f"{report.brier_skill:.0%}, gain {report.brier_gain:.3f} ± {report.brier_gain_se:.3f}).")
    return report
