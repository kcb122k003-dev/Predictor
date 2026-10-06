"""Probability calibration (spec section 25).

Scores become probabilities through Platt scaling on the *percentile rank* of each topic
within its exam: two parameters, fitted only on out-of-sample backtest predictions. The
nested check fits the calibrator for target t on targets before t, so the reported
calibration quality is itself out-of-sample. Probabilities are labelled as validated only
when this nested Brier score beats the base-rate (climatology) Brier score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..config.settings import Settings
from ..evaluation.metrics import brier, log_loss, reliability
from .base import percentile_rank


@dataclass
class Platt:
    a: float
    b: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(x, dtype=float) + self.b)))


def fit_platt(x: np.ndarray, y: np.ndarray) -> Platt:
    clf = LogisticRegression(C=100.0, max_iter=1000)
    clf.fit(np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y, dtype=float))
    return Platt(float(clf.coef_[0][0]), float(clf.intercept_[0]))


@dataclass
class CalibrationReport:
    valid: bool
    reason: str
    n_rows: int = 0
    n_positive: int = 0
    nested_brier: float = math.nan
    climatology_brier: float = math.nan
    brier_skill: float = math.nan
    nested_log_loss: float = math.nan
    ece: float = math.nan
    reliability: list[dict] = field(default_factory=list)
    final: Platt | None = None
    bootstrap: list[Platt] = field(default_factory=list)

    def probabilities(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self.final is None:
            return None
        x = percentile_rank(scores)
        p = self.final.predict(x)
        if self.bootstrap:
            sims = np.vstack([c.predict(x) for c in self.bootstrap])
            low, high = np.percentile(sims, [10, 90], axis=0)
        else:
            low, high = p, p
        return p, low, high

    def as_dict(self) -> dict:
        return {
            "valid": self.valid, "reason": self.reason, "rows": self.n_rows, "positives": self.n_positive,
            "nested_brier": _r(self.nested_brier), "climatology_brier": _r(self.climatology_brier),
            "brier_skill": _r(self.brier_skill), "nested_log_loss": _r(self.nested_log_loss), "ece": _r(self.ece),
            "reliability": self.reliability,
            "platt": {"a": _r(self.final.a), "b": _r(self.final.b)} if self.final else None,
        }


def _r(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def calibrate(predictions: dict[int, np.ndarray], Y: np.ndarray, targets: list[int], settings: Settings,
              seed: int = 13) -> CalibrationReport:
    cfg = settings.calibration
    min_rows, min_pos = int(cfg.min_rows), int(cfg.min_positives)
    xs = {t: percentile_rank(predictions[t]) for t in targets if t in predictions}
    ys = {t: Y[t] for t in xs}
    all_x = np.concatenate([xs[t] for t in xs]) if xs else np.zeros(0)
    all_y = np.concatenate([ys[t] for t in xs]) if xs else np.zeros(0)
    report = CalibrationReport(valid=False, reason="", n_rows=int(len(all_y)), n_positive=int(all_y.sum()))
    if len(all_y) < min_rows or all_y.sum() < min_pos:
        report.reason = (f"Not enough out-of-sample predictions to calibrate ({len(all_y)} rows, "
                         f"{int(all_y.sum())} positives; needs {min_rows} and {min_pos}). Scores are shown as "
                         f"relative scores, not probabilities.")
        return report

    nested_p, nested_y, clim_p = [], [], []
    ordered = sorted(xs)
    for i, t in enumerate(ordered):
        prev = ordered[:i]
        if not prev:
            continue
        px = np.concatenate([xs[s] for s in prev])
        py = np.concatenate([ys[s] for s in prev])
        if len(py) < min_rows // 2 or py.sum() < max(3, min_pos // 3) or py.sum() == len(py):
            continue
        cal = fit_platt(px, py)
        nested_p.append(cal.predict(xs[t]))
        nested_y.append(ys[t])
        clim_p.append(np.full(len(ys[t]), py.mean()))
    if nested_p:
        p = np.concatenate(nested_p)
        y = np.concatenate(nested_y)
        c = np.concatenate(clim_p)
        report.nested_brier = brier(p, y)
        report.climatology_brier = brier(c, y)
        report.brier_skill = 1.0 - report.nested_brier / report.climatology_brier if report.climatology_brier else math.nan
        report.nested_log_loss = log_loss(p, y)
        report.reliability, report.ece = reliability(p, y, int(cfg.reliability_bins))

    report.final = fit_platt(all_x, all_y)
    rng = np.random.default_rng(seed)
    for _ in range(int(cfg.bootstrap_samples)):
        pick = rng.choice(ordered, size=len(ordered), replace=True)
        bx = np.concatenate([xs[s] for s in pick])
        by = np.concatenate([ys[s] for s in pick])
        if 0 < by.sum() < len(by):
            report.bootstrap.append(fit_platt(bx, by))

    if not nested_p:
        report.reason = "Calibration could not be validated on held-out exams; probabilities are not shown."
    elif not (report.brier_skill > 0):
        report.reason = (f"Calibrated probabilities did not beat the base rate on held-out exams (Brier "
                         f"{report.nested_brier:.3f} vs {report.climatology_brier:.3f}). Scores are shown as "
                         f"relative scores.")
    else:
        report.valid = True
        report.reason = (f"Probabilities validated on held-out exams: Brier {report.nested_brier:.3f} vs "
                         f"{report.climatology_brier:.3f} for the base rate (skill {report.brier_skill:.0%}).")
    return report
