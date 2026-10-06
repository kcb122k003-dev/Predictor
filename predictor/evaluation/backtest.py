"""Expanding-window, time-aware backtesting and model selection (spec sections 23, 31, 44, 62).

For every target exam t (from ``min_train_exams`` to the last known exam) each model
predicts exam t from exams 0..t-1. The same code with t = T (one past the last exam)
produces the real forecast. Meta models (tuned windows/half-lives, the ensemble) choose
using only folds before t. The selected model is the simplest one within one standard
error of the best mean primary metric (paired across folds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.settings import Settings
from ..features.builder import FeatureStore
from ..models.base import BaseModel, ModelContext, ModelOutput
from ..models.calibration import CalibrationReport, calibrate
from ..models.registry import DEFAULT_FALLBACK, build_models
from ..temporal.panel import Panel
from ..utils.logging import get_logger, log_event
from .metrics import expected_random, mean_and_se, ranking_metrics

log = get_logger("evaluation")


@dataclass
class ModelReport:
    name: str
    display: str
    family: str
    complexity: int
    enabled: bool
    gate_reason: str
    hidden: bool
    meta: bool
    description: str = ""
    fold_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    se: dict[str, float] = field(default_factory=dict)
    fallback_folds: int = 0
    notes: list[str] = field(default_factory=list)
    final_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestReport:
    layer: str
    k: int
    primary: str
    targets: list[int]
    future_index: int
    models: dict[str, ModelReport]
    predictions: dict[str, dict[int, np.ndarray]]
    final_outputs: dict[str, ModelOutput]
    selected: str
    best_by_mean: str
    selection_reason: str
    calibration: CalibrationReport | None
    notes: list[str] = field(default_factory=list)
    leakage_audit: dict[str, Any] = field(default_factory=dict)

    @property
    def final_scores(self) -> np.ndarray:
        return self.predictions[self.selected][self.future_index]

    def summary_table(self) -> list[dict[str, Any]]:
        rows = []
        for r in self.models.values():
            if r.hidden:
                continue
            rows.append({
                "model": r.name, "display": r.display, "family": r.family, "complexity": r.complexity,
                "enabled": r.enabled, "gate_reason": r.gate_reason, "selected": r.name == self.selected,
                "folds": len(r.fold_metrics), "mean": {k: _round(v) for k, v in r.mean.items()},
                "se": {k: _round(v) for k, v in r.se.items()}, "fallback_folds": r.fallback_folds,
                "notes": r.notes, "description": r.description,
            })
        return rows


def _round(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def resolve_top_k(panel: Panel, settings: Settings) -> int:
    cfg = settings.models
    lo, hi = int(cfg.top_k_min), int(cfg.top_k_max)
    if str(cfg.top_k).lower() != "auto":
        return max(1, min(int(cfg.top_k), panel.K or 1))
    per_exam = panel.Y.sum(axis=1)
    per_exam = per_exam[per_exam > 0]
    k = int(round(float(np.median(per_exam)))) if per_exam.size else lo
    return max(1, min(max(lo, min(k, hi)), panel.K or 1))


def _gains(panel: Panel, t: int) -> np.ndarray:
    total = max(panel.exams[t].total_marks, 1e-9)
    g = panel.marks[t] / total
    # Topics that appeared without known marks still count as relevant.
    return np.where((panel.Y[t] > 0) & (g <= 0), 1.0 / max(panel.Y[t].sum(), 1), g)


class BacktestEngine:
    def __init__(self, panel: Panel, settings: Settings, models: list[BaseModel] | None = None,
                 store: FeatureStore | None = None):
        self.panel = panel
        self.settings = settings
        self.models = models if models is not None else build_models(settings)
        self.store = store or FeatureStore(panel, settings)
        self.primary = str(settings.models.primary_metric)
        self.seed = int(settings.models.random_seed)

    def evaluable_targets(self) -> list[int]:
        T = self.panel.T
        cfg = self.settings.models
        if T < int(cfg.sufficiency.min_exams_backtest):
            return []
        first = max(int(cfg.min_train_exams), 1)
        return list(range(first, T))

    def run(self, *, audit_leakage: bool = True) -> BacktestReport:
        panel, settings = self.panel, self.settings
        T = panel.T
        k = resolve_top_k(panel, settings)
        targets = self.evaluable_targets()
        all_targets = targets + [T]
        ctx = ModelContext(panel=panel, store=self.store, settings=settings, seed=self.seed)
        reports: dict[str, ModelReport] = {}
        outputs: dict[str, dict[int, ModelOutput]] = {}
        notes: list[str] = []
        if not targets:
            notes.append(f"Backtesting needs at least {int(settings.models.sufficiency.min_exams_backtest)} exams; "
                         f"you have {T}. Predictions use a transparent statistical model and cannot be validated.")
        if 0 < T < int(settings.models.sufficiency.low_confidence_exams):
            notes.append(f"Only {T} historical exams were supplied. Predictions are highly uncertain.")

        # Gates.
        tuned_parents = {v: m.name for m in self.models if getattr(m, "variants", None) for v in m.variants}
        for m in self.models:
            ok, why = m.gate(panel, ctx)
            reports[m.name] = ModelReport(m.name, m.display, m.family, m.complexity, ok, why, m.hidden, m.meta,
                                          getattr(m, "description", ""))
        for variant, parent in tuned_parents.items():
            if variant in reports and parent in reports:
                reports[variant].enabled = reports[parent].enabled

        base = [m for m in self.models if not m.meta and reports[m.name].enabled]
        meta = [m for m in self.models if m.meta and reports[m.name].enabled]
        meta.sort(key=lambda m: 1 if m.name == "ensemble" else 0)

        def record(m: BaseModel, t: int, out: ModelOutput) -> None:
            ctx.predictions.setdefault(m.name, {})[t] = np.asarray(out.scores, dtype=float)
            outputs.setdefault(m.name, {})[t] = out
            if out.info.get("fallback"):
                reports[m.name].fallback_folds += 1 if t < T else 0
            if t < T:
                rel = panel.Y[t]
                gains = _gains(panel, t)
                if m.name == "random":
                    metrics = expected_random(panel.K, int(rel.sum()), k, gains)
                else:
                    metrics = ranking_metrics(out.scores, rel, k, gains)
                reports[m.name].fold_metrics[t] = metrics
                ctx.fold_metric.setdefault(m.name, {})[t] = metrics.get(self.primary, math.nan)

        for m in base:
            for t in all_targets:
                record(m, t, m.predict(t, ctx))
        for m in meta:
            for t in all_targets:
                record(m, t, m.predict(t, ctx))

        for r in reports.values():
            if not r.fold_metrics:
                continue
            names = next(iter(r.fold_metrics.values())).keys()
            for name in names:
                mean, se = mean_and_se([fm[name] for fm in r.fold_metrics.values()])
                r.mean[name], r.se[name] = mean, se
            if r.fallback_folds:
                r.notes.append(f"Used the recency-weighted fallback on {r.fallback_folds} early fold(s) with too "
                               f"little training data.")
        for name, outs in outputs.items():
            if T in outs:
                reports[name].final_info = outs[T].info

        selected, best, reason = select_model(reports, self.primary, str(settings.models.selection_rule), targets)
        if selected not in ctx.predictions:
            selected = DEFAULT_FALLBACK if DEFAULT_FALLBACK in ctx.predictions else next(iter(ctx.predictions))
        calibration = calibrate(ctx.predictions[selected], panel.Y, targets, settings, self.seed) if targets else None
        final_outputs = {name: outs[T] for name, outs in outputs.items() if T in outs}
        report = BacktestReport(layer=panel.layer, k=k, primary=self.primary, targets=targets, future_index=T,
                                models=reports, predictions=ctx.predictions, final_outputs=final_outputs,
                                selected=selected, best_by_mean=best, selection_reason=reason,
                                calibration=calibration, notes=notes)
        if audit_leakage and targets:
            report.leakage_audit = leakage_audit(panel, settings, [m for m in base if m.name != "random"],
                                                 targets[0], ctx.predictions)
        log_event(log, "backtest_done", layer=panel.layer, exams=T, folds=len(targets), k=k, selected=selected,
                  best=best)
        return report


def select_model(reports: dict[str, ModelReport], primary: str, rule: str, targets: list[int]
                 ) -> tuple[str, str, str]:
    candidates = [r for r in reports.values() if r.enabled and not r.hidden and r.name != "random"
                  and r.fold_metrics and not math.isnan(r.mean.get(primary, math.nan))]
    if not targets or not candidates:
        return (DEFAULT_FALLBACK, DEFAULT_FALLBACK,
                "No backtest folds were available, so the transparent Bayesian rate model is used. It shrinks each "
                "topic's recency-weighted frequency toward the average and needs no training.")
    best = max(candidates, key=lambda r: (r.mean[primary], -r.complexity))
    metric_label = f"{primary.upper()}@K"
    if rule == "best":
        return best.name, best.name, f"{best.display} had the highest mean {metric_label} ({best.mean[primary]:.3f})."
    acceptable = []
    for r in candidates:
        diffs = [best.fold_metrics[t][primary] - r.fold_metrics[t][primary]
                 for t in targets if t in r.fold_metrics and t in best.fold_metrics]
        mean_diff, se_diff = mean_and_se(diffs)
        if math.isnan(se_diff):
            se_diff = 0.0
        if mean_diff <= se_diff + 1e-12:
            acceptable.append((r, mean_diff, se_diff))
    chosen, diff, se = min(acceptable, key=lambda x: (x[0].complexity, -x[0].mean[primary]))
    random_mean = reports.get("random").mean.get(primary) if "random" in reports and reports["random"].mean else None
    if chosen.name == best.name:
        reason = (f"{best.display} had the highest mean {metric_label} ({best.mean[primary]:.3f} over "
                  f"{len(best.fold_metrics)} held-out exams) and no simpler model came within one standard error.")
    else:
        reason = (f"{best.display} had the highest mean {metric_label} ({best.mean[primary]:.3f}), but "
                  f"{chosen.display} is simpler and only {diff:.3f} behind (standard error of the difference "
                  f"{se:.3f}). By the one-standard-error rule the simpler model is used.")
    if random_mean is not None and not math.isnan(random_mean):
        reason += f" Random selection scores {random_mean:.3f} on the same folds."
    return chosen.name, best.name, reason


def leakage_audit(panel: Panel, settings: Settings, models: list[BaseModel], t: int,
                  reference: dict[str, dict[int, np.ndarray]]) -> dict[str, Any]:
    """Scramble every exam from t onward and confirm predictions for t do not change."""
    rng = np.random.default_rng(99)
    scrambled = Panel(**{**panel.__dict__})
    for attr in ("Y", "marks", "soft", "formats", "n_questions", "exact_repeat", "para_repeat"):
        arr = getattr(panel, attr).copy()
        arr[t:] = rng.permutation(arr[t:].reshape(-1)).reshape(arr[t:].shape) if arr[t:].size else arr[t:]
        if attr == "Y":
            arr[t:] = 1.0 - arr[t:]
        setattr(scrambled, attr, arr)
    store = FeatureStore(scrambled, settings)
    ctx = ModelContext(panel=scrambled, store=store, settings=settings, seed=int(settings.models.random_seed))
    failures = []
    for m in models:
        try:
            out = m.predict(t, ctx)
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(f"{m.name}: {exc}")
            continue
        ref = reference.get(m.name, {}).get(t)
        if ref is not None and not np.allclose(ref, out.scores, atol=1e-9):
            failures.append(m.name)
    return {"target_index": t, "models_checked": len(models), "passed": not failures, "failures": failures}
