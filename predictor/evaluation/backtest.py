"""Rolling-origin, time-aware backtesting and final-model choice.

For every target exam t from ``min_train_exams`` (default 1) to the last known exam, each
model predicts exam t from exams 0..t-1. The same code with t = T (one past the last exam)
produces the real forecast. Meta models (tuned decays, the evidence-aware ensemble) choose
using only folds before t. No model needs a minimum number of exams: with two papers there
is one fold, with fifteen there are fourteen, and the report says how much that is worth.

The evidence-aware ensemble is the final ranking unless a single method beat it on the
backtest by more than one standard error of the paired difference; then that method is
used and the report says so. Baselines are always scored on the same folds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from ..config.settings import Settings
from ..features.builder import FeatureStore
from ..models.base import BaseModel, ModelContext, ModelOutput
from ..models.calibration import CalibrationReport, calibrate
from ..models.registry import FINAL_MODEL, build_models
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
    role: str = "baseline"
    scope: str = "course"
    df: float = 1.0
    fold_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    se: dict[str, float] = field(default_factory=dict)
    sd: dict[str, float] = field(default_factory=dict)
    ci95: dict[str, list[float | None]] = field(default_factory=dict)
    fallback_folds: int = 0
    unavailable_folds: int = 0
    notes: list[str] = field(default_factory=list)
    final_info: dict[str, Any] = field(default_factory=dict)
    # Filled after the run from the ensemble's final weights and the evidence profile.
    weight: float | None = None
    reliability: float | None = None
    skill: float | None = None
    status: str = ""
    status_reason: str = ""


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
    outputs: dict[str, dict[int, ModelOutput]] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    ctx: ModelContext | None = None

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
                "role": r.role, "scope": r.scope, "status": r.status, "status_reason": r.status_reason,
                "weight": _round(r.weight), "reliability": _round(r.reliability), "skill": _round(r.skill),
                "folds": len(r.fold_metrics), "mean": {k: _round(v) for k, v in r.mean.items()},
                "se": {k: _round(v) for k, v in r.se.items()}, "sd": {k: _round(v) for k, v in r.sd.items()},
                "ci95": {k: [_round(a) for a in v] for k, v in r.ci95.items()},
                "fallback_folds": r.fallback_folds, "unavailable_folds": r.unavailable_folds,
                "notes": r.notes, "description": r.description,
            })
        return rows


def _round(v: float | None) -> float | None:
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


def ci95(values: list[float]) -> list[float | None]:
    arr = np.asarray([v for v in values if v == v], dtype=float)
    if arr.size < 2:
        return [None, None]
    m, se = float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))
    q = float(stats.t.ppf(0.975, arr.size - 1))
    return [m - q * se, m + q * se]


def ordered_models(models: list[BaseModel]) -> list[BaseModel]:
    """Base models first, then tuned models, then ensembles (each meta model needs the ones before it)."""
    base = [m for m in models if not m.meta]
    meta = [m for m in models if m.meta]
    meta.sort(key=lambda m: 1 if m.role == "ensemble" else 0)
    return base + meta


class BacktestEngine:
    def __init__(self, panel: Panel, settings: Settings, models: list[BaseModel] | None = None,
                 store: FeatureStore | None = None, general=None):
        self.panel = panel
        self.settings = settings
        self.models = models if models is not None else build_models(settings)
        self.store = store or FeatureStore(panel, settings)
        self.primary = str(settings.models.primary_metric)
        self.seed = int(settings.models.random_seed)
        self.general = general

    def evaluable_targets(self) -> list[int]:
        first = max(int(self.settings.models.min_train_exams), 1)
        return list(range(first, self.panel.T))

    def run(self, *, audit_leakage: bool = True, calibrate_scores: bool = True) -> BacktestReport:
        panel, settings = self.panel, self.settings
        T = panel.T
        k = resolve_top_k(panel, settings)
        targets = self.evaluable_targets()
        all_targets = targets + [T]
        ctx = ModelContext(panel=panel, store=self.store, settings=settings, seed=self.seed, general=self.general)
        ctx.cache["models"] = self.models
        reports: dict[str, ModelReport] = {}
        notes: list[str] = []
        if T == 0:
            notes.append("No past papers yet: the ranking uses only the syllabus structure and the general "
                         "cross-course model.")
        elif not targets:
            notes.append("One past paper: there is no earlier paper to test a prediction against, so the ranking "
                         "cannot be validated yet. Pretrained, cross-course and Bayesian components still run.")

        for m in self.models:
            ok, why = m.gate(panel, ctx)
            reports[m.name] = ModelReport(m.name, m.display, m.family, m.complexity, ok, why, m.hidden, m.meta,
                                          getattr(m, "description", ""), m.role, m.scope, float(getattr(m, "df", 1.0)))
        runnable = [m for m in ordered_models(self.models) if reports[m.name].enabled]

        def record(m: BaseModel, t: int, out: ModelOutput) -> None:
            ctx.predictions.setdefault(m.name, {})[t] = np.asarray(out.scores, dtype=float)
            ctx.outputs.setdefault(m.name, {})[t] = out
            if t >= T:
                return
            if out.info.get("available") is False:
                reports[m.name].unavailable_folds += 1
                return
            if out.info.get("fallback"):
                reports[m.name].fallback_folds += 1
            rel = panel.Y[t]
            gains = _gains(panel, t)
            if m.name == "random":
                metrics = expected_random(panel.K, int(rel.sum()), k, gains)
            else:
                metrics = ranking_metrics(out.scores, rel, k, gains)
            reports[m.name].fold_metrics[t] = metrics
            ctx.fold_metric.setdefault(m.name, {})[t] = metrics.get(self.primary, math.nan)

        for m in runnable:
            for t in all_targets:
                record(m, t, m.predict(t, ctx))

        for r in reports.values():
            if r.fold_metrics:
                names = next(iter(r.fold_metrics.values())).keys()
                for name in names:
                    vals = [fm[name] for fm in r.fold_metrics.values()]
                    mean, se = mean_and_se(vals)
                    r.mean[name], r.se[name] = mean, se
                    r.sd[name] = float(np.std(vals, ddof=1)) if len(vals) > 1 else math.nan
                    if name in ("ndcg", "recall", "precision", "hit@1", "hit@3", "hit@5", "ndcg_marks"):
                        r.ci95[name] = ci95(vals)
            if r.fallback_folds:
                r.notes.append(f"On {r.fallback_folds} early fold(s) every training label was the same, so a "
                               f"classifier could not be fitted; the general model's score was used there.")
            if r.unavailable_folds:
                r.notes.append(f"Inputs were unavailable on {r.unavailable_folds} fold(s) (see the component status).")
        for name, outs in ctx.outputs.items():
            if T in outs and name in reports:
                reports[name].final_info = outs[T].info

        selected, best, reason = select_final(reports, self.primary, targets, settings)
        if selected not in ctx.predictions:
            selected = next(n for n in (FINAL_MODEL, "beta_binomial", "frequency") if n in ctx.predictions) \
                if any(n in ctx.predictions for n in (FINAL_MODEL, "beta_binomial", "frequency")) \
                else next(iter(ctx.predictions))
        calibration = calibrate(ctx.predictions[selected], panel.Y, targets, settings, self.seed) \
            if calibrate_scores else None
        final_outputs = {name: outs[T] for name, outs in ctx.outputs.items() if T in outs}
        report = BacktestReport(layer=panel.layer, k=k, primary=self.primary, targets=targets, future_index=T,
                                models=reports, predictions=ctx.predictions, final_outputs=final_outputs,
                                selected=selected, best_by_mean=best, selection_reason=reason,
                                calibration=calibration, notes=notes, outputs=ctx.outputs, ctx=ctx)
        report.validation = validation_summary(report, panel)
        if audit_leakage and targets:
            t_audit = targets[len(targets) // 2]
            report.leakage_audit = leakage_audit(panel, settings, [m for m in runnable if m.name != "random"],
                                                 t_audit, ctx.predictions, general=self.general)
        log_event(log, "backtest_done", layer=panel.layer, exams=T, folds=len(targets), k=k, selected=selected,
                  best=best)
        return report


def select_final(reports: dict[str, ModelReport], primary: str, targets: list[int], settings: Settings
                 ) -> tuple[str, str, str]:
    """The ensemble, unless a single visible method is reliably better on the backtest."""
    metric_label = f"{primary.upper()}@K"
    singles = [r for r in reports.values() if r.enabled and not r.hidden and r.name not in ("random", FINAL_MODEL)
               and r.fold_metrics and not math.isnan(r.mean.get(primary, math.nan))]
    ens = reports.get(FINAL_MODEL)
    has_ens = ens is not None and ens.enabled
    if not targets or not singles:
        if has_ens:
            return (FINAL_MODEL, FINAL_MODEL,
                    "No earlier paper is available to compare methods, so the evidence-aware ensemble is used with its "
                    "prior weights: pretrained semantics, syllabus structure, the cross-course model and Bayesian "
                    "recurrence, each weighted by how much this course's data can support it.")
        fallback = next((n for n in ("beta_binomial", "frequency") if n in reports), next(iter(reports)))
        return fallback, fallback, "No backtest folds; the Bayesian recurrence rate is used."
    best = max(singles, key=lambda r: (r.mean[primary], -r.complexity))
    if not has_ens or not ens.fold_metrics:
        return _one_se(singles, best, primary, targets, metric_label)
    common = [t for t in targets if t in ens.fold_metrics and t in best.fold_metrics]
    diffs = [ens.fold_metrics[t][primary] - best.fold_metrics[t][primary] for t in common]
    d, se = mean_and_se(diffs)
    threshold = float(settings.ensemble.replace_if_worse_by_se)
    n = len(common)
    if not math.isnan(se) and d < -threshold * se:
        chosen, _, why = _one_se(singles, best, primary, targets, metric_label)
        return chosen, best.name, (f"The evidence-aware ensemble scored {ens.mean[primary]:.3f} {metric_label}, "
                                   f"{-d:.3f} below {best.display} ({best.mean[primary]:.3f}) over {n} held-out papers, "
                                   f"more than one standard error ({se:.3f}). A single method is used instead: {why}")
    se_txt = "n/a with one paper" if math.isnan(se) else f"{se:.3f}"
    if d >= 0:
        verdict = (f"It scored {ens.mean[primary]:.3f} {metric_label} over {n} held-out paper(s), {d:+.3f} against the "
                   f"best single method ({best.display}, {best.mean[primary]:.3f}; standard error of the difference "
                   f"{se_txt}).")
    else:
        verdict = (f"It scored {ens.mean[primary]:.3f} {metric_label} over {n} held-out paper(s), {d:+.3f} against the "
                   f"best single method ({best.display}, {best.mean[primary]:.3f}); the gap is within one standard error "
                   f"({se_txt}), and the best single method was picked after seeing these same papers, so it is "
                   f"not a reliable winner.")
    rnd = reports.get("random")
    if rnd is not None and rnd.mean:
        verdict += f" Random selection scores {rnd.mean.get(primary, float('nan')):.3f} on the same papers."
    return FINAL_MODEL, best.name, "The evidence-aware ensemble is used. " + verdict


def _one_se(singles: list[ModelReport], best: ModelReport, primary: str, targets: list[int], metric_label: str
            ) -> tuple[str, str, str]:
    acceptable = []
    for r in singles:
        diffs = [best.fold_metrics[t][primary] - r.fold_metrics[t][primary]
                 for t in targets if t in r.fold_metrics and t in best.fold_metrics]
        mean_diff, se_diff = mean_and_se(diffs)
        if math.isnan(se_diff):
            se_diff = 0.0
        if mean_diff <= se_diff + 1e-12:
            acceptable.append((r, mean_diff, se_diff))
    chosen, diff, se = min(acceptable, key=lambda x: (x[0].complexity, -x[0].mean[primary]))
    if chosen.name == best.name:
        return best.name, best.name, (f"{best.display} had the highest mean {metric_label} ({best.mean[primary]:.3f}) "
                                      f"and no simpler method came within one standard error.")
    return chosen.name, best.name, (f"{chosen.display} is simpler and within one standard error of {best.display} "
                                    f"({diff:.3f} behind, standard error {se:.3f}).")


# Backward-compatible name used by older callers and tests.
def select_model(reports: dict[str, ModelReport], primary: str, rule: str, targets: list[int]
                 ) -> tuple[str, str, str]:
    singles = [r for r in reports.values() if r.enabled and not r.hidden and r.name != "random"
               and r.fold_metrics and not math.isnan(r.mean.get(primary, math.nan))]
    if not targets or not singles:
        return FINAL_MODEL, FINAL_MODEL, "No backtest folds were available."
    best = max(singles, key=lambda r: (r.mean[primary], -r.complexity))
    if rule == "best":
        return best.name, best.name, f"{best.display} had the highest mean {primary.upper()}@K ({best.mean[primary]:.3f})."
    return _one_se(singles, best, primary, targets, f"{primary.upper()}@K")


def validation_summary(report: BacktestReport, panel: Panel) -> dict[str, Any]:
    """Small-data validation facts: folds, exams, fold variability, confidence intervals, baselines."""
    p = report.primary
    sel = report.models.get(report.selected)
    out: dict[str, Any] = {"exams": panel.T, "folds": len(report.targets), "metric": f"{p}@{report.k}",
                           "first_target_history": report.targets[0] if report.targets else None}
    if sel is None or not sel.fold_metrics:
        out["message"] = ("No held-out paper is available yet, so accuracy cannot be measured. Rankings come from "
                          "prior knowledge and Bayesian estimates with wide uncertainty.")
        return out
    out.update({"selected": sel.name, "mean": _round(sel.mean.get(p)), "se": _round(sel.se.get(p)),
                "sd": _round(sel.sd.get(p)), "ci95": [_round(v) for v in sel.ci95.get(p, [None, None])],
                "per_fold": {str(t): _round(m.get(p)) for t, m in sel.fold_metrics.items()}})
    comps = []
    for base in ("random", "frequency", "recency", "beta_binomial"):
        r = report.models.get(base)
        if r is None or not r.fold_metrics or base == sel.name:
            continue
        common = [t for t in sel.fold_metrics if t in r.fold_metrics]
        d, se = mean_and_se([sel.fold_metrics[t][p] - r.fold_metrics[t][p] for t in common])
        wins = sum(1 for t in common if sel.fold_metrics[t][p] > r.fold_metrics[t][p] + 1e-9)
        losses = sum(1 for t in common if sel.fold_metrics[t][p] < r.fold_metrics[t][p] - 1e-9)
        comps.append({"baseline": base, "display": r.display, "baseline_mean": _round(r.mean.get(p)),
                      "difference": _round(d), "difference_se": _round(se), "folds_better": wins,
                      "folds_worse": losses, "folds": len(common)})
    out["baselines"] = comps
    n = len(report.targets)
    width = None if out["ci95"][0] is None else round(out["ci95"][1] - out["ci95"][0], 3)
    out["ci_width"] = width
    if n < 2:
        out["message"] = (f"{panel.T} papers give {n} held-out fold. One fold shows whether the ranking was sensible "
                          f"for that paper but cannot measure reliability; treat the accuracy figure as anecdotal.")
    else:
        out["message"] = (f"{panel.T} papers give {n} held-out folds. The 95% confidence interval of the final "
                          f"ranking's {p.upper()}@{report.k} is {width:.2f} wide"
                          + ("; differences between methods smaller than that are not meaningful." if width else "."))
    return out


def leakage_audit(panel: Panel, settings: Settings, models: list[BaseModel], t: int,
                  reference: dict[str, dict[int, np.ndarray]], general=None) -> dict[str, Any]:
    """Scramble every exam from t onward, rerun every model up to t and confirm predictions for t do not change.

    Meta models (tuned decays, the ensemble) are rerun too: they read earlier folds, so they
    must see the same earlier predictions and metrics on the scrambled panel.
    """
    rng = np.random.default_rng(99)
    scrambled = Panel(**{**panel.__dict__})
    for attr in ("Y", "marks", "soft", "formats", "n_questions", "exact_repeat", "para_repeat", "semantic", "quality"):
        src = getattr(panel, attr)
        if src is None:
            continue
        arr = src.copy()
        if arr[t:].size:
            arr[t:] = rng.permutation(arr[t:].reshape(-1)).reshape(arr[t:].shape)
        if attr == "Y":
            arr[t:] = 1.0 - arr[t:]
        setattr(scrambled, attr, arr)
    store = FeatureStore(scrambled, settings)
    ctx = ModelContext(panel=scrambled, store=store, settings=settings, seed=int(settings.models.random_seed),
                       general=general)
    k = resolve_top_k(panel, settings)  # same k as the reference run
    primary = str(settings.models.primary_metric)
    first = max(int(settings.models.min_train_exams), 1)
    failures, checked = [], 0
    for m in ordered_models(models):
        for s in range(first, t + 1):
            try:
                out = m.predict(s, ctx)
            except Exception as exc:  # pragma: no cover - defensive
                failures.append(f"{m.name}: {exc}")
                break
            ctx.predictions.setdefault(m.name, {})[s] = np.asarray(out.scores, dtype=float)
            ctx.outputs.setdefault(m.name, {})[s] = out
            if s < t and out.info.get("available") is not False:
                ctx.fold_metric.setdefault(m.name, {})[s] = ranking_metrics(out.scores, scrambled.Y[s], k,
                                                                            _gains(scrambled, s))[primary]
        ref = reference.get(m.name, {}).get(t)
        got = ctx.predictions.get(m.name, {}).get(t)
        if ref is not None and got is not None:
            checked += 1
            if not np.allclose(ref, got, atol=1e-9):
                failures.append(m.name)
    return {"target_index": t, "models_checked": checked, "passed": not failures, "failures": failures}
