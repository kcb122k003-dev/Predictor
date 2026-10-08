"""Ablation study on the same held-out papers as the main backtest.

Staged additions (each stage is the evidence-aware ensemble restricted to the listed
components, with weights learned exactly as in the full model):

    frequency -> + recency -> + Bayesian smoothing -> + semantic -> + topic structure
    -> + temporal dynamics -> full ensemble (+ cross-course and course-learned models)

and leave-one-component-out from the full ensemble. Every row reports Hit@1/3/5,
Recall@K, Precision@K and NDCG@K on topics, recall@K on concepts (finest syllabus level)
and recall of exactly repeated questions, with the paired change against the previous stage
and its standard error. A change is called reliable only when it lies outside a two-sided
95% t-interval of the paired per-paper differences; with few papers most changes do not, and
the table says so instead of claiming an improvement. Reliable gains and reliable losses are
listed separately.

A separate check scores the final model with and without the syllabus filter against the
filtered ground truth.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ..models.base import ModelContext
from ..models.calibration import t_bound
from ..temporal.panel import Panel
from .metrics import mean_and_se, ranking_metrics

STAGES: list[tuple[str, list[str]]] = [
    ("Frequency only", ["frequency"]),
    ("+ recency", ["frequency", "recency"]),
    ("+ Bayesian smoothing", ["beta_binomial"]),
    ("+ semantic", ["semantic"]),
    ("+ topic structure", ["coverage", "question_type", "cooccurrence"]),
    ("+ temporal dynamics", ["hazard", "markov", "hmm"]),
    ("Full ensemble", ["general", "logistic", "gradient_boosting", "random_forest"]),
]
METRICS = ["hit@1", "hit@3", "hit@5", "recall", "precision", "ndcg"]


def _r(v: float | None) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def _stage_predictions(members: list[str], ctx: ModelContext, ensemble, targets: list[int], name: str
                       ) -> dict[int, np.ndarray]:
    present = [m for m in members if m in ctx.predictions]
    if not present:
        return {}
    if len(present) == 1:
        return {t: ctx.predictions[present[0]][t] for t in targets if t in ctx.predictions[present[0]]}
    sub = ensemble.subset(present, name)  # cumulative: earlier stages' models (frequency included) stay in
    return {t: np.asarray(sub.predict(t, ctx).scores, dtype=float) for t in targets}


def _metrics(preds: dict[int, np.ndarray], panel: Panel, targets: list[int], k: int) -> dict[int, dict[str, float]]:
    return {t: ranking_metrics(preds[t], panel.Y[t], k) for t in targets if t in preds}


def _row(label: str, folds: dict[int, dict[str, float]], previous: dict[int, dict[str, float]] | None,
         extra: dict[str, dict[int, float]], prev_extra: dict[str, dict[int, float]] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": label, "folds": len(folds)}
    for m in METRICS:
        mean, se = mean_and_se([f[m] for f in folds.values()])
        row[m] = _r(mean)
        row[f"{m}_se"] = _r(se)
    for key, vals in extra.items():
        mean, se = mean_and_se(list(vals.values()))
        row[key] = _r(mean)
        row[f"{key}_se"] = _r(se)
    if previous:
        common = [t for t in folds if t in previous]
        diffs = [folds[t]["ndcg"] - previous[t]["ndcg"] for t in common]
        d, se = mean_and_se(diffs)
        row["delta"], row["delta_se"] = _r(d), _r(se)
        row["folds_better"] = sum(1 for x in diffs if x > 1e-9)
        row["folds_worse"] = sum(1 for x in diffs if x < -1e-9)
        n = len(diffs)
        row["reliable"] = bool(n >= 2 and not math.isnan(se) and se > 0 and abs(d) > t_bound(n, 0.975) * se)
        for key, vals in extra.items():
            if prev_extra and key in prev_extra:
                c = [t for t in vals if t in prev_extra[key]]
                dd, ss = mean_and_se([vals[t] - prev_extra[key][t] for t in c])
                row[f"{key}_delta"], row[f"{key}_delta_se"] = _r(dd), _r(ss)
    return row


def run_ablation(report, concept_report=None, recurrence_fn: Callable[[dict[int, np.ndarray]], dict[int, float]] | None
                 = None) -> dict[str, Any]:
    targets = report.targets
    ctx = report.ctx
    if not targets or ctx is None:
        return {"available": False, "reason": ("No held-out paper exists yet (one paper or none), so no ablation can "
                                               "be measured. Every component still runs.")}
    from ..models.meta import EvidenceEnsemble

    ensemble = next((m for m in _models_of(ctx) if isinstance(m, EvidenceEnsemble)), None)
    if ensemble is None:
        return {"available": False, "reason": "The evidence-aware ensemble is not enabled in Settings."}
    panel = ctx.panel
    k = report.k
    c_ens = None
    if concept_report is not None and concept_report.ctx is not None:
        c_ens = next((m for m in _models_of(concept_report.ctx) if isinstance(m, EvidenceEnsemble)), None)

    def extras(members: list[str], name: str, preds: dict[int, np.ndarray]) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        if c_ens is not None and concept_report.targets:
            cp = _stage_predictions(members, concept_report.ctx, c_ens, concept_report.targets, name)
            if cp:
                out["concept_recall"] = {t: ranking_metrics(cp[t], concept_report.ctx.panel.Y[t], concept_report.k)["recall"]
                                         for t in cp if concept_report.ctx.panel.Y[t].sum() > 0}
        if recurrence_fn is not None and preds:
            rec = recurrence_fn(preds)
            if rec:
                out["exact_recurrence_recall"] = rec
        return out

    staged, prev, prev_extra = [], None, None
    members: list[str] = []
    for i, (label, add) in enumerate(STAGES):
        members = members + add
        name = f"ablation-stage-{i}"
        preds = _stage_predictions(members, ctx, ensemble, targets, name)
        if not preds:
            continue
        folds = _metrics(preds, panel, targets, k)
        ex = extras(members, name, preds)
        staged.append(_row(label, folds, prev, ex, prev_extra))
        prev, prev_extra = folds, ex
    full_members = list(ensemble.members)
    full = _stage_predictions(full_members, ctx, ensemble, targets, "ablation-full")
    full_folds = _metrics(full, panel, targets, k)
    leave_one = [_row("Full ensemble", full_folds, None, {}, None)]
    for m in full_members:
        if m not in ctx.predictions:
            continue
        rest = [x for x in full_members if x != m]
        preds = _stage_predictions(rest, ctx, ensemble, targets, f"ablation-no-{m}")
        if not preds:
            continue
        row = _row(f"Without {ensemble.displays.get(m, m)}", _metrics(preds, panel, targets, k), full_folds, {}, None)
        row["component"] = m
        leave_one.append(row)
    n = len(targets)
    better = [r["variant"] for r in staged[1:] if r.get("reliable") and r["delta"] > 0]
    worse = [r["variant"] for r in staged[1:] if r.get("reliable") and r["delta"] < 0]
    note = ("Each stage adds components to the evidence-aware ensemble; 'Change' is the paired NDCG difference "
            "against the previous stage over the same held-out papers, with its standard error. A change counts as "
            "reliable only outside a two-sided 95% t-interval. ")
    if n < 5:
        note += (f"With {n} held-out paper(s) the standard errors are large, so these differences show direction only "
                 f"and do not establish that a stage helps.")
    elif better or worse:
        parts = []
        if better:
            parts.append("reliably better: " + ", ".join(better))
        if worse:
            parts.append("reliably worse: " + ", ".join(worse))
        note += "On these papers, " + "; ".join(parts) + ". Every other change is within the noise."
    else:
        note += "No stage changed NDCG reliably on these papers; every change is within the noise."
    return {"available": True, "metric": f"ndcg@{k}", "k": k, "folds": n, "staged": staged, "leave_one_out": leave_one,
            "metrics": METRICS, "note": note}


def _models_of(ctx: ModelContext) -> list:
    return list(ctx.cache.get("models", []))


def syllabus_filter_check(filtered_predictions: dict[int, np.ndarray], unfiltered_predictions: dict[int, np.ndarray],
                          filtered: Panel, unfiltered: Panel, targets: list[int], k: int, primary: str,
                          model_display: str) -> dict[str, Any]:
    """Score the final model with and without the syllabus filter against the filtered ground truth."""
    targets = [t for t in targets if t in filtered_predictions and t in unfiltered_predictions]
    if not targets:
        return {"available": False, "reason": "No held-out paper is available yet."}
    with_filter = {t: ranking_metrics(filtered_predictions[t], filtered.Y[t], k)[primary] for t in targets}
    without = {t: ranking_metrics(unfiltered_predictions[t], filtered.Y[t], k)[primary] for t in targets}
    extra = float((unfiltered.Y - filtered.Y).clip(min=0).sum())

    def summary(name, folds, ref):
        mean, se = mean_and_se(list(folds.values()))
        row = {"variant": name, "mean": _r(mean), "se": _r(se)}
        if ref:
            d, s = mean_and_se([folds[t] - ref[t] for t in folds if t in ref])
            row["delta"], row["delta_se"] = _r(d), _r(s)
        return row

    return {
        "available": True, "metric": f"{primary}@{k}", "model": model_display,
        "with_filter": summary("With syllabus filter", with_filter, None),
        "without_filter": summary("Without syllabus filter", without, with_filter),
        "extra_appearances_without_filter": int(extra),
        "note": ("Without the filter, uncertain and out-of-syllabus questions are forced onto their nearest topic. "
                 "Both runs are scored against the filtered ground truth."),
    }
