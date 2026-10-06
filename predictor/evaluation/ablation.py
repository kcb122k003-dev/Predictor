"""Ablation study (spec sections 62 and 63).

Two views, both on the same held-out folds as the main backtest:

* Leave-one-group-out: the logistic model without one feature group at a time.
* Staged additions: frequency only, then + recency, + temporal dynamics, + semantic, then
  all groups. Each step shows the incremental benefit (or harm) of a signal family.

A separate check runs the selected model on a panel built without the syllabus filter
(uncertain and out-of-syllabus questions forced onto their nearest topic) and scores it
against the filtered ground truth.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ..config.settings import Settings
from ..features.builder import FEATURE_GROUPS, GROUP_LABELS, FeatureStore
from ..models.base import BaseModel, ModelContext
from ..models.learned import LogisticModel
from ..temporal.panel import Panel
from .metrics import mean_and_se, ranking_metrics

STAGES = [
    ("Frequency only", ("frequency",)),
    ("+ recency", ("frequency", "recency")),
    ("+ temporal dynamics", ("frequency", "recency", "temporal")),
    ("+ semantic similarity", ("frequency", "recency", "temporal", "semantic")),
    ("All signals", tuple(FEATURE_GROUPS)),
]


def _evaluate(model: BaseModel, panel: Panel, store: FeatureStore, settings: Settings, targets: list[int],
              k: int, primary: str, labels: np.ndarray | None = None) -> dict[int, float]:
    ctx = ModelContext(panel=panel, store=store, settings=settings, seed=int(settings.models.random_seed))
    Y = panel.Y if labels is None else labels
    out = {}
    for t in targets:
        scores = model.predict(t, ctx).scores
        out[t] = ranking_metrics(scores, Y[t], k)[primary]
    return out


def _summary(name: str, folds: dict[int, float], reference: dict[int, float] | None) -> dict[str, Any]:
    mean, se = mean_and_se(list(folds.values()))
    row = {"variant": name, "mean": _r(mean), "se": _r(se)}
    if reference:
        diffs = [folds[t] - reference[t] for t in folds if t in reference]
        d_mean, d_se = mean_and_se(diffs)
        row["delta"] = _r(d_mean)
        row["delta_se"] = _r(d_se)
        wins = sum(1 for d in diffs if d > 1e-9)
        losses = sum(1 for d in diffs if d < -1e-9)
        row["folds_better"], row["folds_worse"] = wins, losses
    return row


def _r(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def run_ablation(panel: Panel, settings: Settings, store: FeatureStore, targets: list[int], k: int,
                 primary: str, logistic_enabled: bool, logistic_reason: str = "") -> dict[str, Any]:
    if not targets:
        return {"available": False, "reason": "No backtest folds are available."}
    if not logistic_enabled:
        return {"available": False,
                "reason": ("The ablation study varies the feature groups of the logistic model, which is disabled "
                           f"for this course: {logistic_reason}")}
    full = _evaluate(LogisticModel(), panel, store, settings, targets, k, primary)
    leave_one = [_summary("All signals", full, None)]
    for group in FEATURE_GROUPS:
        model = LogisticModel(exclude_groups=(group,), name=f"logistic-no-{group}")
        folds = _evaluate(model, panel, store, settings, targets, k, primary)
        row = _summary(f"Without {GROUP_LABELS[group].lower()}", folds, full)
        row["group"] = group
        leave_one.append(row)
    staged = []
    prev: dict[int, float] | None = None
    for label, groups in STAGES:
        excluded = tuple(g for g in FEATURE_GROUPS if g not in groups)
        model = LogisticModel(exclude_groups=excluded, name=f"logistic-stage-{len(staged)}")
        folds = _evaluate(model, panel, store, settings, targets, k, primary)
        staged.append(_summary(label, folds, prev))
        prev = folds
    return {"available": True, "metric": f"{primary}@{k}", "leave_one_out": leave_one, "staged": staged,
            "note": ("Delta is the paired difference against the full model (leave-one-out) or the previous stage "
                     "(staged). A delta within about one standard error is not a reliable difference.")}


def syllabus_filter_check(model_factory: Callable[[], BaseModel], filtered: Panel, unfiltered: Panel,
                          settings: Settings, targets: list[int], k: int, primary: str) -> dict[str, Any]:
    if not targets:
        return {"available": False, "reason": "No backtest folds are available."}
    with_filter = _evaluate(model_factory(), filtered, FeatureStore(filtered, settings), settings, targets, k, primary)
    without = _evaluate(model_factory(), unfiltered, FeatureStore(unfiltered, settings), settings, targets, k,
                        primary, labels=filtered.Y)
    extra = float((unfiltered.Y - filtered.Y).clip(min=0).sum())
    return {
        "available": True, "metric": f"{primary}@{k}",
        "with_filter": _summary("With syllabus filter", with_filter, None),
        "without_filter": _summary("Without syllabus filter", without, with_filter),
        "extra_appearances_without_filter": int(extra),
        "note": ("Without the filter, uncertain and out-of-syllabus questions are forced onto their nearest topic. "
                 "Both runs are scored against the filtered ground truth."),
    }
