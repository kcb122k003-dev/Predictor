"""Models that choose among other models using earlier backtest folds only.

* ``TunedModel`` picks a hyperparameter (window length, half-life) for target t by its mean
  primary metric over targets s < t.
* ``EnsembleModel`` averages the percentile ranks of the best members over targets s < t.

Each stored prediction for target s was computed from exams before s, so choosing with
them for a later target t never looks at exam t.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, ModelContext, ModelOutput, percentile_rank

MIN_EARLIER_FOLDS = 2


def _mean_before(ctx: ModelContext, name: str, t: int) -> tuple[float, int]:
    vals = [v for s, v in ctx.fold_metric.get(name, {}).items() if s < t and v == v]
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


class TunedModel(BaseModel):
    meta = True

    def __init__(self, name: str, display: str, variants: list[str], default: str, complexity: int = 2,
                 description: str = ""):
        self.name, self.display, self.variants, self.default = name, display, variants, default
        self.family = "baseline"
        self.complexity = complexity
        self.description = description

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        best, best_score = self.default, -np.inf
        enough = False
        for v in self.variants:
            score, n = _mean_before(ctx, v, t)
            if n >= MIN_EARLIER_FOLDS:
                enough = True
                if score > best_score + 1e-12:
                    best, best_score = v, score
        if not enough:
            best = self.default
        return ModelOutput(ctx.predictions[best][t], {"chosen": best, "tuned_on_earlier_folds": enough})


class EnsembleModel(BaseModel):
    name, display, family, complexity, meta = "ensemble", "Ensemble (rank average)", "ensemble", 5, True
    description = ("Averages the rankings of the best-performing models, chosen on earlier exams only. "
                   "No weights are fitted, so there is nothing extra to overfit.")

    def __init__(self, candidates: list[str], defaults: list[str], max_members: int = 3):
        self.candidates = candidates
        self.defaults = defaults
        self.max_members = max_members

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        ranked = []
        for name in self.candidates:
            if t not in ctx.predictions.get(name, {}):
                continue
            score, n = _mean_before(ctx, name, t)
            if n >= MIN_EARLIER_FOLDS:
                ranked.append((score, name))
        ranked.sort(reverse=True)
        members = [n for _, n in ranked[: self.max_members]]
        if len(members) < 2:
            members = [n for n in self.defaults if t in ctx.predictions.get(n, {})][: self.max_members]
        stacked = np.vstack([percentile_rank(ctx.predictions[m][t]) for m in members])
        return ModelOutput(stacked.mean(axis=0), {"members": members})
