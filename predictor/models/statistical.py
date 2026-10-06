"""Transparent statistical models: baselines, Bayesian rates, Markov, hazard, semantic."""

from __future__ import annotations

import numpy as np

from ..temporal import dynamics as dyn
from .base import BaseModel, ModelContext, ModelOutput, percentile_rank


def _feat(ctx: ModelContext, t: int, name: str) -> np.ndarray:
    return ctx.store.at(t).column(name)


class RandomModel(BaseModel):
    name, display, family, complexity = "random", "Random selection", "baseline", 0
    description = "Uniformly random ranking. Its metrics are exact expectations, not one random draw."

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        rng = np.random.default_rng(ctx.seed + t)
        return ModelOutput(rng.random(ctx.panel.K), {"expected_metrics": True})


class FrequencyModel(BaseModel):
    name, display, family, complexity = "frequency", "Most frequent topics", "baseline", 1
    description = "Share of past exams in which the topic appeared."

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        freq = _feat(ctx, t, "freq_all")
        return ModelOutput(freq + 1e-3 * _feat(ctx, t, "ewma_short"))


class LastExamModel(BaseModel):
    name, display, family, complexity = "last_exam", "Most recent topics", "baseline", 1
    description = "Topics in the previous exam first, then by frequency."

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        return ModelOutput(_feat(ctx, t, "last1") + 0.1 * _feat(ctx, t, "freq_all"))


class WindowModel(BaseModel):
    family, complexity, hidden = "baseline", 2, True

    def __init__(self, window: int):
        self.window = int(window)
        self.name = f"window[{self.window}]"
        self.display = f"Last {self.window} exams frequency"

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        Y = ctx.panel.until(t).Y
        if Y.shape[0] == 0:
            return ModelOutput(np.zeros(ctx.panel.K))
        return ModelOutput(Y[-self.window:].mean(axis=0) + 1e-3 * _feat(ctx, t, "freq_all"))


class EwmaModel(BaseModel):
    family, complexity, hidden = "baseline", 2, True

    def __init__(self, half_life: float):
        self.half_life = float(half_life)
        self.name = f"ewma[{self.half_life:g}]"
        self.display = f"Recency-weighted frequency (half-life {self.half_life:g} exams)"

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        Y = ctx.panel.until(t).Y
        if Y.shape[0] == 0:
            return ModelOutput(np.zeros(ctx.panel.K))
        return ModelOutput(dyn.ewma(Y, self.half_life) + 1e-4 * Y.mean(axis=0))


class LinearDecayModel(BaseModel):
    name, display, family, complexity = "linear_decay", "Linear-decay weighted frequency", "baseline", 2
    description = "Older exams count linearly less."

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        return ModelOutput(_feat(ctx, t, "linear_decay"))


class FreqRecencyModel(BaseModel):
    name, display, family, complexity = "freq_recency", "Frequency + recency", "baseline", 2
    description = "Average of the frequency rank and the recency-weighted rank."

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        a = percentile_rank(_feat(ctx, t, "freq_all"))
        b = percentile_rank(_feat(ctx, t, "ewma_short"))
        return ModelOutput(0.5 * a + 0.5 * b)


class BetaBinomialModel(BaseModel):
    name, display, family, complexity = "beta_binomial", "Bayesian rate (empirical Bayes)", "bayesian", 3
    description = ("Recency-weighted appearance rate per topic, shrunk toward the pooled rate of all topics, "
                   "so a topic seen once in three exams is not treated as a 33% certainty.")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        Y = ctx.panel.until(t).Y
        T = Y.shape[0]
        if T == 0:
            return ModelOutput(np.zeros(ctx.panel.K))
        cfg = ctx.settings.temporal
        h = float(cfg.default_half_life)
        ages = np.arange(T - 1, -1, -1, dtype=float)
        w = 0.5 ** (ages / h)
        hits = w @ Y
        n = w.sum()
        p0 = float(Y.mean())
        m = float(cfg.rate_prior_strength)
        post = (hits + m * p0) / (n + m)
        return ModelOutput(post, {"pooled_rate": p0, "prior_strength": m})


class MarkovModel(BaseModel):
    name, display, family, complexity = "markov", "Two-state Markov (empirical Bayes)", "temporal", 3
    description = ("P(appear | appeared or not in the previous exam), per topic, shrunk toward the pooled "
                   "transition rates. Captures 'just tested, less likely again' or 'tends to persist'.")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        return ModelOutput(_feat(ctx, t, "markov_next") + 1e-3 * _feat(ctx, t, "ewma_short"))


class HazardModel(BaseModel):
    name, display, family, complexity = "hazard", "Pooled hazard (time since last appearance)", "survival", 3
    description = ("Discrete-time hazard: chance of appearing given the number of exams since the topic last "
                   "appeared, pooled across topics and scaled by each topic's own rate.")

    def gate(self, panel, ctx):
        need = int(ctx.settings.models.sufficiency.min_exams_hazard)
        if panel.T < need:
            return False, f"Needs at least {need} exams to estimate gap-specific hazards (have {panel.T})."
        return True, ""

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        hazard = _feat(ctx, t, "hazard")
        Y = ctx.panel.until(t).Y
        if Y.shape[0] == 0:
            return ModelOutput(np.zeros(ctx.panel.K))
        m = float(ctx.settings.temporal.rate_prior_strength)
        p0 = float(Y.mean()) + 1e-9
        rate = (Y.sum(axis=0) + m * p0) / (Y.shape[0] + m)
        return ModelOutput(np.clip(hazard * np.sqrt(rate / p0), 0.0, 1.0))


class SemanticModel(BaseModel):
    name, display, family, complexity = "semantic", "Semantic soft recurrence", "semantic", 3
    description = ("Recency-weighted similarity between each topic and every past question, so near-miss "
                   "and multi-topic questions also count.")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        return ModelOutput(_feat(ctx, t, "soft_ewma") + 0.1 * _feat(ctx, t, "ewma_long"))
