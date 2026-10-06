"""Builds the list of candidate models from settings."""

from __future__ import annotations

from ..config.settings import Settings
from .base import BaseModel
from .learned import GradientBoostingModel, LogisticModel, PooledHMMModel, RandomForestModel
from .meta import EnsembleModel, TunedModel
from .statistical import (BetaBinomialModel, EwmaModel, FreqRecencyModel, FrequencyModel, HazardModel,
                          LastExamModel, LinearDecayModel, MarkovModel, RandomModel, SemanticModel, WindowModel)

DEFAULT_FALLBACK = "beta_binomial"


def build_models(settings: Settings) -> list[BaseModel]:
    enabled = set(settings.models.enabled)
    models: list[BaseModel] = []

    def want(name: str) -> bool:
        return name in enabled

    if want("random"):
        models.append(RandomModel())
    if want("frequency"):
        models.append(FrequencyModel())
    if want("last_exam"):
        models.append(LastExamModel())
    if want("window"):
        windows = sorted({int(w) for w in settings.temporal.windows})
        variants = [WindowModel(w) for w in windows]
        models.extend(variants)
        default = f"window[{3 if 3 in windows else windows[len(windows) // 2]}]"
        models.append(TunedModel("window", "Recent-window frequency", [v.name for v in variants], default,
                                 description="Share of the last N exams with the topic; N chosen on earlier exams."))
    if want("ewma"):
        hls = sorted({float(h) for h in settings.temporal.half_lives} | {float(settings.temporal.default_half_life)})
        variants = [EwmaModel(h) for h in hls]
        models.extend(variants)
        default = EwmaModel(float(settings.temporal.default_half_life)).name
        models.append(TunedModel("ewma", "Recency-weighted frequency", [v.name for v in variants], default,
                                 description="Exponentially decaying weights on older exams; half-life chosen "
                                             "on earlier exams."))
    if want("linear_decay"):
        models.append(LinearDecayModel())
    if want("freq_recency"):
        models.append(FreqRecencyModel())
    if want("beta_binomial"):
        models.append(BetaBinomialModel())
    if want("markov"):
        models.append(MarkovModel())
    if want("hazard"):
        models.append(HazardModel())
    if want("semantic"):
        models.append(SemanticModel())
    if want("logistic"):
        models.append(LogisticModel())
    if want("random_forest"):
        models.append(RandomForestModel())
    if want("gradient_boosting"):
        models.append(GradientBoostingModel())
    if want("hmm"):
        models.append(PooledHMMModel())
    if want("ensemble"):
        candidates = [m.name for m in models if not m.hidden and m.name not in ("random",)]
        defaults = [n for n in ("ewma", "beta_binomial", "frequency") if n in candidates]
        models.append(EnsembleModel(candidates, defaults or candidates[:3], int(settings.models.ensemble_max_members)))
    return models
