"""Builds the list of baselines, ensemble components and the evidence-aware ensemble from settings."""

from __future__ import annotations

from ..config.settings import Settings
from .base import BaseModel
from .components import (BayesianRecurrenceModel, CooccurrenceModel, CourseLogisticModel, CoverageModel,
                         GeneralRankingModel, QuestionTypeModel, SemanticComponent, TemporalModel)
from .learned import GradientBoostingModel, PooledHMMModel, RandomForestModel
from .meta import EvidenceEnsemble, TunedModel
from .statistical import (EwmaModel, FreqRecencyModel, FrequencyModel, LastExamModel, LinearDecayModel, MarkovModel,
                          RandomModel, WindowModel)

FINAL_MODEL = "ensemble"
# Order matters only for display; meta models always run after the models they combine.
COMPONENT_ORDER = ["recency", "beta_binomial", "semantic", "coverage", "question_type", "cooccurrence", "hazard",
                   "markov", "hmm", "general", "logistic", "gradient_boosting", "random_forest"]


def _shown(model: BaseModel, visible: bool) -> BaseModel:
    """A baseline built only as a variant of another model is hidden from reports."""
    if not visible:
        model.hidden = True
    return model


def build_models(settings: Settings) -> list[BaseModel]:
    enabled = set(settings.models.enabled)
    models: list[BaseModel] = []
    by_name: dict[str, BaseModel] = {}

    def add(m: BaseModel) -> None:
        models.append(m)
        by_name[m.name] = m

    def want(name: str) -> bool:
        return name in enabled

    windows = sorted({int(w) for w in settings.temporal.windows})
    hls = sorted({float(h) for h in settings.temporal.half_lives} | {float(settings.temporal.default_half_life)})
    default_ewma = EwmaModel(float(settings.temporal.default_half_life)).name
    need_variants = want("window") or want("ewma") or want("recency")

    if want("random"):
        add(RandomModel())
    if want("frequency") or want("recency"):
        add(_shown(FrequencyModel(), want("frequency")))
    if want("last_exam"):
        add(LastExamModel())
    if need_variants:
        for w in windows:
            add(WindowModel(w))
        for h in hls:
            add(EwmaModel(h))
    if want("linear_decay") or want("recency"):
        add(_shown(LinearDecayModel(), want("linear_decay")))
    if want("window"):
        default = f"window[{3 if 3 in windows else windows[len(windows) // 2]}]"
        add(TunedModel("window", "Recent-window frequency", [f"window[{w}]" for w in windows], default,
                       description="Share of the last N exams with the topic; N chosen on earlier exams."))
    if want("ewma"):
        add(TunedModel("ewma", "Recency-weighted frequency", [EwmaModel(h).name for h in hls], default_ewma,
                       description="Exponentially decaying weights on older exams; half-life chosen on earlier exams."))
    if want("freq_recency"):
        add(FreqRecencyModel())
    if want("recency"):
        variants = ["frequency", "linear_decay"] + [f"window[{w}]" for w in windows] + [EwmaModel(h).name for h in hls]
        add(TunedModel("recency", "Recency-frequency (decay chosen by backtest)", variants, default_ewma,
                       description=("Topic frequency with a decay strategy (none, linear, last-N window or exponential "
                                    "half-life) chosen on earlier papers; the default is kept unless another strategy "
                                    "is reliably better."), family="recency", role="component", df=1.0))
    if want("beta_binomial"):
        half_lives = [None] + [float(h) for h in settings.temporal.bayes_half_lives]
        variants = [BayesianRecurrenceModel(h) for h in half_lives]
        for v in variants:
            add(v)
        # Default before a course's own papers show otherwise: half-life 6 exams. On 300 simulated courses
        # (seed 4242, not used for training) it beat no decay by 0.003 NDCG (standard error 0.001).
        default = next((v.name for v in variants if v.half_life == 6.0), variants[0].name)
        add(TunedModel("beta_binomial", "Hierarchical Bayesian recurrence", [v.name for v in variants],
                       default, complexity=3,
                       description=("Beta-binomial over exams with partial pooling (course, unit, topic) and an "
                                    "empirical-Bayes prior strength; recency discount chosen on earlier papers. Gives "
                                    "a posterior mean, credible interval and the share of the estimate that comes "
                                    "from the prior."), family="bayesian", role="component", df=1.0))
    if want("markov"):
        add(MarkovModel())
    if want("hazard"):
        add(TemporalModel())
    if want("semantic"):
        add(SemanticComponent())
    if want("coverage"):
        add(CoverageModel())
    if want("question_type"):
        add(QuestionTypeModel())
    if want("cooccurrence"):
        add(CooccurrenceModel())
    if want("general"):
        add(GeneralRankingModel())
    if want("logistic"):
        add(CourseLogisticModel())
    if want("random_forest"):
        add(RandomForestModel())
    if want("gradient_boosting"):
        add(GradientBoostingModel())
    if want("hmm"):
        add(PooledHMMModel())
    if want("ensemble"):
        cfg = settings.ensemble
        members = [n for n in COMPONENT_ORDER if n in by_name]
        add(EvidenceEnsemble(members, {n: by_name[n].df for n in members},
                             {n for n in members if by_name[n].prior_knowledge},
                             {n: by_name[n].display for n in members}, tau=float(cfg.skill_temperature),
                             prior_folds=float(cfg.skill_prior_folds), min_sd=float(cfg.min_metric_sd),
                             gate_strength=float(cfg.gate_strength)))
    return models
