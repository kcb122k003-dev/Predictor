"""Members of the evidence-aware ensemble.

Each component answers "how likely is this topic in the next exam?" from one kind of
evidence, using only exams before the target. None of them has a minimum dataset size.
A component that cannot run because an input does not exist (for example co-occurrence
before two papers exist, or no syllabus units) says so in ``info["available"]`` and the
ensemble leaves it out for that target.

* ``BayesianRecurrenceModel``  hierarchical beta-binomial (course -> unit -> topic)
* ``TemporalModel``            gap-dependent hazard, or the simpler Bayesian recurrence when
                                the hazard is not justified by prequential evidence
* ``SemanticComponent``        pretrained/hybrid semantic soft counts and semantic neighbours
* ``CoverageModel``            syllabus structure: unit-level coverage x syllabus weight
* ``QuestionTypeModel``        fit between a topic's usual formats and the course's recent mix
* ``CooccurrenceModel``        topics that tend to follow the previous paper's topics
* ``GeneralRankingModel``      cross-course model on scale-free features (no course data)
* ``CourseLogisticModel``      course-specific logistic regression pulled toward the general model
"""

from __future__ import annotations

import numpy as np

from ..features.builder import FEATURE_NAMES, GENERIC_FEATURES
from ..inference.bayes import fit_recurrence, prequential_log_scores
from ..inference.generic import fit_map_logistic, generic_columns
from ..prediction.type_forecast import global_mix, topic_mix
from ..temporal import dynamics as dyn
from ..temporal.panel import FORMATS
from .base import BaseModel, ModelContext, ModelOutput, percentile_rank


def unavailable(K: int, reason: str) -> ModelOutput:
    return ModelOutput(np.zeros(K), {"available": False, "unavailable_reason": reason})


# ------------------------------------------------------------------------- Bayesian recurrence
class BayesianRecurrenceModel(BaseModel):
    family, complexity, hidden, role = "bayesian", 3, True, "variant"
    df = 1.0

    def __init__(self, half_life: float | None):
        self.half_life = half_life
        label = "inf" if half_life is None else f"{half_life:g}"
        self.name = f"bayes[{label}]"
        self.display = ("Hierarchical Bayesian recurrence (no decay)" if half_life is None else
                        f"Hierarchical Bayesian recurrence (half-life {half_life:g} exams)")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        hist = ctx.panel.until(t)
        units = hist.item_unit if hist.item_unit is not None and len(hist.item_unit) == hist.K else None
        post = fit_recurrence(hist.Y, units, self.half_life)
        info = {"half_life": self.half_life, "kappa": round(post.kappa, 4), "n_eff": round(post.n_eff, 3),
                "course_rate": round(post.course_rate, 4), "kappa_source": post.kappa_source,
                "posterior": post}
        # Tiny tie-break by the upper credible bound: with equal means, the less certain topic ranks
        # higher only by a hair (it cannot overtake a topic with a higher mean).
        return ModelOutput(post.mean + 1e-6 * post.high, info)


# ------------------------------------------------------------------------- temporal recurrence
def _hazard_predict(Y: np.ndarray, prior_strength: float, rate_prior: float) -> np.ndarray:
    T, K = Y.shape
    if T == 0:
        return np.full(K, 0.5)
    haz = dyn.pooled_hazard(Y, prior_strength)
    since = dyn.since_last(Y)
    seen = Y.sum(axis=0) > 0
    h = haz.at(np.where(seen, since, 0), seen)
    p0 = float(Y.mean()) + 1e-9
    rate = (Y.sum(axis=0) + rate_prior * p0) / (T + rate_prior)
    return np.clip(h * np.sqrt(rate / p0), 1e-4, 1 - 1e-4)


def _simple_predict(Y: np.ndarray) -> np.ndarray:
    T, K = Y.shape
    if T == 0:
        return np.full(K, 0.5)
    return fit_recurrence(Y).mean


class TemporalModel(BaseModel):
    """Gap-dependent hazard when the history supports it, otherwise simplified Bayesian recurrence.

    The choice is made for every target from the prequential log score of exams before the
    target: each earlier exam s is predicted from the exams before s by both models. The
    hazard model is used only if its cumulative log score is higher (a positive log Bayes
    factor). Its extra parameters cost it on early exams, so it has to earn its place.
    """

    name, display, family, complexity, role = "hazard", "Temporal recurrence (hazard if justified)", "survival", 3, "component"
    df = 3.0
    description = ("Time since a topic last appeared. Uses a gap-specific hazard only when earlier papers show that "
                   "gaps predict better than a constant rate; otherwise falls back to the Bayesian recurrence rate.")

    def _scores(self, ctx: ModelContext) -> tuple[np.ndarray, np.ndarray]:
        key = "prequential_hazard"
        if key not in ctx.cache:
            Y = ctx.panel.Y
            m = float(ctx.settings.temporal.hazard_prior_strength)
            r = float(ctx.settings.temporal.rate_prior_strength)
            ctx.cache[key] = (prequential_log_scores(Y, lambda h: _hazard_predict(h, m, r)),
                              prequential_log_scores(Y, _simple_predict))
        return ctx.cache[key]

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        hist = ctx.panel.until(t)
        haz_ls, simple_ls = self._scores(ctx)
        lo = min(t, ctx.panel.T)
        d = float(np.nansum(haz_ls[1:lo] - simple_ls[1:lo])) if lo > 1 else 0.0
        use_hazard = d > 0.0
        m = float(ctx.settings.temporal.hazard_prior_strength)
        r = float(ctx.settings.temporal.rate_prior_strength)
        scores = _hazard_predict(hist.Y, m, r) if use_hazard else _simple_predict(hist.Y)
        info = {"mode": "gap hazard" if use_hazard else "simplified Bayesian recurrence",
                "log_bayes_factor": round(d, 3), "exams_compared": max(lo - 1, 0)}
        return ModelOutput(scores + 1e-4 * _feat(ctx, t, "ewma_short"), info)


def _feat(ctx: ModelContext, t: int, name: str) -> np.ndarray:
    return ctx.store.at(t).column(name)


# ------------------------------------------------------------------------- semantic
class SemanticComponent(BaseModel):
    name, display, family, complexity, role = "semantic", "Semantic evidence (pretrained)", "semantic", 3, "component"
    df = 0.5
    prior_knowledge = True
    scope = "global"
    description = ("Each past in-syllabus question spreads its evidence over topics by semantic similarity (bundled "
                   "pretrained model plus syllabus TF-IDF), so near-misses and multi-topic questions count a little. "
                   "Topics semantically close to recently tested ones get a small share too.")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        K = ctx.panel.K
        if ctx.panel.meta.get("semantic_source") in (None, "none") and not np.any(ctx.panel.soft):
            return unavailable(K, "No semantic similarity scores are available for this layer.")
        if t == 0:
            return unavailable(K, "Semantic recurrence needs at least one earlier paper to compare with.")
        sem = _feat(ctx, t, "sem_ewma")
        soft = _feat(ctx, t, "soft_ewma")
        neigh = _feat(ctx, t, "sem_neighbors")
        score = 0.6 * percentile_rank(sem) + 0.2 * percentile_rank(soft) + 0.2 * percentile_rank(neigh)
        return ModelOutput(score + 1e-6 * sem, {"source": ctx.panel.meta.get("semantic_source", "alignment scores")})


# ------------------------------------------------------------------------- syllabus coverage
def syllabus_share(panel) -> tuple[np.ndarray, str]:
    """Prior share of each item from the syllabus: hours, then marks weight, then breadth."""
    K = panel.K
    for key, label in (("hours_share", "teaching hours"), ("marks_weight_share", "marks weights")):
        v = panel.static.get(key)
        if v is not None and len(v) == K and np.any(v > 0):
            return np.asarray(v, dtype=float), label
    v = panel.static.get("breadth")
    if v is not None and len(v) == K and np.ptp(v) > 0:
        return np.asarray(v, dtype=float) + 0.5, "breadth (sub-topics and concepts)"
    return np.ones(K), "uniform"


class CoverageModel(BaseModel):
    name, display, family, complexity, role = "coverage", "Syllabus coverage structure", "structure", 2, "component"
    df = 0.5
    prior_knowledge = True
    description = ("How often each syllabus unit is examined (Bayesian, at the exam level) times the topic's weight "
                   "in its unit from the syllabus (hours, marks or breadth). Works before any paper is uploaded.")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        panel = ctx.panel
        K = panel.K
        share, source = syllabus_share(panel)
        units = panel.item_unit if panel.item_unit is not None and len(panel.item_unit) == K else np.zeros(K, int)
        n_units = len(np.unique(units)) if K else 0
        if n_units <= 1 and source == "uniform":
            return unavailable(K, "The syllabus has a single unit and no hours, marks or sub-topic structure.")
        Y = panel.until(t).Y
        U = int(units.max()) + 1 if K else 0
        unit_Y = np.zeros((Y.shape[0], U))
        for k in range(K):
            unit_Y[:, units[k]] = np.maximum(unit_Y[:, units[k]], Y[:, k])
        unit_rate = fit_recurrence(unit_Y).mean if U else np.zeros(0)
        within = np.zeros(K)
        for u in range(U):
            members = units == u
            if members.any():
                within[members] = share[members] / share[members].mean()
        score = unit_rate[units] * np.sqrt(np.clip(within, 1e-6, None))
        return ModelOutput(score, {"weight_source": source, "units": n_units})


# ------------------------------------------------------------------------- question type
class QuestionTypeModel(BaseModel):
    name, display, family, complexity, role = "question_type", "Question-type fit", "structure", 2, "component"
    df = 1.0
    description = ("Compatibility between a topic's usual question formats (its history, smoothed toward syllabus tags) "
                   "and the format mix of recent papers.")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        panel = ctx.panel
        K = panel.K
        hist = panel.until(t)
        F = hist.formats
        h = float(ctx.settings.temporal.default_half_life)
        prior_tags = panel.format_prior if panel.format_prior is not None and panel.format_prior.shape == (K, len(FORMATS)) else None
        observed = F.sum(axis=(0, 1)) if F.size else np.zeros(len(FORMATS))
        if (observed > 0).sum() < 2 and (prior_tags is None or not np.any(np.ptp(prior_tags, axis=0) > 0)):
            return unavailable(K, "Past papers use a single question format and the syllabus has no format tags.")
        g = global_mix(F, h)
        tm = topic_mix(F, h, g)
        if prior_tags is not None:
            tags = prior_tags / np.maximum(prior_tags.sum(axis=1, keepdims=True), 1e-9)
            has = prior_tags.sum(axis=1) > 0
            seen = F.sum(axis=(0, 2)) if F.size else np.zeros(K)
            weight = np.where(has, 1.0 / (1.0 + seen), 0.0)[:, None]  # tags matter most for unseen topics
            tm = (1 - weight) * tm + weight * tags
        score = tm @ g
        return ModelOutput(score, {"recent_mix": {FORMATS[i]: round(float(v), 3) for i, v in enumerate(g)}})


# ------------------------------------------------------------------------- co-occurrence
class CooccurrenceModel(BaseModel):
    name, display, family, complexity, role = "cooccurrence", "Topic co-occurrence", "structure", 3, "component"
    df = 3.0
    description = "Topics that, in this course, tend to follow the topics of the previous paper (smoothed lift)."

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        K = ctx.panel.K
        if t < 2:
            return unavailable(K, "Needs two consecutive earlier papers to see which topics follow which.")
        lift = _feat(ctx, t, "cooc_lift_last")
        return ModelOutput(lift + 1e-3 * _feat(ctx, t, "ewma_short"), {})


# ------------------------------------------------------------------------- general and course models
class GeneralRankingModel(BaseModel):
    name, display, family, complexity, role = "general", "General ranking model (cross-course)", "transfer", 2, "component"
    df = 0.0
    prior_knowledge = True
    scope = "global"
    description = ("Logistic model on scale-free recurrence features, trained on simulated examiner behaviour and "
                   "updated with other real courses in your library (never this course). Needs no course data.")

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        gm = ctx.general_model()
        X = ctx.store.at(t).X[:, generic_columns()]
        return ModelOutput(gm.predict(X), {"source": gm.source, "summary": gm.summary},
                           gm.contributions(X), list(GENERIC_FEATURES))


COURSE_ONLY = ["count_log", "soft_ewma", "soft_mean", "sem_ewma", "sem_neighbors", "marks_share", "marks_share_recent",
               "high_mark_rate", "numerical_share", "derivation_share", "theory_share", "type_entropy",
               "hours_share", "marks_weight_share", "breadth", "exact_repeat_rate", "para_repeat_rate"]


class CourseLogisticModel(BaseModel):
    """Course-specific logistic regression, hierarchical over courses.

    Weights of the scale-free features have a Gaussian prior centred on the general model's
    weights; course-only features (text, marks, syllabus) have a prior centred on zero. With
    few rows the model stays close to the general model; with many rows the course data
    dominate. No minimum number of rows: zero rows return the general model unchanged.
    """

    family, complexity, role = "ml", 4, "component"
    description = ("Logistic regression trained on this course's own earlier papers, pulled toward the general "
                   "model so that a handful of papers cannot produce extreme weights.")

    def __init__(self, exclude: tuple[str, ...] = (), name: str = "logistic",
                 display: str = "Course-specific logistic (pooled toward the general model)"):
        self.name, self.display = name, display
        self.exclude = set(exclude)
        self.hidden = bool(exclude)
        self.gen_cols = [i for i, f in zip(generic_columns(), GENERIC_FEATURES) if f not in self.exclude]
        self.gen_idx = [j for j, f in enumerate(GENERIC_FEATURES) if f not in self.exclude]
        self.course_cols = [FEATURE_NAMES.index(f) for f in COURSE_ONLY if f not in self.exclude]
        self.df = 0.5 * (len(self.gen_cols) + len(self.course_cols))

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        gm = ctx.general_model()
        precision = float(ctx.settings.models.course_prior_precision)
        min_hist = max(1, int(ctx.settings.models.min_history))
        cols = self.gen_cols + self.course_cols
        X, y, _ = ctx.store.training_rows(t, min_hist, cols)
        ng = len(self.gen_cols)
        mean_g, std_g = gm.mean[self.gen_idx], gm.std[self.gen_idx]
        if len(y):
            mean_c = X[:, ng:].mean(axis=0)
            std_c = X[:, ng:].std(axis=0)
        else:
            mean_c = np.zeros(len(self.course_cols))
            std_c = np.ones(len(self.course_cols))
        std_c = np.where(std_c < 1e-9, 1.0, std_c)
        mean = np.concatenate([mean_g, mean_c])
        std = np.concatenate([std_g, std_c])
        prior = np.concatenate([gm.coef[self.gen_idx], np.zeros(len(self.course_cols))])
        Z = (X - mean) / std if len(y) else np.zeros((0, len(cols)))
        coef, b = fit_map_logistic(Z, y, prior, gm.intercept, precision)
        x = ctx.store.at(t).X[:, cols]
        z = (x - mean) / std
        prob = 1.0 / (1.0 + np.exp(-np.clip(z @ coef + b, -30, 30)))
        names = [FEATURE_NAMES[i] for i in cols]
        info = {"train_rows": int(len(y)), "train_exams": int(max(t - min_hist, 0)), "intercept": round(b, 4),
                "prior_precision": precision,
                "coefficients": {n: round(float(c), 4) for n, c in zip(names, coef)},
                "shift_from_general": round(float(np.linalg.norm(coef - prior)), 4)}
        return ModelOutput(prob, info, z * coef[None, :], names)


def has_both_classes(y: np.ndarray) -> bool:
    return len(y) > 0 and 0 < float(np.sum(y)) < len(y)


def general_fallback(t: int, ctx: ModelContext, reason: str) -> ModelOutput:
    """Used only when a model is mathematically impossible to fit (e.g. one class in the labels)."""
    gm = ctx.general_model()
    X = ctx.store.at(t).X[:, generic_columns()]
    return ModelOutput(gm.predict(X), {"fallback": True, "fallback_reason": reason})


__all__ = ["BayesianRecurrenceModel", "TemporalModel", "SemanticComponent", "CoverageModel", "QuestionTypeModel",
           "CooccurrenceModel", "GeneralRankingModel", "CourseLogisticModel", "has_both_classes", "general_fallback",
           "syllabus_share", "unavailable"]
