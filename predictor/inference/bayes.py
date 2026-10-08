"""Hierarchical Bayesian recurrence: course -> unit -> topic, at the exam level.

Every quantity here counts exams, never questions. A topic asked three times in one paper
is one appearance in one exam, so long papers cannot inflate the temporal sample size.

Model, for topic k in unit u, with exams weighted by recency (weight 1 for the latest exam,
halving every ``half_life`` exams; ``half_life=None`` means no decay):

    n_eff        = sum of exam weights                       (same for every topic)
    h_k          = weighted number of exams containing k
    mu_course    ~ generic prior rate, updated by all topics' appearances
    mu_unit      = unit rate shrunk toward mu_course
    p_k          ~ Beta(mu_unit * kappa, (1 - mu_unit) * kappa)
    h_k | p_k    ~ Binomial(n_eff, p_k)                      (continuous counts)

``kappa`` (how strongly a topic is pulled toward its unit) is estimated by empirical Bayes:
it maximises the beta-binomial marginal likelihood of all topics' counts, with a weak
log-normal hyperprior that dominates when there are only one or two exams. The posterior
for each topic is Beta(mu_unit * kappa + h_k, (1 - mu_unit) * kappa + n_eff - h_k).

From the posterior we keep the mean, median, a central credible interval, the effective
sample size (kappa + n_eff), the prior contribution kappa / (kappa + n_eff) and the raw
observed evidence (appearances, exams). One appearance in two exams and eight in sixteen
have similar means but very different intervals, and the ensemble uses that difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import optimize, special, stats

GENERIC_PRIOR_RATE = 0.4   # typical share of syllabus topics that appear in one exam
GENERIC_PRIOR_KAPPA = 2.0  # pseudo-exams of pull toward the unit/course rate before any data
KAPPA_LOG_SD = 1.0


def recency_weights(T: int, half_life: float | None) -> np.ndarray:
    if T <= 0:
        return np.zeros(0)
    if half_life is None or not math.isfinite(half_life) or half_life <= 0:
        return np.ones(T)
    ages = np.arange(T - 1, -1, -1, dtype=float)
    return 0.5 ** (ages / half_life)


@dataclass
class RecurrencePosterior:
    mean: np.ndarray
    median: np.ndarray
    low: np.ndarray
    high: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    hits: np.ndarray          # raw number of exams containing the topic
    exams: int                # raw number of exams
    n_eff: float              # recency-weighted number of exams
    kappa: float
    prior_mean: np.ndarray    # unit-level prior mean per topic
    prior_contribution: np.ndarray   # weight of the prior mean in the posterior mean, kappa / (kappa + n_eff)
    variance_ratio: np.ndarray       # posterior variance / prior variance per topic (1 = data taught nothing)
    ess: np.ndarray
    course_rate: float
    unit_rates: dict[int, float] = field(default_factory=dict)
    half_life: float | None = None
    interval: float = 0.8
    kappa_source: str = ""

    def summary(self, k: int) -> dict[str, Any]:
        return {
            "posterior_mean": round(float(self.mean[k]), 4), "posterior_median": round(float(self.median[k]), 4),
            "credible_interval": [round(float(self.low[k]), 4), round(float(self.high[k]), 4)],
            "interval_level": self.interval, "effective_sample_size": round(float(self.ess[k]), 2),
            "prior_contribution": round(float(self.prior_contribution[k]), 3),
            "variance_ratio": round(float(self.variance_ratio[k]), 3),
            "prior_mean": round(float(self.prior_mean[k]), 4),
            "observed": {"appearances": int(self.hits[k]), "exams": int(self.exams)},
            "half_life": self.half_life, "kappa": round(self.kappa, 3),
        }

    def draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, K) samples from each topic's posterior."""
        return rng.beta(self.alpha[None, :], self.beta[None, :], size=(n, len(self.alpha)))


def _bb_loglik(log_kappa: float, h: np.ndarray, n: float, mu: np.ndarray, kappa0: float, sd: float) -> float:
    kappa = math.exp(log_kappa)
    a = np.clip(mu * kappa, 1e-6, None)
    b = np.clip((1.0 - mu) * kappa, 1e-6, None)
    ll = float(np.sum(special.betaln(h + a, n - h + b) - special.betaln(a, b)))
    prior = -0.5 * ((log_kappa - math.log(kappa0)) / sd) ** 2
    return ll + prior


def estimate_kappa(h: np.ndarray, n: float, mu: np.ndarray, kappa0: float = GENERIC_PRIOR_KAPPA,
                   sd: float = KAPPA_LOG_SD) -> tuple[float, str]:
    if n <= 0 or len(h) < 2:
        return kappa0, "generic prior (no exams yet)"
    res = optimize.minimize_scalar(lambda lk: -_bb_loglik(lk, h, n, mu, kappa0, sd),
                                   bounds=(math.log(0.05), math.log(500.0)), method="bounded",
                                   options={"xatol": 1e-4})
    kappa = float(math.exp(res.x))
    return kappa, "empirical Bayes (marginal likelihood with a weak generic hyperprior)"


def fit_recurrence(Y: np.ndarray, item_unit: np.ndarray | None = None, half_life: float | None = None,
                   prior_rate: float = GENERIC_PRIOR_RATE, prior_kappa: float = GENERIC_PRIOR_KAPPA,
                   interval: float = 0.8) -> RecurrencePosterior:
    """Posterior appearance probability of every item at the next exam."""
    Y = np.asarray(Y, dtype=float)
    T, K = Y.shape
    w = recency_weights(T, half_life)
    n_eff = float(w.sum())
    h = w @ Y if T else np.zeros(K)
    raw_hits = Y.sum(axis=0) if T else np.zeros(K)
    # Course rate: generic prior worth two exams of every topic, updated by the data.
    a0, b0 = 2.0 * prior_rate, 2.0 * (1.0 - prior_rate)
    course = float((h.sum() + a0 * max(K, 1)) / (n_eff * K + (a0 + b0) * max(K, 1))) if K else prior_rate
    mu = np.full(K, course)
    unit_rates: dict[int, float] = {}
    if item_unit is not None and len(item_unit) == K and K:
        units = np.asarray(item_unit, dtype=int)
        for u in np.unique(units):
            members = units == u
            # Unit rate: the unit's own topic-exam record, shrunk toward the course rate.
            m = prior_kappa * 2.0
            rate = float((h[members].sum() + m * course) / (n_eff * members.sum() + m))
            unit_rates[int(u)] = rate
            mu[members] = rate
    mu = np.clip(mu, 1e-3, 1 - 1e-3)
    kappa, source = estimate_kappa(h, n_eff, mu, prior_kappa)
    alpha = mu * kappa + h
    beta_ = (1.0 - mu) * kappa + (n_eff - h)
    beta_ = np.clip(beta_, 1e-6, None)
    alpha = np.clip(alpha, 1e-6, None)
    mean = alpha / (alpha + beta_)
    lo_q, hi_q = (1 - interval) / 2, 1 - (1 - interval) / 2
    median = stats.beta.median(alpha, beta_)
    low = stats.beta.ppf(lo_q, alpha, beta_)
    high = stats.beta.ppf(hi_q, alpha, beta_)
    ess = alpha + beta_
    prior_contribution = np.full(K, kappa / (kappa + n_eff)) if K else np.zeros(0)
    prior_var = mu * (1 - mu) / (kappa + 1.0)
    post_var = alpha * beta_ / ((alpha + beta_) ** 2 * (alpha + beta_ + 1.0))
    variance_ratio = np.clip(post_var / np.maximum(prior_var, 1e-12), 0.0, 1.0)
    return RecurrencePosterior(mean=mean, median=np.asarray(median), low=np.asarray(low), high=np.asarray(high),
                               alpha=alpha, beta=beta_, hits=raw_hits, exams=T, n_eff=n_eff, kappa=kappa,
                               prior_mean=mu, prior_contribution=prior_contribution, variance_ratio=variance_ratio,
                               ess=ess, course_rate=course,
                               unit_rates=unit_rates, half_life=half_life, interval=interval, kappa_source=source)


# ------------------------------------------------------------------ prequential model comparison
def prequential_log_scores(Y: np.ndarray, predict: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Log score of predicting each exam s >= 1 from exams before s: entry s is sum_k log p(Y[s, k]).

    Entry 0 is NaN (nothing to predict from). The sum of entries 1..t-1 is the sequential
    (prequential) log marginal likelihood of exams 1..t-1, which compares models fairly:
    a model with more parameters pays for them by predicting early exams badly.
    """
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    out = np.full(T, np.nan)
    for s in range(1, T):
        p = np.clip(np.asarray(predict(Y[:s]), dtype=float), 1e-4, 1 - 1e-4)
        out[s] = float(np.sum(Y[s] * np.log(p) + (1 - Y[s]) * np.log(1 - p)))
    return out


def evidence_label(variance_ratio: float) -> str:
    """How much a topic's own history has narrowed its estimate.

    ``variance_ratio`` is posterior variance / prior variance for that topic: 1 means the papers
    taught nothing about it, values near 0 mean its rate is well measured (that includes a topic
    absent from many papers, which is strong evidence that it is rarely tested).
    """
    if variance_ratio < 0.3:
        return "Strong"
    if variance_ratio < 0.5:
        return "Moderate"
    if variance_ratio < 0.75:
        return "Limited"
    return "Minimal"
