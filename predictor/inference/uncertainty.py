"""Per-topic uncertainty: rank intervals, posterior intervals and evidence strength.

Two sources of uncertainty are combined into a rank interval for every topic:

1. Which papers happen to exist. Each past paper is left out in turn (jackknife) and the
   final ranking is recomputed with the ensemble weights held fixed. Components that refit
   classifiers (course logistic, trees, HMM) keep their full-data scores to keep this fast;
   the report says so.
2. Posterior uncertainty of each topic's appearance rate: rankings are recomputed with the
   Bayesian recurrence component replaced by draws from its posterior.

The interval is the wider of the jackknife range and the 10th-90th percentile of the
posterior-draw ranks. Uncertainty level: Low when the interval spans less than a fifth of
the topics, Medium below two fifths, High otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata

from ..features.builder import FeatureStore
from ..models.base import ModelContext, percentile_rank
from .bayes import evidence_label

SLOW = {"logistic", "gradient_boosting", "random_forest", "hmm"}


def _ranks(scores: np.ndarray) -> np.ndarray:
    return rankdata(-np.asarray(scores, dtype=float), method="average")


def _member_model(name: str, report, models_by_name: dict) -> Any:
    info = report.final_outputs.get(name).info if name in report.final_outputs else {}
    chosen = info.get("chosen")
    return models_by_name.get(chosen) if chosen else models_by_name.get(name)


def topic_uncertainty(panel, settings, report, models: list, general=None, draws: int = 200,
                      seed: int = 13) -> dict[str, Any]:
    K = panel.K
    T = panel.T
    by_name = {m.name: m for m in models}
    sel = report.selected
    final = np.asarray(report.final_scores, dtype=float)
    base_ranks = _ranks(final)
    samples: list[np.ndarray] = []
    jack: list[np.ndarray] = []
    ens = report.final_outputs.get("ensemble")
    use_ensemble = sel == "ensemble" and ens is not None and ens.info.get("members")
    held_fixed: list[str] = []

    def combine(member_scores: dict[str, np.ndarray], posterior) -> np.ndarray:
        info = ens.info
        members = info["members"]
        u = np.asarray(posterior.variance_ratio, dtype=float) if posterior is not None else np.ones(K)
        model = by_name["ensemble"]
        P = np.vstack([percentile_rank(member_scores[m]) for m in members])
        G = np.vstack([1.0 + model.gate_strength * u if m in model.prior_knowledge else np.ones(K) for m in members])
        W = np.array([info["weights"][m] for m in members])[:, None] * G
        W = W / W.sum(axis=0, keepdims=True)
        return (W * P).sum(axis=0)

    # 1. Jackknife over papers.
    for j in range(T):
        sub = panel.drop_exam(j)
        store = FeatureStore(sub, settings)
        ctx = ModelContext(panel=sub, store=store, settings=settings, seed=seed, general=general)
        t = sub.T
        try:
            if use_ensemble:
                scores = {}
                for m in ens.info["members"]:
                    if m in SLOW:
                        scores[m] = report.final_outputs[m].scores
                        if m not in held_fixed:
                            held_fixed.append(m)
                        continue
                    model = _member_model(m, report, by_name)
                    scores[m] = model.predict(t, ctx).scores if model is not None else report.final_outputs[m].scores
                s = combine(scores, store.at(t).extras.get("posterior"))
            else:
                model = _member_model(sel, report, by_name)
                s = model.predict(t, ctx).scores
        except Exception:  # pragma: no cover - a jackknife replicate must never break the analysis
            continue
        jack.append(_ranks(s))
    # 2. Posterior draws of the Bayesian recurrence component.
    rng = np.random.default_rng(seed)
    bb = report.final_outputs.get("beta_binomial")
    post = bb.info.get("posterior") if bb is not None else None
    if post is not None and use_ensemble and "beta_binomial" in ens.info["members"]:
        member_scores = {m: report.final_outputs[m].scores for m in ens.info["members"]}
        fm_post = report.ctx.store.at(T).extras.get("posterior") if report.ctx is not None else None
        for d in post.draws(draws, rng):
            member_scores["beta_binomial"] = d
            samples.append(_ranks(combine(member_scores, fm_post)))
    elif post is not None:
        for d in post.draws(draws, rng):
            samples.append(_ranks(d))
    jack_arr = np.vstack(jack) if jack else base_ranks[None, :]
    samp_arr = np.vstack(samples) if samples else base_ranks[None, :]
    q_lo, q_hi = np.percentile(samp_arr, [10, 90], axis=0)
    lo = np.minimum(jack_arr.min(axis=0), q_lo)
    hi = np.maximum(jack_arr.max(axis=0), q_hi)
    lo = np.minimum(lo, base_ranks)
    hi = np.maximum(hi, base_ranks)
    width = hi - lo
    level = np.where(width < 0.2 * K, "Low", np.where(width < 0.4 * K, "Medium", "High"))
    return {"rank_low": np.floor(lo).astype(int), "rank_high": np.ceil(hi).astype(int), "level": level.tolist(),
            "jackknife_replicates": len(jack), "posterior_draws": len(samples), "held_fixed": held_fixed}


def evidence_strength(post, k: int) -> str:
    if post is None:
        return "Minimal"
    return evidence_label(float(post.prior_contribution[k]))
