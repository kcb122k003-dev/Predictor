"""Models that combine or choose among other models using earlier backtest folds only.

* ``TunedModel`` picks a hyperparameter (window length, half-life, decay strategy) for
  target t from the folds before t. It keeps the default unless another variant is ahead
  by more than one standard error of the paired difference, so two lucky folds cannot move it.
* ``EvidenceEnsemble`` is the final ranking model:

      S(x) = sum_m w_m(x) * S_m(x) / sum_m w_m(x)

  where S_m(x) is component m's percentile rank of topic x and

      w_m(x) = rho_m * exp(tau * skill_m) * g_m(x)

  rho_m    reliability prior: exams / (exams + df_m), where df_m is the number of parameters
           the component estimates from this course (0 for the cross-course model). Components
           that bring outside knowledge (pretrained semantics, syllabus structure) get one extra
           pseudo-exam. Exams are counted at the exam level; question counts never enter.
  skill_m  how much better than the average member it ranked earlier papers (folds s < t),
           in units of the within-paper spread between members, shrunk toward 0 by
           folds / (folds + n0).
  g_m(x)   per-topic gate: components that do not depend on the topic's own history
           (pretrained semantics, syllabus structure, cross-course model) get 1 + u(x), where
           u(x) is the Bayesian posterior-to-prior variance ratio of topic x (1 when its own
           history taught nothing, near 0 when it is well measured).

Stored predictions for target s were computed from exams before s, so choosing weights from
them for a later target t never looks at exam t.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..evaluation.metrics import mean_and_se
from .base import BaseModel, ModelContext, ModelOutput, percentile_rank


def _folds_before(ctx: ModelContext, name: str, t: int) -> dict[int, float]:
    return {s: v for s, v in ctx.fold_metric.get(name, {}).items() if s < t and v == v}


class TunedModel(BaseModel):
    meta = True

    def __init__(self, name: str, display: str, variants: list[str], default: str, complexity: int = 2,
                 description: str = "", family: str = "baseline", role: str = "baseline", df: float = 1.0):
        self.name, self.display, self.variants, self.default = name, display, variants, default
        self.family = family
        self.complexity = complexity
        self.description = description
        self.role = role
        self.df = df

    def choose(self, t: int, ctx: ModelContext) -> tuple[str, str]:
        if self.default not in ctx.predictions:
            available = [v for v in self.variants if v in ctx.predictions]
            self.default = available[0] if available else self.default
        best, best_mean = self.default, -math.inf
        folds = {v: _folds_before(ctx, v, t) for v in self.variants if v in ctx.predictions}
        for v, f in folds.items():
            if f:
                m = float(np.mean(list(f.values())))
                if m > best_mean + 1e-12:
                    best, best_mean = v, m
        if best == self.default or not folds.get(self.default):
            return self.default, "default (no earlier paper shows a reliably better variant)"
        common = [s for s in folds[best] if s in folds[self.default]]
        diff, se = mean_and_se([folds[best][s] - folds[self.default][s] for s in common])
        if math.isnan(se) or diff <= se:
            return self.default, (f"default kept: '{best}' was ahead by {diff:.3f} on {len(common)} earlier paper(s), "
                                  f"within one standard error")
        return best, f"'{best}' ahead of the default by {diff:.3f} ± {se:.3f} on {len(common)} earlier papers"

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        chosen, why = self.choose(t, ctx)
        inner = ctx.outputs.get(chosen, {}).get(t)
        info = dict(inner.info) if inner is not None else {}
        info.update({"chosen": chosen, "choice_reason": why})
        return ModelOutput(ctx.predictions[chosen][t], info,
                           inner.contributions if inner is not None else None,
                           inner.contribution_names if inner is not None else None)


class EvidenceEnsemble(BaseModel):
    name, display, family, complexity, meta, role = "ensemble", "Evidence-aware ensemble", "ensemble", 5, True, "ensemble"
    description = ("Combines every available component. Each weight multiplies a reliability prior (exams versus the "
                   "number of parameters the component estimates from this course) by the component's measured skill "
                   "on earlier papers; topics with little history lean more on pretrained, syllabus and cross-course "
                   "knowledge.")

    def __init__(self, members: list[str], df: dict[str, float], prior_knowledge: set[str],
                 displays: dict[str, str], tau: float = 2.0, prior_folds: float = 3.0, min_sd: float = 0.05,
                 gate_strength: float = 1.0):
        self.members = list(members)
        self.member_df = dict(df)
        self.df = 0.0
        self.prior_knowledge = set(prior_knowledge)
        self.displays = dict(displays)
        self.tau, self.n0, self.min_sd, self.gate_strength = tau, prior_folds, min_sd, gate_strength

    def subset(self, members: list[str], name: str) -> "EvidenceEnsemble":
        e = EvidenceEnsemble([m for m in self.members if m in members], self.member_df, self.prior_knowledge, self.displays,
                             self.tau, self.n0, self.min_sd, self.gate_strength)
        e.name = name
        e.hidden = True
        return e

    # ------------------------------------------------------------------ weights
    def weights(self, t: int, ctx: ModelContext) -> dict[str, Any]:
        exams = ctx.effective_exams(t)
        available, excluded = [], {}
        for m in self.members:
            out = ctx.outputs.get(m, {}).get(t)
            if out is None or m not in ctx.predictions or t not in ctx.predictions[m]:
                excluded[m] = "not computed"
            elif out.info.get("available") is False:
                excluded[m] = out.info.get("unavailable_reason", "inputs unavailable")
            else:
                available.append(m)
        # Relative skill on earlier folds: each member against the mean of the members scored on that fold.
        folds = sorted({s for m in available for s in _folds_before(ctx, m, t)})
        per_fold: dict[int, dict[str, float]] = {}
        for s in folds:
            vals = {m: ctx.fold_metric[m][s] for m in available if s in ctx.fold_metric.get(m, {})
                    and ctx.outputs.get(m, {}).get(s) is not None
                    and ctx.outputs[m][s].info.get("available") is not False}
            if len(vals) >= 2:
                per_fold[s] = vals
        # Spread of members around the fold mean (within-paper), not across papers: papers differ in
        # difficulty, which says nothing about which member is better.
        within = [v - float(np.mean(list(d.values()))) for d in per_fold.values() for v in d.values()]
        sd = max(float(np.std(within)) if len(within) > 1 else 0.0, self.min_sd)
        rows = {}
        for m in available:
            diffs = [d[m] - float(np.mean(list(d.values()))) for d in per_fold.values() if m in d]
            n = len(diffs)
            raw = float(np.mean(diffs)) if n else 0.0
            skill = raw / sd * (n / (n + self.n0))
            df = float(self.member_df.get(m, 1.0))
            # Components that bring outside knowledge (pretrained, syllabus, cross-course) start with one
            # pseudo-exam of evidence; components estimated only from this course start with none.
            pk = 1.0 if m in self.prior_knowledge else 0.0
            rho = 1.0 if df <= 0 else (exams + pk) / (exams + pk + df)
            rows[m] = {"reliability": rho, "skill": skill, "raw_gain": raw, "folds": n,
                       "base": rho * math.exp(self.tau * skill)}
        total = sum(r["base"] for r in rows.values())
        if total <= 0:  # no exams at all: only cross-course and syllabus knowledge carry weight
            for m, r in rows.items():
                r["base"] = 1.0 if m in self.prior_knowledge else 0.0
            total = sum(r["base"] for r in rows.values()) or 1.0
        for r in rows.values():
            r["weight"] = r["base"] / total
        return {"rows": rows, "excluded": excluded, "exams": exams, "folds": len(per_fold), "sd": sd}

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:
        K = ctx.panel.K
        info = self.weights(t, ctx)
        rows = info["rows"]
        members = [m for m in rows if rows[m]["weight"] > 0]
        if not members:
            return ModelOutput(np.zeros(K), {"available": False, "unavailable_reason": "no component available",
                                             **_public(info, self.displays)})
        post = ctx.store.at(t).extras.get("posterior")
        u = np.asarray(post.variance_ratio, dtype=float) if post is not None else np.ones(K)
        P = np.vstack([percentile_rank(ctx.predictions[m][t]) for m in members])  # (M, K)
        G = np.vstack([1.0 + self.gate_strength * u if m in self.prior_knowledge else np.ones(K) for m in members])
        W = np.array([rows[m]["weight"] for m in members])[:, None] * G
        W = W / W.sum(axis=0, keepdims=True)
        score = (W * P).sum(axis=0)
        contrib = (W * P).T  # (K, M): each component's share of the final score
        public = _public(info, self.displays)
        public["members"] = members
        public["gate_mean"] = {m: round(float(W[i].mean()), 4) for i, m in enumerate(members)}
        return ModelOutput(score, public, contrib, members)


def _public(info: dict[str, Any], displays: dict[str, str]) -> dict[str, Any]:
    return {
        "weights": {m: round(r["weight"], 4) for m, r in info["rows"].items()},
        "reliability": {m: round(r["reliability"], 4) for m, r in info["rows"].items()},
        "skill": {m: round(r["skill"], 4) for m, r in info["rows"].items()},
        "raw_gain": {m: round(r["raw_gain"], 4) for m, r in info["rows"].items()},
        "skill_folds": {m: r["folds"] for m, r in info["rows"].items()},
        "excluded": info["excluded"], "exams": info["exams"], "weight_folds": info["folds"],
        "displays": {m: displays.get(m, m) for m in list(info["rows"]) + list(info["excluded"])},
    }
