"""Model interface shared by every candidate predictor.

A model scores every item for target exam ``t`` using only information available before
``t``: features from ``ctx.store.at(t)`` (computed on ``panel.until(t)``) and, for learned
models, training rows from cutoffs ``s < t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.settings import Settings
from ..features.builder import FeatureStore
from ..temporal.panel import Panel


@dataclass
class ModelContext:
    panel: Panel
    store: FeatureStore
    settings: Settings
    seed: int = 13
    # Filled by the backtest engine for meta models (tuned variants, ensemble):
    predictions: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    fold_metric: dict[str, dict[int, float]] = field(default_factory=dict)  # model -> target -> primary metric
    outputs: dict[str, dict[int, "ModelOutput"]] = field(default_factory=dict)
    # The general (cross-course) ranking model; None means the shipped simulated prior.
    general: Any = None
    cache: dict[str, Any] = field(default_factory=dict)  # per-panel caches (prequential scores, ...)

    def general_model(self):
        if self.general is None:
            from ..inference.generic import load_prior

            self.general = load_prior()
        return self.general

    def effective_exams(self, t: int) -> float:
        """Exams before ``t`` at the exam level; papers with no year count half (their order is uncertain)."""
        exams = self.panel.exams[:t]
        return float(len(exams) - 0.5 * sum(1 for e in exams if e.year is None))


@dataclass
class ModelOutput:
    scores: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    contributions: np.ndarray | None = None  # (K, F) additive contributions where available
    contribution_names: list[str] | None = None


class BaseModel:
    name = "base"
    display = "Base"
    family = "baseline"
    complexity = 0
    meta = False
    hidden = False
    description = ""
    # role: "baseline" (reference methods), "component" (members of the evidence-aware ensemble)
    # or "ensemble". df is the effective number of parameters estimated from the course itself; the
    # ensemble's reliability prior is exams / (exams + df). prior_knowledge marks components whose
    # information does not come from the course's own history (pretrained, syllabus, cross-course).
    role = "baseline"
    df = 1.0
    prior_knowledge = False
    scope = "course"  # "course" (estimated from this course) or "global" (cross-course knowledge)

    def gate(self, panel: Panel, ctx: ModelContext) -> tuple[bool, str]:
        """Can the model run at all (its inputs exist)? There is no minimum dataset size."""
        return True, ""

    def predict(self, t: int, ctx: ModelContext) -> ModelOutput:  # pragma: no cover - interface
        raise NotImplementedError


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Average-rank percentile in [0, 1] (ties share a rank)."""
    from scipy.stats import rankdata

    n = len(scores)
    if n <= 1:
        return np.full(n, 0.5)
    return (rankdata(scores, method="average") - 1) / (n - 1)
