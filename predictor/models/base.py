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

    def gate(self, panel: Panel, ctx: ModelContext) -> tuple[bool, str]:
        """Is the model statistically justified for this panel? Returns (enabled, reason)."""
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
