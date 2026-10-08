"""Question-format forecast per topic (spec section 17), chosen by backtest.

Three candidate methods, from simple to complex:

* global      the course-wide recency-weighted format mix
* topic       the topic's own recency-weighted mix, smoothed toward the global mix
* transition  the topic mix blended with a pooled transition matrix
              P(next format | format the last time this topic appeared)

For each held-out exam, a method is scored by whether its top format matches a format the
topic actually took. The simplest method within one standard error of the best is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.settings import Settings
from ..evaluation.metrics import mean_and_se
from ..temporal.panel import FORMATS, Panel

METHODS = ["global", "topic", "transition"]
COMPLEXITY = {"global": 0, "topic": 1, "transition": 2}


def _weights(T: int, half_life: float) -> np.ndarray:
    return 0.5 ** (np.arange(T - 1, -1, -1, dtype=float) / half_life)


def global_mix(F: np.ndarray, half_life: float) -> np.ndarray:
    """F: (T, K, nF) counts. Returns (nF,) distribution."""
    T = F.shape[0]
    if T == 0:
        return np.full(len(FORMATS), 1 / len(FORMATS))
    w = _weights(T, half_life)
    counts = np.tensordot(w, F.sum(axis=1), axes=(0, 0)) + 0.5
    return counts / counts.sum()


def topic_mix(F: np.ndarray, half_life: float, prior: np.ndarray, strength: float = 2.0) -> np.ndarray:
    T = F.shape[0]
    if T == 0:
        return np.tile(prior, (F.shape[1], 1))
    w = _weights(T, half_life)
    counts = np.tensordot(w, F, axes=(0, 0))  # (K, nF)
    mix = counts + strength * prior[None, :]
    return mix / mix.sum(axis=1, keepdims=True)


def transition_matrix(F: np.ndarray) -> np.ndarray:
    """Pooled P(format at next appearance | format at previous appearance) for the same topic."""
    nF = len(FORMATS)
    M = np.ones((nF, nF)) * 0.5
    T, K, _ = F.shape
    for k in range(K):
        idx = [t for t in range(T) if F[t, k].sum() > 0]
        for a, b in zip(idx, idx[1:]):
            pa = F[a, k] / F[a, k].sum()
            pb = F[b, k] / F[b, k].sum()
            M += np.outer(pa, pb)
    return M / M.sum(axis=1, keepdims=True)


def forecast(F: np.ndarray, method: str, half_life: float) -> np.ndarray:
    """(K, nF) format distribution for the next exam, using history F only."""
    g = global_mix(F, half_life)
    if method == "global":
        return np.tile(g, (F.shape[1], 1))
    tm = topic_mix(F, half_life, g)
    if method == "topic":
        return tm
    M = transition_matrix(F)
    out = tm.copy()
    for k in range(F.shape[1]):
        idx = [t for t in range(F.shape[0]) if F[t, k].sum() > 0]
        if idx:
            last = F[idx[-1], k] / F[idx[-1], k].sum()
            out[k] = 0.5 * tm[k] + 0.5 * (last @ M)
    return out


@dataclass
class TypeForecastReport:
    method: str
    reason: str
    accuracy: dict[str, dict[str, float]] = field(default_factory=dict)
    per_topic: dict[int, dict[str, Any]] = field(default_factory=dict)
    gated: dict[str, str] = field(default_factory=dict)


def run_type_forecast(panel: Panel, settings: Settings, targets: list[int],
                      fine_types: dict[int, dict[str, int]] | None = None) -> TypeForecastReport:
    half_life = float(settings.temporal.default_half_life)
    # All three methods always compete; with few papers the simplest one within one standard error wins.
    methods = list(METHODS)
    gated: dict[str, str] = {}
    folds: dict[str, dict[int, float]] = {m: {} for m in methods}
    for t in targets:
        hist = panel.formats[:t]
        actual = panel.formats[t]
        present = [k for k in range(panel.K) if actual[k].sum() > 0]
        if not present:
            continue
        for m in methods:
            pred = forecast(hist, m, half_life)
            hits = [1.0 if actual[k, int(np.argmax(pred[k]))] > 0 else 0.0 for k in present]
            folds[m][t] = float(np.mean(hits))
    accuracy = {}
    for m in methods:
        mean, se = mean_and_se(list(folds[m].values()))
        accuracy[m] = {"mean": None if math.isnan(mean) else round(mean, 4),
                       "se": None if math.isnan(se) else round(se, 4), "folds": len(folds[m])}
    scored = [m for m in methods if folds[m]]
    if scored:
        best = max(scored, key=lambda m: np.mean(list(folds[m].values())))
        chosen = best
        for m in sorted(scored, key=lambda x: COMPLEXITY[x]):
            diffs = [folds[best][t] - folds[m][t] for t in folds[best] if t in folds[m]]
            d, se = mean_and_se(diffs)
            if d <= (0.0 if math.isnan(se) else se) + 1e-12:
                chosen = m
                break
        reason = (f"'{chosen}' format forecast chosen: top-1 format accuracy "
                  f"{np.mean(list(folds[chosen].values())):.0%} on held-out exams "
                  f"(best '{best}' {np.mean(list(folds[best].values())):.0%}).")
    else:
        chosen = "topic"
        reason = "Not enough held-out exams to compare format forecasts; using each topic's own recent mix."
    pred = forecast(panel.formats, chosen, half_life)
    per_topic: dict[int, dict[str, Any]] = {}
    for k, item_id in enumerate(panel.item_ids):
        dist = {FORMATS[j]: round(float(pred[k, j]), 3) for j in np.argsort(-pred[k])[:4]}
        top = max(dist, key=dist.get)
        entry = {"format": top, "probability": dist[top], "distribution": dist, "method": chosen}
        if fine_types and item_id in fine_types:
            entry["common_types"] = sorted(fine_types[item_id], key=fine_types[item_id].get, reverse=True)[:3]
        per_topic[item_id] = entry
    return TypeForecastReport(chosen, reason, accuracy, per_topic, gated)
