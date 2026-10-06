"""Ranking and probability metrics (spec section 24).

All ranking metrics take a score vector over items and the 0/1 vector of items that
appeared in the target exam. ``expected_random`` gives the exact expectation of each
metric under a uniformly random ranking, which is the "random selection" baseline.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np

RANK_KS = (1, 3, 5, 10)


def _order(scores: np.ndarray, tie_break: np.ndarray | None = None) -> np.ndarray:
    if tie_break is None:
        tie_break = np.zeros_like(scores)
    # Stable: ties keep item order, so equal scores do not get lucky orderings.
    return np.lexsort((np.arange(len(scores)), -tie_break, -scores))


def precision_at_k(order: np.ndarray, rel: np.ndarray, k: int) -> float:
    k = min(k, len(order))
    return float(rel[order[:k]].sum() / k) if k else 0.0


def recall_at_k(order: np.ndarray, rel: np.ndarray, k: int) -> float:
    total = rel.sum()
    return float(rel[order[:k]].sum() / total) if total else 0.0


def hit_at_k(order: np.ndarray, rel: np.ndarray, k: int) -> float:
    return float(rel[order[:k]].sum() > 0)


def ndcg_at_k(order: np.ndarray, gains: np.ndarray, k: int) -> float:
    k = min(k, len(order))
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains[order[:k]] * discounts).sum())
    ideal = np.sort(gains)[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def mrr(order: np.ndarray, rel: np.ndarray) -> float:
    hits = np.flatnonzero(rel[order])
    return 1.0 / (hits[0] + 1) if hits.size else 0.0


def average_precision(order: np.ndarray, rel: np.ndarray) -> float:
    total = rel.sum()
    if not total:
        return 0.0
    hits = rel[order]
    cum = np.cumsum(hits)
    return float((cum / np.arange(1, len(hits) + 1) * hits).sum() / total)


def ranking_metrics(scores: np.ndarray, rel: np.ndarray, k: int, gains: np.ndarray | None = None,
                    tie_break: np.ndarray | None = None) -> dict[str, float]:
    rel = (np.asarray(rel) > 0).astype(float)
    order = _order(np.asarray(scores, dtype=float), tie_break)
    out = {
        "precision": precision_at_k(order, rel, k),
        "recall": recall_at_k(order, rel, k),
        "hit_rate": hit_at_k(order, rel, k),
        "ndcg": ndcg_at_k(order, rel, k),
        "mrr": mrr(order, rel),
        "map": average_precision(order, rel),
    }
    for kk in RANK_KS:
        out[f"hit@{kk}"] = hit_at_k(order, rel, kk)
        out[f"recall@{kk}"] = recall_at_k(order, rel, kk)
    if gains is not None:
        out["ndcg_marks"] = ndcg_at_k(order, np.asarray(gains, dtype=float), k)
    out["n_relevant"] = float(rel.sum())
    return out


def expected_random(n_items: int, n_rel: int, k: int, gains: np.ndarray | None = None) -> dict[str, float]:
    """Exact expectations of each metric for a uniformly random ranking."""
    N, R = n_items, n_rel
    if N == 0:
        return {}

    def hit(kk: int) -> float:
        kk = min(kk, N)
        if R == 0:
            return 0.0
        return 1.0 - comb(N - R, kk) / comb(N, kk) if N - R >= kk else 1.0

    kk = min(k, N)
    discounts = 1.0 / np.log2(np.arange(2, kk + 2))
    idcg = float(discounts[: min(R, kk)].sum())
    ndcg = float((R / N) * discounts.sum() / idcg) if idcg > 0 else 0.0
    # E[1 / rank of first relevant item]: P(first relevant at position i) = C(N-i, R-1) / C(N, R)
    if R:
        e_mrr = sum((1.0 / i) * comb(N - i, R - 1) / comb(N, R) for i in range(1, N - R + 2))
    else:
        e_mrr = 0.0
    out = {
        "precision": R / N,
        "recall": (kk / N) if R else 0.0,
        "hit_rate": hit(k),
        "ndcg": ndcg,
        "mrr": e_mrr,
        "map": R / N,
    }
    for k2 in RANK_KS:
        out[f"hit@{k2}"] = hit(k2)
        out[f"recall@{k2}"] = (min(k2, N) / N) if R else 0.0
    if gains is not None:
        g = np.asarray(gains, dtype=float)
        ideal = np.sort(g)[::-1][:kk]
        idcg_g = float((ideal * discounts[: len(ideal)]).sum())
        out["ndcg_marks"] = float(g.mean() * discounts.sum() / idcg_g) if idcg_g > 0 else 0.0
    out["n_relevant"] = float(R)
    return out


def brier(prob: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(y)) ** 2)) if len(y) else math.nan


def log_loss(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else math.nan


def reliability(prob: np.ndarray, y: np.ndarray, bins: int = 5) -> tuple[list[dict], float]:
    """Reliability table and expected calibration error (equal-count bins)."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(prob) == 0:
        return [], math.nan
    order = np.argsort(prob)
    chunks = np.array_split(order, min(bins, len(prob)))
    table, ece = [], 0.0
    for chunk in chunks:
        if not len(chunk):
            continue
        mean_p, rate = float(prob[chunk].mean()), float(y[chunk].mean())
        table.append({"mean_predicted": round(mean_p, 4), "observed_rate": round(rate, 4), "count": int(len(chunk))})
        ece += len(chunk) / len(prob) * abs(mean_p - rate)
    return table, float(ece)


def mean_and_se(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), math.nan
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))
