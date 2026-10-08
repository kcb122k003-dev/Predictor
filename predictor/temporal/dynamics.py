"""Gap, hazard, transition and rotation statistics computed from a (truncated) panel.

All functions take the 0/1 appearance matrix ``Y`` (exams x items) of the history only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ewma(y: np.ndarray, half_life: float) -> np.ndarray:
    """Recency-weighted mean down the time axis. ``y`` is (T, K) or (T,)."""
    T = y.shape[0]
    if T == 0:
        return np.zeros(y.shape[1:]) if y.ndim > 1 else np.array(0.0)
    ages = np.arange(T - 1, -1, -1, dtype=float)
    w = 0.5 ** (ages / max(half_life, 1e-6))
    w = w / w.sum()
    return np.tensordot(w, y, axes=(0, 0))


def linear_decay(y: np.ndarray) -> np.ndarray:
    T = y.shape[0]
    if T == 0:
        return np.zeros(y.shape[1:])
    w = np.arange(1, T + 1, dtype=float)
    return np.tensordot(w / w.sum(), y, axes=(0, 0))


def since_last(Y: np.ndarray) -> np.ndarray:
    """Exams since the last appearance (1 = appeared in the latest exam); T+1 if never seen."""
    T, K = Y.shape
    out = np.full(K, float(T + 1))
    for k in range(K):
        idx = np.flatnonzero(Y[:, k])
        if idx.size:
            out[k] = T - idx[-1]
    return out


def streaks(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(consecutive appearances, consecutive absences) ending at the latest exam."""
    T, K = Y.shape
    present = np.zeros(K)
    absent = np.zeros(K)
    for k in range(K):
        col = Y[:, k]
        run = 0
        last = col[-1] if T else 0
        for v in col[::-1]:
            if v == last:
                run += 1
            else:
                break
        if T:
            if last:
                present[k] = run
            else:
                absent[k] = run
    return present, absent


def gap_stats(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean gap between appearances, its std, and the number of gaps observed."""
    T, K = Y.shape
    mean = np.full(K, np.nan)
    std = np.full(K, np.nan)
    n = np.zeros(K)
    for k in range(K):
        idx = np.flatnonzero(Y[:, k])
        if idx.size >= 2:
            gaps = np.diff(idx).astype(float)
            mean[k] = gaps.mean()
            std[k] = gaps.std(ddof=1) if gaps.size >= 2 else np.nan
            n[k] = gaps.size
    return mean, std, n


@dataclass
class HazardTable:
    """Pooled discrete-time hazard P(appear at gap g | not seen since), shrunk to the base rate."""

    by_gap: dict[int, float]
    counts: dict[int, tuple[float, float]]  # gap -> (appearances, opportunities)
    never_seen: float
    base_rate: float
    max_gap: int

    def at(self, gap: np.ndarray, seen: np.ndarray) -> np.ndarray:
        out = np.empty(gap.shape, dtype=float)
        for i, (g, s) in enumerate(zip(gap, seen)):
            if not s:
                out[i] = self.never_seen
            else:
                out[i] = self.by_gap.get(int(min(g, self.max_gap)), self.base_rate)
        return out


def pooled_hazard(Y: np.ndarray, prior_strength: float = 3.0, max_gap: int = 6) -> HazardTable:
    """Estimate the hazard from every (item, exam) opportunity in the history.

    For exam s and item k with a previous appearance at s', the gap is g = s - s'
    (1 = consecutive papers). Gaps above ``max_gap`` share one bucket.
    """
    T, K = Y.shape
    base = float(Y.mean()) if Y.size else 0.0
    hits: dict[int, float] = {}
    opps: dict[int, float] = {}
    never_hits = never_opps = 0.0
    for k in range(K):
        last = None
        for s in range(T):
            if last is None:
                if s > 0:
                    never_opps += 1
                    never_hits += Y[s, k]
            else:
                g = min(s - last, max_gap)
                opps[g] = opps.get(g, 0.0) + 1
                hits[g] = hits.get(g, 0.0) + Y[s, k]
            if Y[s, k]:
                last = s
    by_gap = {g: (hits.get(g, 0.0) + prior_strength * base) / (opps[g] + prior_strength) for g in opps if opps[g] + prior_strength}
    never = (never_hits + prior_strength * base) / (never_opps + prior_strength) if (never_opps + prior_strength) else base
    counts = {g: (hits.get(g, 0.0), opps[g]) for g in opps}
    return HazardTable(by_gap, counts, float(never), base, max_gap)


def transition_counts(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-item counts of 0->0, 0->1, 1->0, 1->1 transitions between consecutive exams."""
    if Y.shape[0] < 2:
        z = np.zeros(Y.shape[1])
        return z, z.copy(), z.copy(), z.copy()
    prev, nxt = Y[:-1], Y[1:]
    n00 = ((prev == 0) & (nxt == 0)).sum(axis=0).astype(float)
    n01 = ((prev == 0) & (nxt == 1)).sum(axis=0).astype(float)
    n10 = ((prev == 1) & (nxt == 0)).sum(axis=0).astype(float)
    n11 = ((prev == 1) & (nxt == 1)).sum(axis=0).astype(float)
    return n00, n01, n10, n11


def trend_slope(Y: np.ndarray, window: int = 6) -> np.ndarray:
    T, K = Y.shape
    if T < 3:
        return np.zeros(K)
    w = min(window, T)
    x = np.arange(w, dtype=float)
    x = x - x.mean()
    seg = Y[-w:]
    return (x @ (seg - seg.mean(axis=0))) / (x @ x)


def sequential_lift(Y: np.ndarray, prior_strength: float = 2.0) -> np.ndarray:
    """L[j, k] = smoothed P(k at s | j at s-1) / P(k), from consecutive exam pairs."""
    T, K = Y.shape
    if T < 2:  # no consecutive pair of exams yet
        return np.ones((K, K))
    prev, nxt = Y[:-1], Y[1:]
    base = (Y.mean(axis=0) + 1e-6)
    joint = prev.T @ nxt  # j present at s-1 and k present at s
    count_j = prev.sum(axis=0)[:, None]
    cond = (joint + prior_strength * base[None, :]) / (count_j + prior_strength)
    return cond / base[None, :]


def same_exam_lift(Y: np.ndarray, prior_strength: float = 2.0) -> np.ndarray:
    """L[j, k] = smoothed P(k | j in the same exam) / P(k)."""
    T, K = Y.shape
    if T < 2:
        return np.ones((K, K))
    base = Y.mean(axis=0) + 1e-6
    joint = Y.T @ Y
    count_j = Y.sum(axis=0)[:, None]
    cond = (joint + prior_strength * base[None, :]) / (count_j + prior_strength)
    lift = cond / base[None, :]
    np.fill_diagonal(lift, 1.0)
    return lift


@dataclass
class RotationTest:
    item: int
    gaps: list[int]
    cv: float
    p_value: float
    period: float
    significant: bool


def rotation_tests(Y: np.ndarray, *, min_gaps: int = 3, alpha: float = 0.10, permutations: int = 2000,
                   seed: int = 0) -> list[RotationTest]:
    """Test whether appearance gaps are more regular than chance (permutation test).

    Statistic: coefficient of variation (CV) of the gaps between appearances. Under the null,
    the same number of appearances is placed at random positions in the same number of exams.
    A small p-value means the observed gaps are unusually regular (a rotation).
    """
    rng = np.random.default_rng(seed)
    T, K = Y.shape
    out: list[RotationTest] = []
    for k in range(K):
        idx = np.flatnonzero(Y[:, k])
        if idx.size < min_gaps + 1 or idx.size >= T:
            continue
        gaps = np.diff(idx)
        cv = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
        if gaps.mean() <= 1.05:  # consecutive appearances are frequency, not rotation
            continue
        count = 0
        for _ in range(permutations):
            pos = np.sort(rng.choice(T, size=idx.size, replace=False))
            g = np.diff(pos)
            if g.std() / g.mean() <= cv + 1e-12:
                count += 1
        p = (count + 1) / (permutations + 1)
        out.append(RotationTest(k, gaps.tolist(), round(cv, 3), round(p, 4), round(float(gaps.mean()), 2),
                                p < alpha))
    return out
