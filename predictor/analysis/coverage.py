"""Topic and unit coverage analysis and co-occurrence testing (spec section 18)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import fisher_exact

from ..temporal.panel import Panel


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        prev = min(prev, pvals[idx] * n / rank)
        adjusted[idx] = prev
    return adjusted.tolist()


def unit_coverage(panel: Panel, unit_labels: list[str]) -> dict[str, Any]:
    """Marks share per unit per exam, unit representation and a balance index per exam."""
    if panel.item_unit is None or not panel.T:
        return {"available": False}
    U = len(unit_labels)
    marks = np.zeros((panel.T, U))
    present = np.zeros((panel.T, U))
    for k in range(panel.K):
        u = int(panel.item_unit[k])
        marks[:, u] += panel.marks[:, k]
        present[:, u] = np.maximum(present[:, u], panel.Y[:, k])
    totals = marks.sum(axis=1, keepdims=True)
    share = np.divide(marks, totals, out=np.zeros_like(marks), where=totals > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(share > 0, share * np.log(share), 0.0), axis=1)
    balance = ent / np.log(U) if U > 1 else np.ones(panel.T)
    return {
        "available": True,
        "units": unit_labels,
        "exams": [e.label for e in panel.exams],
        "marks_share": share.round(4).tolist(),
        "unit_frequency": present.mean(axis=0).round(3).tolist(),
        "balance_index": balance.round(3).tolist(),
        "mean_balance": round(float(balance.mean()), 3),
        "note": ("Balance index is the normalised entropy of marks across units: 1 = marks spread evenly over all "
                 "units, 0 = all marks on one unit."),
    }


def topic_coverage(panel: Panel) -> dict[str, Any]:
    if not panel.T:
        return {"available": False}
    per_exam = panel.Y.sum(axis=1)
    freq = panel.Y.mean(axis=0)
    return {
        "available": True,
        "topics_per_exam": per_exam.astype(int).tolist(),
        "mean_topics_per_exam": round(float(per_exam.mean()), 2),
        "share_of_syllabus_per_exam": (per_exam / max(panel.K, 1)).round(3).tolist(),
        "never_tested": [panel.item_labels[k] for k in range(panel.K) if freq[k] == 0],
        "rarely_tested": [panel.item_labels[k] for k in range(panel.K) if 0 < freq[k] <= 0.15],
        "always_tested": [panel.item_labels[k] for k in range(panel.K) if freq[k] >= 0.9],
    }


def cooccurrence(panel: Panel, min_support: int = 3, alpha: float = 0.10) -> dict[str, Any]:
    """Same-exam co-occurrence with Fisher's exact test and Benjamini-Hochberg correction."""
    T, K = panel.Y.shape
    if T < 4:
        return {"available": False, "reason": "Needs at least 4 exams."}
    Y = panel.Y > 0
    counts = Y.sum(axis=0)
    pairs = []
    for i in range(K):
        if counts[i] < min_support or counts[i] == T:
            continue
        for j in range(i + 1, K):
            if counts[j] < min_support or counts[j] == T:
                continue
            a = int((Y[:, i] & Y[:, j]).sum())
            b = int((Y[:, i] & ~Y[:, j]).sum())
            c = int((~Y[:, i] & Y[:, j]).sum())
            d = int((~Y[:, i] & ~Y[:, j]).sum())
            _, p = fisher_exact([[a, b], [c, d]])
            expected = counts[i] * counts[j] / T
            lift = a / expected if expected else 0.0
            pairs.append({"a": i, "b": j, "a_label": panel.item_labels[i], "b_label": panel.item_labels[j],
                          "together": a, "lift": round(float(lift), 3), "p_value": float(p)})
    adjusted = benjamini_hochberg([p["p_value"] for p in pairs])
    for p, q in zip(pairs, adjusted):
        p["q_value"] = round(float(q), 4)
        p["p_value"] = round(p["p_value"], 4)
        p["significant"] = q < alpha
    pairs.sort(key=lambda p: p["p_value"])
    return {
        "available": True, "tested_pairs": len(pairs),
        "significant": [p for p in pairs if p["significant"]],
        "top_positive": [p for p in pairs if p["lift"] > 1][:15],
        "top_negative": [p for p in pairs if p["lift"] < 1][:15],
        "note": (f"Pairs tested with Fisher's exact test; q-values control the false discovery rate. With {T} exams "
                 f"only strong associations can reach significance."),
    }
