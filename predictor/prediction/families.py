"""Exact-question layer: which past question families are likely to recur (secondary layer).

Score(family f of topic k) = P(topic k) x recurrence share of f within k, where the share is
the recency-weighted number of exams containing f divided by those containing k (smoothed).
Backtested as exact-question recall@K: the share of questions in a held-out exam that repeat
or paraphrase a family ranked in the top K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..evaluation.metrics import mean_and_se
from ..models.base import percentile_rank
from ..topic_modeling.families import RecurrenceResult, families_before


@dataclass
class FamilyRow:
    key: str
    topic_col: int | None
    question_ids: list[int]
    exam_indices: list[int]


def _family_rows(fams: dict[str, list[int]], q_exam: dict[int, int], q_topic: dict[int, int | None]) -> list[FamilyRow]:
    rows = []
    for key, qids in fams.items():
        topics = [q_topic.get(q) for q in qids if q_topic.get(q) is not None]
        topic = max(set(topics), key=topics.count) if topics else None
        rows.append(FamilyRow(key, topic, sorted(qids), sorted({q_exam[q] for q in qids})))
    return rows


def score_families(rows: list[FamilyRow], topic_scores: np.ndarray, T: int, half_life: float) -> np.ndarray:
    topic_p = percentile_rank(topic_scores) if len(topic_scores) else np.zeros(0)
    topic_exams: dict[int, set[int]] = {}
    for r in rows:
        if r.topic_col is not None:
            topic_exams.setdefault(r.topic_col, set()).update(r.exam_indices)
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        w = sum(0.5 ** ((T - 1 - e) / half_life) for e in r.exam_indices)
        tw = sum(0.5 ** ((T - 1 - e) / half_life) for e in topic_exams.get(r.topic_col, set())) if r.topic_col is not None else w
        share = (w + 0.25) / (tw + 1.0)
        tp = float(topic_p[r.topic_col]) if r.topic_col is not None and r.topic_col < len(topic_p) else 0.0
        out[i] = tp * share + 1e-6 * len(r.exam_indices)
    return out


def backtest_families(recurrence: RecurrenceResult, q_exam: dict[int, int], q_topic: dict[int, int | None],
                      topic_predictions: dict[int, np.ndarray], targets: list[int], k: int,
                      half_life: float) -> dict[str, Any]:
    folds_model, folds_recency = {}, {}
    for t in targets:
        fams = families_before(recurrence, q_exam, t)
        rows = _family_rows(fams, q_exam, q_topic)
        if not rows or t not in topic_predictions:
            continue
        target_qs = [q for q, e in q_exam.items() if e == t]
        if not target_qs:
            continue
        member_of = {q: i for i, r in enumerate(rows) for q in r.question_ids}
        hits_for = []
        for q in target_qs:
            linked = set(recurrence.exact_prev.get(q, [])) | set(recurrence.para_prev.get(q, []))
            hits_for.append({member_of[p] for p in linked if p in member_of})
        recurring = [h for h in hits_for if h]
        if not recurring:
            continue
        model = score_families(rows, topic_predictions[t], t, half_life)
        recency = np.array([sum(0.5 ** ((t - 1 - e) / half_life) for e in r.exam_indices) for r in rows])
        for scores, store in ((model, folds_model), (recency, folds_recency)):
            top = set(np.argsort(-scores, kind="stable")[:k].tolist())
            store[t] = float(np.mean([1.0 if h & top else 0.0 for h in recurring]))
    m_mean, m_se = mean_and_se(list(folds_model.values()))
    r_mean, r_se = mean_and_se(list(folds_recency.values()))
    return {
        "k": k, "folds": len(folds_model),
        "model": {"mean": _r(m_mean), "se": _r(m_se)},
        "recency_baseline": {"mean": _r(r_mean), "se": _r(r_se)},
        "note": ("Exact-question recall@K: of the questions in a held-out exam that repeated or paraphrased an "
                 "earlier question, the share whose family was in the top K predicted families."),
    }


def predict_families(recurrence: RecurrenceResult, q_exam: dict[int, int], q_topic: dict[int, int | None],
                     topic_scores: np.ndarray, T: int, half_life: float, limit: int = 30) -> list[dict[str, Any]]:
    rows = _family_rows(recurrence.families, q_exam, q_topic)
    rows = [r for r in rows if r.topic_col is not None]
    if not rows:
        return []
    scores = score_families(rows, topic_scores, T, half_life)
    order = np.argsort(-scores, kind="stable")[:limit]
    return [{"family": rows[i].key, "topic_col": rows[i].topic_col, "score": round(float(scores[i]), 4),
             "question_ids": rows[i].question_ids, "exam_indices": rows[i].exam_indices,
             "appearances": len(rows[i].exam_indices)} for i in order]


def _r(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)
