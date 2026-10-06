"""Read-side access to analysis runs: predictions, topic detail, artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select

from ..database.models import (AnalysisArtifact, AnalysisRun, BacktestFold, CourseTopic, Exam, ExamQuestion,
                               ModelResult, PredictedQuestion, Prediction, QuestionTopicMapping)
from ..generation.paper import simulate_papers
from .context import AppContext
from .syllabus_store import current_version, to_tree, topics_of


def run_dict(run: AnalysisRun) -> dict[str, Any]:
    return {"id": run.id, "course_id": run.course_id, "status": run.status, "progress": run.progress,
            "message": run.message, "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None, "summary": run.summary or {}}


def prediction_dict(p: Prediction) -> dict[str, Any]:
    return {"id": p.id, "layer": p.layer, "topic_id": p.topic_id, "key": p.item_key, "label": p.label,
            "rank": p.rank, "score": p.score, "probability": p.probability, "prob_low": p.prob_low,
            "prob_high": p.prob_high, "calibrated": p.calibrated, "category": p.category, "confidence": p.confidence,
            "facts": (p.features or {}).get("facts", p.features or {}), "signals_for": (p.features or {}).get("signals_for", []),
            "relative_score": (p.features or {}).get("relative_score"),
            "contributions": (p.contributions or {}).get("values", {}),
            "contribution_source": (p.contributions or {}).get("source", ""),
            "evidence": (p.evidence or {}).get("lines", []), "why_not": p.why_not or []}


def question_dict(q: ExamQuestion, exam: Exam | None = None) -> dict[str, Any]:
    exam = exam or q.exam
    return {"id": q.id, "exam_id": q.exam_id, "exam": (exam.structure or {}).get("label") if exam else None,
            "year": exam.year if exam else None, "parent_id": q.parent_id, "label": q.label,
            "path_label": q.path_label, "depth": q.depth, "text": q.text, "context": q.context_text,
            "marks": q.marks, "marks_source": q.marks_source, "or_group": q.or_group, "is_optional": q.is_optional,
            "is_leaf": q.is_leaf, "page": q.page_no, "line": q.line_no, "types": q.question_types or [],
            "type_scores": q.type_scores or {}, "type_user_edited": q.type_user_edited, "options": q.options or [],
            "equations": q.equations or [], "flags": q.quality_flags or [], "confidence": q.parse_confidence,
            "needs_review": q.needs_review, "user_edited": q.user_edited,
            "mappings": [mapping_dict(m) for m in q.mappings]}


def mapping_dict(m: QuestionTopicMapping) -> dict[str, Any]:
    return {"id": m.id, "topic_id": m.topic_id, "topic": m.topic.title if m.topic else None, "rank": m.rank,
            "status": m.status, "confidence": m.confidence, "semantic": m.semantic_similarity,
            "keyword": m.keyword_overlap, "unknown_ratio": m.unknown_term_ratio, "matched_terms": m.matched_terms,
            "evidence_text": m.evidence_text, "evidence": m.evidence or {}, "method": m.method}


class ResultsService:
    def __init__(self, app: AppContext):
        self.app = app

    def latest_run(self, course_id: int, done_only: bool = True) -> int | None:
        with self.app.db.session() as s:
            q = select(AnalysisRun).where(AnalysisRun.course_id == course_id)
            if done_only:
                q = q.where(AnalysisRun.status == "done")
            run = s.execute(q.order_by(AnalysisRun.id.desc())).scalars().first()
            return run.id if run else None

    def run(self, run_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            run = s.get(AnalysisRun, run_id)
            if run is None:
                raise KeyError(run_id)
            return run_dict(run)

    def runs(self, course_id: int) -> list[dict[str, Any]]:
        with self.app.db.session() as s:
            rows = s.execute(select(AnalysisRun).where(AnalysisRun.course_id == course_id)
                             .order_by(AnalysisRun.id.desc()).limit(30)).scalars().all()
            out = []
            for r in rows:
                d = run_dict(r)
                d["summary"] = {k: d["summary"].get(k) for k in ("exams", "selected_display", "seconds", "error")}
                out.append(d)
            return out

    def predictions(self, run_id: int, layer: str = "topic") -> list[dict[str, Any]]:
        with self.app.db.session() as s:
            rows = s.execute(select(Prediction).where(Prediction.run_id == run_id, Prediction.layer == layer)
                             .order_by(Prediction.rank)).scalars().all()
            return [prediction_dict(p) for p in rows]

    def artifact(self, run_id: int, key: str) -> dict[str, Any]:
        with self.app.db.session() as s:
            row = s.execute(select(AnalysisArtifact).where(AnalysisArtifact.run_id == run_id,
                                                           AnalysisArtifact.key == key)).scalars().first()
            if row is None:
                raise KeyError(key)
            return row.data

    def models(self, run_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            models = s.execute(select(ModelResult).where(ModelResult.run_id == run_id)).scalars().all()
            folds = s.execute(select(BacktestFold).where(BacktestFold.run_id == run_id)
                              .order_by(BacktestFold.target_index)).scalars().all()
            return {
                "models": [{"layer": m.layer, "name": m.model_name, "display": m.display_name, "family": m.family,
                            "complexity": m.complexity, "enabled": m.enabled, "gate_reason": m.gate_reason,
                            "selected": m.selected, "metrics": m.metrics, "se": m.metric_se, "notes": m.notes}
                           for m in models],
                "folds": [{"model": f.model_name, "target": f.target_label, "target_index": f.target_index,
                           "train_exams": f.n_train_exams, "metrics": f.metrics} for f in folds],
            }

    def topic_detail(self, run_id: int, topic_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            pred = s.execute(select(Prediction).where(Prediction.run_id == run_id, Prediction.layer == "topic",
                                                      Prediction.topic_id == topic_id)).scalars().first()
            topic = s.get(CourseTopic, topic_id)
            if topic is None:
                raise KeyError(topic_id)
            version = current_version(s, topic.course_id, create=False)
            tree = to_tree(topics_of(s, topic.course_id, version.id if version else None))
            node_ids = [topic_id] + tree.descendants(topic_id)
            mappings = s.execute(select(QuestionTopicMapping).where(QuestionTopicMapping.topic_id.in_(node_ids))).scalars().all()
            questions = []
            for m in mappings:
                q = m.question
                if not q.exam.include_in_analysis:
                    continue
                d = question_dict(q)
                d["mapping"] = mapping_dict(m)
                d["order"] = q.exam.order_index
                questions.append(d)
            questions.sort(key=lambda d: (-(d["order"] or 0), d["path_label"]))
            forms = s.execute(select(PredictedQuestion).where(PredictedQuestion.run_id == run_id,
                                                              PredictedQuestion.topic_id == topic_id)
                              .order_by(PredictedQuestion.rank)).scalars().all()
            families = s.execute(select(Prediction).where(Prediction.run_id == run_id, Prediction.layer == "family",
                                                          Prediction.topic_id == topic_id)
                                 .order_by(Prediction.rank)).scalars().all()
            return {
                "topic": {"id": topic.id, "title": topic.title, "number": topic.number, "path": tree.path_label(topic_id),
                          "concepts": topic.concepts, "hours": topic.hours, "kinds": topic.kinds,
                          "source_refs": topic.source_refs, "children": [tree.nodes[c].label() for c in tree.kids(topic_id)]},
                "prediction": prediction_dict(pred) if pred else None,
                "questions": questions,
                "formulations": [{"text": f.text, "format": f.question_type, "marks_low": f.marks_low,
                                  "marks_high": f.marks_high, "basis": f.basis, "evidence_question_ids": f.evidence_question_ids,
                                  "grounding": f.grounding, "label": (f.grounding or {}).get("label")} for f in forms],
                "families": [{"rank": f.rank, "score": f.score, **(f.features or {})} for f in families],
            }

    def regenerate_papers(self, run_id: int, seed: int, variants: int = 3) -> list[dict[str, Any]]:
        with self.app.db.session() as s:
            run = s.get(AnalysisRun, run_id)
            if run is None:
                raise KeyError(run_id)
            course_id = run.course_id
            preds = s.execute(select(Prediction).where(Prediction.run_id == run_id, Prediction.layer == "topic")
                              .order_by(Prediction.rank)).scalars().all()
            forms = s.execute(select(PredictedQuestion).where(PredictedQuestion.run_id == run_id)
                              .order_by(PredictedQuestion.rank)).scalars().all()
            structure = s.execute(select(AnalysisArtifact).where(AnalysisArtifact.run_id == run_id,
                                                                 AnalysisArtifact.key == "structure")).scalars().first()
            coverage = s.execute(select(AnalysisArtifact).where(AnalysisArtifact.run_id == run_id,
                                                                AnalysisArtifact.key == "coverage")).scalars().first()
            version = current_version(s, course_id, create=False)
            tree = to_tree(topics_of(s, course_id, version.id if version else None))
            preds = [p for p in preds if p.topic_id in tree.nodes]
            unit_ids = tree.unit_ids()
            unit_col = {u: i for i, u in enumerate(unit_ids)}
            topic_ids = [p.topic_id for p in preds]
            labels = [p.label for p in preds]
            scores = np.array([p.probability if p.probability is not None else
                               max((p.features or {}).get("relative_score") or 0.01, 0.01) ** 2 for p in preds])
            units = np.array([unit_col.get(tree.unit_of(t), 0) for t in topic_ids])
            formulations: dict[int, list[dict[str, Any]]] = {}
            for f in forms:
                formulations.setdefault(f.topic_id, []).append(
                    {"text": f.text, "format": f.question_type, "marks_low": f.marks_low, "marks_high": f.marks_high,
                     "basis": f.basis})
            typical = (structure.data or {}).get("typical", {}) if structure else {}
            share = ((coverage.data or {}).get("units") or {}).get("marks_share") if coverage else None
            max_share = float(np.max(share)) if share else 0.35
            type_fc = {p.topic_id: ((p.features or {}).get("facts") or {}).get("type_forecast") or {} for p in preds}
        return simulate_papers(topic_ids=topic_ids, topic_labels=labels, scores=scores, units=units,
                               unit_labels=[tree.nodes[u].label() for u in unit_ids], typical=typical,
                               formulations=formulations, type_forecast=type_fc, max_unit_share=max_share,
                               variants=variants, seed=seed)
