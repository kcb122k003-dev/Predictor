"""Keyword (SQLite FTS5) and semantic search over questions and syllabus topics (spec section 52)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sqlalchemy import select, text

from ..database.models import CourseTopic, Exam, ExamQuestion
from ..embeddings.backends import TfidfBackend, get_pretrained
from ..embeddings.pretrained import HybridBackend
from ..services.context import AppContext
from ..services.syllabus_store import current_version, to_tree, topics_of


def _fts_query(q: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", q)
    return " ".join(f'"{t}"*' for t in tokens[:12])


class SearchService:
    def __init__(self, app: AppContext):
        self.app = app

    def search(self, course_id: int, query: str, *, year: int | None = None, unit_id: int | None = None,
               qtype: str | None = None, topic_id: int | None = None, limit: int = 50) -> dict[str, Any]:
        query = (query or "").strip()
        settings = self.app.settings
        q_scores: dict[int, dict[str, float]] = {}
        t_scores: dict[int, dict[str, float]] = {}
        if query and self.app.db.fts_available:
            fts = _fts_query(query)
            if fts:
                with self.app.db.session() as s:
                    rows = s.execute(text("SELECT kind, ref_id, bm25(search_index) AS r FROM search_index "
                                          "WHERE search_index MATCH :q AND course_id = :c ORDER BY r LIMIT 400"),
                                     {"q": fts, "c": course_id}).all()
                if rows:
                    worst = max(-r[2] for r in rows) or 1.0
                    for kind, ref, rank in rows:
                        target = q_scores if kind == "question" else t_scores
                        target.setdefault(int(ref), {})["keyword"] = round(-rank / worst, 4)
        with self.app.db.session() as s:
            version = current_version(s, course_id, create=False)
            topics = topics_of(s, course_id, version.id) if version else []
            tree = to_tree(topics) if topics else None
            leaves = s.execute(select(ExamQuestion, Exam).join(Exam).where(
                Exam.course_id == course_id, ExamQuestion.is_leaf.is_(True))).all()
            if query and tree is not None and len(tree):
                pre, _ = get_pretrained(settings)
                backend = HybridBackend(settings, pre) if pre is not None else TfidfBackend(settings)
                backend.fit([tree.document(i) for i in tree.nodes] + [query])
                qv = backend.encode([query])[0]
                if leaves:
                    qvecs = backend.encode([q.context_text or q.text or "" for q, _ in leaves])
                    sims = qvecs @ qv
                    for (qrow, _), sim in zip(leaves, sims):
                        if sim >= float(settings.search.min_semantic_score):
                            q_scores.setdefault(qrow.id, {})["semantic"] = round(float(sim), 4)
                ids = list(tree.nodes)
                tv = backend.encode([tree.document(i) for i in ids])
                for nid, sim in zip(ids, tv @ qv):
                    if sim >= float(settings.search.min_semantic_score):
                        t_scores.setdefault(nid, {})["semantic"] = round(float(sim), 4)
            questions = []
            by_id = {q.id: (q, e) for q, e in leaves}
            for qid, sc in q_scores.items():
                if qid not in by_id:
                    continue
                q, e = by_id[qid]
                primary = q.mappings[0] if q.mappings else None
                if year is not None and e.year != year:
                    continue
                if qtype and qtype not in (q.question_types or []):
                    continue
                if topic_id is not None and not any(m.topic_id == topic_id for m in q.mappings):
                    continue
                if unit_id is not None and tree is not None:
                    if not primary or primary.topic_id not in tree.nodes or tree.unit_of(primary.topic_id) != unit_id:
                        continue
                questions.append({
                    "id": q.id, "exam_id": e.id, "exam": (e.structure or {}).get("label") or str(e.year),
                    "year": e.year, "label": q.path_label, "text": q.text, "marks": q.marks,
                    "types": q.question_types, "score": max(sc.values()), "match": sc,
                    "topic_id": primary.topic_id if primary else None,
                    "topic": tree.path_label(primary.topic_id) if primary and tree and primary.topic_id in tree.nodes else None,
                    "status": primary.status if primary else None})
            questions.sort(key=lambda r: (-r["score"], -(r["year"] or 0)))
            topic_rows = []
            for tid, sc in t_scores.items():
                if tree is None or tid not in tree.nodes:
                    continue
                node = tree.nodes[tid]
                topic_rows.append({"id": tid, "path": tree.path_label(tid), "title": node.title,
                                   "concepts": node.concepts, "source_refs": node.source_refs[:3],
                                   "score": max(sc.values()), "match": sc})
            topic_rows.sort(key=lambda r: -r["score"])
        return {"query": query, "questions": questions[:limit], "topics": topic_rows[: max(10, limit // 3)]}
