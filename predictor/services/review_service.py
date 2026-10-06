"""Manual review and override (spec section 40): exams, questions, topics and mappings.

Every edit marks the record as user-edited so re-analysis never overwrites it.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select

from ..database.models import (CourseTopic, Exam, ExamQuestion, QuestionTopicMapping, SourceFile, SyllabusVersion)
from ..parsing.metadata import order_index_for
from ..parsing.question_types import QuestionTypeClassifier
from ..preprocessing.textnorm import extract_equations, normalize_for_matching
from .context import AppContext
from .ingest_service import IngestService
from .results_service import mapping_dict, question_dict
from .syllabus_store import current_version, to_tree, topics_of

EXAM_FIELDS = {"title", "year", "calendar", "session", "exam_type", "exam_date", "order_index", "full_marks",
               "pass_marks", "duration", "examiner", "include_in_analysis"}
SESSION_FRACTIONS = {"spring": 0.3, "summer": 0.55, "fall": 0.8, "autumn": 0.8, "winter": 0.05}


class ReviewService:
    def __init__(self, app: AppContext):
        self.app = app
        self.ingest = IngestService(app)

    # ------------------------------------------------------------------ exams
    def exams(self, course_id: int) -> list[dict[str, Any]]:
        with self.app.db.session() as s:
            rows = s.execute(select(Exam).where(Exam.course_id == course_id)
                             .order_by(Exam.order_index, Exam.id)).scalars().all()
            out = []
            for e in rows:
                n_leaf = s.execute(select(func.count()).select_from(ExamQuestion).where(
                    ExamQuestion.exam_id == e.id, ExamQuestion.is_leaf.is_(True))).scalar_one()
                n_review = s.execute(select(func.count()).select_from(ExamQuestion).where(
                    ExamQuestion.exam_id == e.id, ExamQuestion.needs_review.is_(True))).scalar_one()
                f = s.get(SourceFile, e.source_file_id) if e.source_file_id else None
                out.append({
                    "id": e.id, "label": (e.structure or {}).get("label"), "title": e.title, "subject": e.subject,
                    "year": e.year, "calendar": e.calendar, "session": e.session, "exam_type": e.exam_type,
                    "exam_date": e.exam_date, "order_index": e.order_index, "full_marks": e.full_marks,
                    "pass_marks": e.pass_marks, "duration": e.duration, "examiner": e.examiner,
                    "include_in_analysis": e.include_in_analysis, "exclusion_reason": e.exclusion_reason,
                    "duplicate_of_id": e.duplicate_of_id, "instructions": e.instructions,
                    "metadata_confidence": e.metadata_confidence, "user_edited_fields": e.user_edited_fields,
                    "questions": n_leaf, "needs_review": n_review, "file": f.filename if f else None,
                    "file_id": e.source_file_id, "warnings": (e.structure or {}).get("warnings", []),
                    "extraction": f.extraction_summary if f else {},
                })
            return out

    def update_exam(self, exam_id: int, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = set(changes) - EXAM_FIELDS
        if unknown:
            raise ValueError(f"Unknown exam fields: {', '.join(sorted(unknown))}")
        with self.app.db.session() as s:
            e = s.get(Exam, exam_id)
            if e is None:
                raise KeyError(exam_id)
            for k, v in changes.items():
                setattr(e, k, v)
            edited = set(e.user_edited_fields or []) | set(changes)
            e.user_edited_fields = sorted(edited)
            if ({"year", "session", "calendar", "exam_type"} & set(changes)) and "order_index" not in changes:
                frac = SESSION_FRACTIONS.get((e.session or "").lower(), 0.5)
                e.order_index = order_index_for(e.year, e.calendar or "AD", frac, e.exam_type or "")
            if "include_in_analysis" in changes and changes["include_in_analysis"]:
                e.exclusion_reason = ""
            if "year" in changes and e.year and not e.include_in_analysis and "Exam year not detected" in (e.exclusion_reason or ""):
                e.include_in_analysis = True
                e.exclusion_reason = ""
            label_bits = [str(e.year) if e.year else "Year?", e.session or "", (e.exam_type or "").title()]
            e.structure = {**(e.structure or {}), "label": " ".join(b for b in label_bits if b)}
        return {"ok": True}

    def delete_exam(self, exam_id: int) -> None:
        with self.app.db.session() as s:
            e = s.get(Exam, exam_id)
            if e is None:
                raise KeyError(exam_id)
            s.delete(e)

    # -------------------------------------------------------------- questions
    def question_tree(self, exam_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            e = s.get(Exam, exam_id)
            if e is None:
                raise KeyError(exam_id)
            rows = s.execute(select(ExamQuestion).where(ExamQuestion.exam_id == exam_id)
                             .order_by(ExamQuestion.order_no)).scalars().all()
            items = {q.id: {**question_dict(q, e), "children": []} for q in rows}
            roots = []
            for q in rows:
                (items[q.parent_id]["children"] if q.parent_id in items else roots).append(items[q.id])
            pages = []
            if e.source_file_id:
                f = s.get(SourceFile, e.source_file_id)
                if f:
                    pages = [{"page": p.page_no, "method": p.method, "confidence": p.ocr_confidence,
                              "flags": p.quality_flags, "text": p.text} for p in f.pages]
            return {"exam_id": exam_id, "questions": roots, "pages": pages,
                    "sections": [{"id": sec.id, "label": sec.label, "title": sec.title,
                                  "attempt_count": sec.attempt_count} for sec in e.sections]}

    def questions_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        with self.app.db.session() as s:
            rows = s.execute(select(ExamQuestion).where(ExamQuestion.id.in_(ids))).scalars().all()
            return [question_dict(q) for q in rows]

    def update_question(self, qid: int, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"text", "marks", "label", "question_types", "is_optional", "or_group", "needs_review"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown question fields: {', '.join(sorted(unknown))}")
        with self.app.db.session() as s:
            q = s.get(ExamQuestion, qid)
            if q is None:
                raise KeyError(qid)
            for k, v in changes.items():
                setattr(q, k, v)
            if "text" in changes:
                q.normalized_text = normalize_for_matching(q.text)
                q.equations = extract_equations(q.text)
                q.context_text = self._context(s, q)
                q.quality_flags = [f for f in (q.quality_flags or []) if f not in ("implausible_words", "garbled_symbols")]
            if "question_types" in changes:
                q.type_user_edited = True
            if "marks" in changes:
                q.marks_source = "manual"
            if "needs_review" not in changes:
                q.needs_review = False
            q.user_edited = True
            exam_id, course_id = q.exam_id, q.exam.course_id
            out = question_dict(q)
        self._refresh_structure(exam_id)
        self.ingest.reindex(course_id)
        return out

    def add_question(self, exam_id: int, parent_id: int | None, label: str, text: str,
                     marks: float | None = None) -> dict[str, Any]:
        with self.app.db.session() as s:
            e = s.get(Exam, exam_id)
            if e is None:
                raise KeyError(exam_id)
            parent = s.get(ExamQuestion, parent_id) if parent_id else None
            order = (s.execute(select(func.max(ExamQuestion.order_no)).where(ExamQuestion.exam_id == exam_id)).scalar() or 0) + 1
            if parent is not None:
                parent.is_leaf = False
            q = ExamQuestion(exam_id=exam_id, parent_id=parent_id, label=label,
                             path_label=(parent.path_label + f"({label})") if parent else label,
                             depth=(parent.depth + 1) if parent else 1, text=text, raw_text=text,
                             normalized_text=normalize_for_matching(text), marks=marks,
                             marks_source="manual" if marks is not None else "", is_leaf=True, order_no=order,
                             user_edited=True, equations=extract_equations(text))
            s.add(q)
            s.flush()
            q.context_text = self._context(s, q)
            tr = QuestionTypeClassifier.load(self.app.data_dir).classify(text, marks=marks)
            q.question_types = tr.types
            course_id = e.course_id
            out = question_dict(q, e)
        self._refresh_structure(exam_id)
        self.ingest.reindex(course_id)
        return out

    def delete_question(self, qid: int) -> None:
        with self.app.db.session() as s:
            q = s.get(ExamQuestion, qid)
            if q is None:
                raise KeyError(qid)
            exam_id, parent_id, course_id = q.exam_id, q.parent_id, q.exam.course_id
            s.delete(q)
            s.flush()
            if parent_id:
                parent = s.get(ExamQuestion, parent_id)
                remaining = s.execute(select(func.count()).select_from(ExamQuestion).where(
                    ExamQuestion.parent_id == parent_id)).scalar_one()
                if parent is not None and remaining == 0:
                    parent.is_leaf = True
        self._refresh_structure(exam_id)
        self.ingest.reindex(course_id)

    def split_question(self, qid: int, parts: list[str]) -> list[dict[str, Any]]:
        """Split one question into consecutive siblings (for questions the parser merged)."""
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) < 2:
            raise ValueError("Give at least two non-empty parts.")
        with self.app.db.session() as s:
            q = s.get(ExamQuestion, qid)
            if q is None:
                raise KeyError(qid)
            later = s.execute(select(ExamQuestion).where(ExamQuestion.exam_id == q.exam_id,
                                                         ExamQuestion.order_no > q.order_no)).scalars().all()
            for row in later:
                row.order_no += len(parts) - 1
            q.text = parts[0]
            q.normalized_text = normalize_for_matching(parts[0])
            q.user_edited = True
            q.needs_review = False
            new_ids = [q.id]
            for i, text in enumerate(parts[1:], start=1):
                label = f"{q.label}-{i + 1}"
                row = ExamQuestion(exam_id=q.exam_id, section_id=q.section_id, parent_id=q.parent_id, label=label,
                                   path_label=f"{q.path_label}-{i + 1}", depth=q.depth, text=text, raw_text=text,
                                   normalized_text=normalize_for_matching(text), is_leaf=True,
                                   order_no=q.order_no + i, page_no=q.page_no, user_edited=True,
                                   equations=extract_equations(text))
                s.add(row)
                s.flush()
                new_ids.append(row.id)
            for nid in new_ids:
                row = s.get(ExamQuestion, nid)
                row.context_text = self._context(s, row)
            exam_id, course_id = q.exam_id, q.exam.course_id
        self._refresh_structure(exam_id)
        self.ingest.reindex(course_id)
        return self.questions_by_ids(new_ids)

    def merge_with_next(self, qid: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            q = s.get(ExamQuestion, qid)
            if q is None:
                raise KeyError(qid)
            nxt = s.execute(select(ExamQuestion).where(ExamQuestion.exam_id == q.exam_id,
                                                       ExamQuestion.parent_id == q.parent_id,
                                                       ExamQuestion.order_no > q.order_no)
                            .order_by(ExamQuestion.order_no)).scalars().first()
            if nxt is None:
                raise ValueError("There is no following question at the same level to merge with.")
            if not nxt.is_leaf or not q.is_leaf:
                raise ValueError("Only questions without sub-parts can be merged.")
            q.text = f"{q.text} {nxt.text}".strip()
            q.normalized_text = normalize_for_matching(q.text)
            if q.marks is not None or nxt.marks is not None:
                q.marks = (q.marks or 0) + (nxt.marks or 0)
            q.user_edited = True
            s.delete(nxt)
            s.flush()
            q.context_text = self._context(s, q)
            exam_id, course_id = q.exam_id, q.exam.course_id
            out = question_dict(q)
        self._refresh_structure(exam_id)
        self.ingest.reindex(course_id)
        return out

    @staticmethod
    def _context(s, q: ExamQuestion) -> str:
        stems = []
        parent = s.get(ExamQuestion, q.parent_id) if q.parent_id else None
        while parent is not None:
            if parent.text:
                stems.append(" ".join(parent.text.split()[:60]))
            parent = s.get(ExamQuestion, parent.parent_id) if parent.parent_id else None
        return " ".join(list(reversed(stems)) + [q.text or ""]).strip()

    def _refresh_structure(self, exam_id: int) -> None:
        with self.app.db.session() as s:
            e = s.get(Exam, exam_id)
            if e is not None:
                e.structure = IngestService.structure_of(s, exam_id)

    # --------------------------------------------------------------- mappings
    def mappings(self, course_id: int, status: str | None = None, exam_id: int | None = None) -> list[dict[str, Any]]:
        with self.app.db.session() as s:
            q = select(ExamQuestion, Exam).join(Exam).where(Exam.course_id == course_id, ExamQuestion.is_leaf.is_(True))
            if exam_id:
                q = q.where(Exam.id == exam_id)
            rows = s.execute(q.order_by(Exam.order_index, ExamQuestion.order_no)).all()
            out = []
            for qrow, exam in rows:
                maps = [mapping_dict(m) for m in qrow.mappings]
                st = maps[0]["status"] if maps else "unmapped"
                if status and st != status:
                    continue
                out.append({"question": question_dict(qrow, exam), "status": st, "mappings": maps,
                            "included": exam.include_in_analysis})
            return out

    def set_mapping(self, qid: int, topic_ids: list[int], status: str = "A") -> dict[str, Any]:
        if status not in ("A", "B", "D"):
            raise ValueError("status must be A (in syllabus), B (probably in) or D (outside)")
        with self.app.db.session() as s:
            q = s.get(ExamQuestion, qid)
            if q is None:
                raise KeyError(qid)
            s.execute(delete(QuestionTopicMapping).where(QuestionTopicMapping.question_id == qid))
            if status == "D" and not topic_ids:
                s.add(QuestionTopicMapping(question_id=qid, topic_id=None, rank=1, confidence=1.0, status="D",
                                           method="manual", evidence={"reason": "Marked outside the syllabus by you."}))
            for rank, tid in enumerate(topic_ids, start=1):
                if s.get(CourseTopic, tid) is None:
                    raise ValueError(f"Topic {tid} does not exist.")
                s.add(QuestionTopicMapping(question_id=qid, topic_id=tid, rank=rank, confidence=1.0, status=status,
                                           method="manual", evidence={"reason": "Set by you."}))
            s.flush()
            s.refresh(q)
            return {"question_id": qid, "mappings": [mapping_dict(m) for m in q.mappings]}

    def clear_manual_mapping(self, qid: int) -> None:
        with self.app.db.session() as s:
            s.execute(delete(QuestionTopicMapping).where(QuestionTopicMapping.question_id == qid,
                                                         QuestionTopicMapping.method == "manual"))

    # ----------------------------------------------------------------- topics
    def syllabus(self, course_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            versions = s.execute(select(SyllabusVersion).where(SyllabusVersion.course_id == course_id)).scalars().all()
            out_versions = []
            for v in versions:
                topics = topics_of(s, course_id, v.id)
                tree = to_tree(topics)
                by_id = {t.id: t for t in topics}

                def node(tid: int) -> dict[str, Any]:
                    t = by_id[tid]
                    return {"id": t.id, "parent_id": t.parent_id, "depth": t.depth, "number": t.number,
                            "title": t.title, "description": t.description, "concepts": t.concepts,
                            "objectives": t.objectives, "hours": t.hours, "marks_weight": t.marks_weight,
                            "kinds": t.kinds, "aliases": t.aliases, "source_refs": t.source_refs,
                            "excluded": t.excluded, "user_edited": t.user_edited,
                            "level": "unit" if t.depth == 1 and tree.kids(tid) else (
                                "topic" if tid in set(tree.topic_ids()) else "concept"),
                            "children": [node(c) for c in tree.kids(tid)]}

                out_versions.append({"id": v.id, "label": v.label, "is_current": v.is_current,
                                     "effective_from_order": v.effective_from_order,
                                     "topic_level": tree.topic_level, "topics": [node(r) for r in tree.roots()],
                                     "counts": {"units": len(tree.unit_ids()), "topics": len(tree.topic_ids()),
                                                "concepts": len(tree.concept_ids())}})
            return {"versions": out_versions}

    def add_topic(self, course_id: int, title: str, parent_id: int | None = None, number: str = "",
                  concepts: list[str] | None = None, hours: float | None = None) -> dict[str, Any]:
        if not title.strip():
            raise ValueError("A topic needs a title.")
        with self.app.db.session() as s:
            version = current_version(s, course_id)
            parent = s.get(CourseTopic, parent_id) if parent_id else None
            order = (s.execute(select(func.max(CourseTopic.order_no)).where(CourseTopic.course_id == course_id)).scalar() or 0) + 1
            t = CourseTopic(course_id=course_id, syllabus_version_id=parent.syllabus_version_id if parent else version.id,
                            parent_id=parent_id, depth=(parent.depth + 1) if parent else 1, number=number,
                            title=title.strip(), concepts=concepts or [], hours=hours, order_no=order, user_edited=True,
                            source_refs=[{"file": "added by you"}])
            s.add(t)
            s.flush()
            tid = t.id
        self.ingest.reindex(course_id)
        return {"id": tid}

    def update_topic(self, topic_id: int, changes: dict[str, Any]) -> dict[str, Any]:
        allowed = {"title", "number", "description", "concepts", "aliases", "hours", "marks_weight", "excluded",
                   "parent_id", "order_no", "kinds"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown topic fields: {', '.join(sorted(unknown))}")
        with self.app.db.session() as s:
            t = s.get(CourseTopic, topic_id)
            if t is None:
                raise KeyError(topic_id)
            if "parent_id" in changes and changes["parent_id"]:
                parent = s.get(CourseTopic, changes["parent_id"])
                if parent is None or parent.id == topic_id:
                    raise ValueError("Invalid parent topic.")
                tree = to_tree(topics_of(s, t.course_id, t.syllabus_version_id))
                if parent.id in tree.descendants(topic_id):
                    raise ValueError("A topic cannot be moved under its own sub-topic.")
            for k, v in changes.items():
                setattr(t, k, v)
            if "parent_id" in changes:
                parent = s.get(CourseTopic, t.parent_id) if t.parent_id else None
                self._fix_depth(s, t, (parent.depth + 1) if parent else 1)
            t.user_edited = True
            course_id = t.course_id
        self.ingest.reindex(course_id)
        return {"ok": True}

    def _fix_depth(self, s, t: CourseTopic, depth: int) -> None:
        t.depth = depth
        for c in s.execute(select(CourseTopic).where(CourseTopic.parent_id == t.id)).scalars():
            self._fix_depth(s, c, depth + 1)

    def delete_topic(self, topic_id: int) -> None:
        with self.app.db.session() as s:
            t = s.get(CourseTopic, topic_id)
            if t is None:
                raise KeyError(topic_id)
            course_id = t.course_id
            s.delete(t)
        self.ingest.reindex(course_id)
