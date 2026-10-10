"""Detect when the latest analysis no longer matches the course data.

An analysis run stores a fingerprint of everything it read: which papers were included, the
question text and marks (and any types you set), your manual topic mappings, the current
syllabus and the settings. Outputs that the analysis itself writes (automatic mappings and
automatic question types) are left out, so re-running never makes a run look stale.

The Syllabus Explorer and the Predictions tab compare the stored fingerprint with the current
one and say what changed, instead of showing old counts and predictions as if they were current.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import select

from ..database.models import AnalysisRun, CourseTopic, Exam, ExamQuestion, QuestionTopicMapping
from .syllabus_store import current_version

NON_HISTORICAL_SOURCES = ("generated", "simulated")
PART_MESSAGES = {
    "papers": "papers were added, removed, included or excluded, or their year changed",
    "questions": "question text, marks, optional flags or types you set were edited",
    "mappings": "topic mappings were corrected",
    "syllabus": "the syllabus changed",
    "settings": "analysis settings changed",
}


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def data_fingerprint(session, course_id: int, settings) -> dict[str, str]:
    """Hashes of the inputs an analysis of this course would read, by part."""
    exams = session.execute(select(Exam).where(Exam.course_id == course_id, Exam.source.notin_(NON_HISTORICAL_SOURCES))
                            .order_by(Exam.id)).scalars().all()
    papers = [(e.id, bool(e.include_in_analysis), round(float(e.order_index or 0.0), 4), e.year, e.source,
               (e.structure or {}).get("label"), e.full_marks) for e in exams]
    # Questions of every paper (included or not): including or excluding a paper is a "papers" change only.
    all_ids = [e.id for e in exams]
    questions, mappings = [], []
    if all_ids:
        rows = session.execute(select(ExamQuestion).where(ExamQuestion.exam_id.in_(all_ids))
                               .order_by(ExamQuestion.exam_id, ExamQuestion.order_no)).scalars().all()
        for q in rows:
            questions.append((q.id, q.exam_id, q.parent_id, bool(q.is_leaf), q.text, q.marks, bool(q.is_optional),
                              q.or_group, list(q.question_types or []) if q.type_user_edited else None))
        ids = [q.id for q in rows]
        for start in range(0, len(ids), 500):
            for m in session.execute(select(QuestionTopicMapping).where(
                    QuestionTopicMapping.question_id.in_(ids[start:start + 500]),
                    QuestionTopicMapping.method == "manual")).scalars():
                mappings.append((m.question_id, m.topic_id, m.status, m.rank))
    version = current_version(session, course_id, create=False)
    topics = []
    if version is not None:
        for t in session.execute(select(CourseTopic).where(CourseTopic.course_id == course_id,
                                                           CourseTopic.syllabus_version_id == version.id)
                                 .order_by(CourseTopic.id)).scalars():
            topics.append((t.id, t.parent_id, t.number, t.title, list(t.concepts or []), bool(t.excluded),
                           list(t.kinds or []), t.hours, t.marks_weight, t.order_no))
    return {"papers": _h(papers), "questions": _h(questions), "mappings": _h(sorted(mappings, key=str)),
            "syllabus": _h(topics), "settings": _h(settings.as_dict())}


def combined(parts: dict[str, str]) -> str:
    return _h(parts)


def freshness(session, run: AnalysisRun | None, settings) -> dict[str, Any]:
    """Whether the run still matches the course data, and what changed if not."""
    if run is None:
        return {"run_id": None, "checked": False, "stale": False, "changed": [], "message": ""}
    stored = (run.config or {}).get("fingerprint_parts")
    if not stored:
        return {"run_id": run.id, "checked": False, "stale": False, "changed": [],
                "message": "This analysis was made before change tracking existed; re-run it to be sure the results "
                           "match your current data."}
    current = data_fingerprint(session, run.course_id, settings)
    changed = [k for k in PART_MESSAGES if stored.get(k) != current.get(k)]
    message = ""
    if changed:
        message = ("Since this analysis ran, " + "; ".join(PART_MESSAGES[k] for k in changed)
                   + ". The counts and predictions shown are from the earlier analysis; re-analyse to update them.")
    return {"run_id": run.id, "checked": True, "stale": bool(changed), "changed": changed, "message": message}
