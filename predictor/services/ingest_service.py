"""File ingestion: store, extract, parse and persist exams and course contents.

Unchanged files are never processed twice: a file whose SHA-256 already exists in the
course is reported as a duplicate. A paper whose questions match an existing paper (for
example the same paper as PDF and DOCX) is kept but excluded from analysis so it cannot
inflate frequencies (spec section 38).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from ..database.models import (Course, CourseTopic, DocumentPage, Exam, ExamQuestion, ExamSection,
                               QuestionTopicMapping, SourceFile)
from ..ingestion.extract import UnsupportedFileError, extract_document, sha256_of, sniff_type
from ..ingestion.types import ExtractionResult, PageText
from ..parsing.exam_parser import ExamParser, ParsedExam, ParsedQuestion
from ..parsing.metadata import order_index_for
from ..parsing.question_types import QuestionTypeClassifier
from ..preprocessing.textnorm import extract_equations, normalize_for_matching
from ..syllabus.merge import merge_syllabi
from ..syllabus.parser import ParsedSyllabus, SyllabusParser
from ..utils.logging import get_logger, log_event
from .context import AppContext
from .syllabus_store import get_or_create_version, persist_tree, topics_of, tree_from_db_as_parsed

log = get_logger("ingestion")
SHORT_NOTES_RE = re.compile(r"short notes?", re.IGNORECASE)
BENIGN_FLAGS = {"mcq_options_collapsed", "marks_moved_to_parent"}


@dataclass
class AddResult:
    file_id: int
    duplicate: bool
    message: str = ""


class IngestService:
    def __init__(self, app: AppContext):
        self.app = app

    # ------------------------------------------------------------------ files
    def add_file(self, course_id: int, src: Path, filename: str, kind: str,
                 syllabus_version: str | None = None) -> AddResult:
        if kind not in ("exam", "syllabus"):
            raise ValueError("kind must be 'exam' or 'syllabus'")
        settings = self.app.settings
        ext = Path(filename).suffix.lower()
        if ext not in settings.ingestion.supported_extensions:
            raise UnsupportedFileError(f"'{filename}': unsupported file type {ext or '(none)'}. "
                                       f"Supported: {', '.join(settings.ingestion.supported_extensions)}")
        size_mb = src.stat().st_size / 1e6
        if size_mb > float(settings.ingestion.max_file_mb):
            raise ValueError(f"'{filename}' is {size_mb:.0f} MB; the limit is {settings.ingestion.max_file_mb} MB.")
        sniff_type(src)  # raises for unsupported content
        digest = sha256_of(src)
        with self.app.db.session() as s:
            existing = s.execute(select(SourceFile).where(SourceFile.course_id == course_id,
                                                          SourceFile.sha256 == digest)).scalars().first()
            if existing is not None:
                log_event(log, "duplicate_file", file=filename, existing=existing.filename)
                return AddResult(existing.id, True, f"'{filename}' is identical to '{existing.filename}' "
                                                    f"(already uploaded); it was not added again.")
            target_dir = self.app.course_dir(course_id) / "files"
            target_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)[-120:]
            target = target_dir / f"{digest[:12]}_{safe}"
            shutil.copyfile(src, target)
            version_id = None
            if kind == "syllabus":
                version_id = get_or_create_version(s, course_id, syllabus_version).id
            row = SourceFile(course_id=course_id, kind=kind, filename=Path(filename).name, sha256=digest,
                             size_bytes=src.stat().st_size, stored_path=str(target), status="pending",
                             syllabus_version_id=version_id)
            s.add(row)
            s.flush()
            log_event(log, "file_added", file=filename, kind=kind, file_id=row.id)
            return AddResult(row.id, False, "")

    def process_file(self, file_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            f = s.get(SourceFile, file_id)
            if f is None:
                raise ValueError("file not found")
            f.status = "processing"
        try:
            with self.app.db.session() as s:
                f = s.get(SourceFile, file_id)
                course = s.get(Course, f.course_id)
                settings = self.app.course_settings(course)
                path, kind, filename, course_id = Path(f.stored_path), f.kind, f.filename, f.course_id
            ocr = self.app.ocr if settings.ocr.enabled else None
            result = extract_document(path, settings, ocr)
            with self.app.db.session() as s:
                f = s.get(SourceFile, file_id)
                s.execute(delete(DocumentPage).where(DocumentPage.file_id == file_id))
                for p in result.pages:
                    s.add(DocumentPage(file_id=file_id, page_no=p.page_no, text=p.text, method=p.method,
                                       ocr_confidence=p.confidence, quality_flags=p.flags, details=p.details))
                f.page_count = len(result.pages)
                f.extraction_summary = result.summary()
            if kind == "exam":
                outcome = self._ingest_exam(course_id, file_id, filename, result)
            else:
                outcome = self._ingest_syllabus(course_id, file_id)
            with self.app.db.session() as s:
                f = s.get(SourceFile, file_id)
                f.status = "done"
                f.error = ""
                f.extraction_summary = {**(f.extraction_summary or {}), **outcome}
            self.reindex(course_id)
            log_event(log, "file_processed", file=filename, kind=kind, **{k: v for k, v in outcome.items()
                                                                          if isinstance(v, (int, str, float))})
            return outcome
        except Exception as exc:
            log.exception("file processing failed", extra={"event": "file_failed", "file_id": file_id})
            with self.app.db.session() as s:
                f = s.get(SourceFile, file_id)
                if f is not None:
                    f.status = "error"
                    f.error = str(exc)[:2000]
            raise

    # ------------------------------------------------------------------ exams
    def _ingest_exam(self, course_id: int, file_id: int, filename: str, result: ExtractionResult) -> dict[str, Any]:
        with self.app.db.session() as s:
            course = s.get(Course, course_id)
            settings = self.app.course_settings(course)
        parsed = ExamParser(settings).parse(result.pages, filename)
        classifier = QuestionTypeClassifier.load(self.app.data_dir)
        with self.app.db.session() as s:
            # Re-processing the same file replaces its exam (unless the user edited it).
            old = s.execute(select(Exam).where(Exam.source_file_id == file_id)).scalars().first()
            if old is not None:
                s.delete(old)
                s.flush()
            exam = self._exam_row(course_id, file_id, parsed)
            s.add(exam)
            s.flush()
            section_ids: list[int] = []
            for i, sec in enumerate(parsed.sections):
                row = ExamSection(exam_id=exam.id, label=sec.label, title=sec.title[:300],
                                  instructions="\n".join(sec.instructions), attempt_count=sec.attempt_count, order_no=i)
                s.add(row)
                s.flush()
                section_ids.append(row.id)
            counter = [0]
            for q in parsed.questions:
                self._save_question(s, exam.id, q, None, section_ids, classifier, counter)
            s.flush()
            exam.structure = self.structure_of(s, exam.id)
            dup = self._find_duplicate_paper(s, course_id, exam)
            n_q = counter[0]
            leaves = s.execute(select(ExamQuestion).where(ExamQuestion.exam_id == exam.id,
                                                          ExamQuestion.is_leaf.is_(True))).scalars().all()
            review = sum(1 for q in leaves if q.needs_review)
            exam_id = exam.id
        return {"exam_id": exam_id, "questions": n_q, "needs_review": review, "warnings": parsed.warnings,
                "duplicate_of": dup}

    def _exam_row(self, course_id: int, file_id: int, parsed: ParsedExam) -> Exam:
        md = parsed.metadata
        year = md.get("year")
        calendar = md.get("calendar") or "unknown"
        etype = md.get("exam_type") or ""
        frac = float(md.get("session_fraction") or 0.5)
        order = order_index_for(year, calendar, frac, etype)
        label_bits = [str(year) if year else "Year?", md.get("session") or "", etype.title() if etype else ""]
        with self.app.db.session() as s:
            course = s.get(Course, course_id)
            source = "demo" if course is not None and course.is_synthetic else "upload"
        return Exam(course_id=course_id, source_file_id=file_id, source=source, title=(md.get("title") or "")[:400],
                    subject=(md.get("subject") or "")[:400], year=year, calendar=calendar,
                    session=md.get("session") or "", exam_type=etype, exam_date=md.get("exam_date") or "",
                    order_index=order if year else 0.0, full_marks=md.get("full_marks"), pass_marks=md.get("pass_marks"),
                    duration=md.get("duration") or "", examiner=md.get("examiner") or "",
                    instructions=parsed.instructions, metadata_confidence=md.confidence_map(),
                    include_in_analysis=year is not None,
                    exclusion_reason="" if year is not None else "Exam year not detected; set it in Review to include this paper.",
                    structure={"label": " ".join(b for b in label_bits if b).strip(),
                               "warnings": parsed.warnings, "attempt_count": parsed.attempt_count})

    def _save_question(self, s: Session, exam_id: int, q: ParsedQuestion, parent_id: int | None,
                       section_ids: list[int], classifier: QuestionTypeClassifier, counter: list[int]) -> None:
        counter[0] += 1
        context = q.context_text() if q.parent is not None else q.text
        tr = classifier.classify(q.text, marks=q.marks, options=q.options, context=context if context != q.text else "")
        serious = [f for f in q.flags if f not in BENIGN_FLAGS and not f.startswith("attempt_any")]
        needs_review = bool(serious) or q.confidence < 0.7
        row = ExamQuestion(
            exam_id=exam_id, section_id=section_ids[q.section_index] if q.section_index is not None and
            q.section_index < len(section_ids) else None, parent_id=parent_id, label=q.label,
            path_label=q.path_label, depth=q.depth, text=q.text, raw_text="\n".join(q.lines),
            normalized_text=normalize_for_matching(q.text), context_text=context, marks=q.marks,
            marks_source=q.marks_source, or_group=q.or_group, is_optional=q.is_optional, is_leaf=q.is_leaf,
            order_no=counter[0], page_no=q.page_no, line_no=q.line_no, question_types=tr.types,
            type_scores={"scores": tr.scores, "format": tr.format, "confidence": tr.confidence, "evidence": tr.evidence},
            options=q.options, equations=extract_equations(q.text), quality_flags=q.flags,
            parse_confidence=round(q.confidence, 3), needs_review=needs_review)
        s.add(row)
        s.flush()
        for child in q.children:
            self._save_question(s, exam_id, child, row.id, section_ids, classifier, counter)

    @staticmethod
    def structure_of(s: Session, exam_id: int) -> dict[str, Any]:
        exam = s.get(Exam, exam_id)
        rows = s.execute(select(ExamQuestion).where(ExamQuestion.exam_id == exam_id)
                         .order_by(ExamQuestion.order_no)).scalars().all()
        mains = [r for r in rows if r.parent_id is None]
        children: dict[int, int] = {}
        for r in rows:
            if r.parent_id is not None:
                children[r.parent_id] = children.get(r.parent_id, 0) + 1
        base = dict(exam.structure or {})
        base.update({
            "main_questions": [{"label": m.label, "marks": m.marks, "parts": children.get(m.id, 0), "or_group": m.or_group,
                                "optional": m.is_optional, "short_notes": bool(SHORT_NOTES_RE.search(m.text or ""))}
                               for m in mains],
            "leaves": sum(1 for r in rows if r.is_leaf),
            "sections": len(exam.sections),
        })
        return base

    def _find_duplicate_paper(self, s: Session, course_id: int, exam: Exam) -> int | None:
        mine = [q.normalized_text for q in s.execute(select(ExamQuestion).where(
            ExamQuestion.exam_id == exam.id, ExamQuestion.is_leaf.is_(True))).scalars() if q.normalized_text]
        if len(mine) < 2:
            return None
        others = s.execute(select(Exam).where(Exam.course_id == course_id, Exam.id != exam.id)).scalars().all()
        for other in others:
            theirs = [q.normalized_text for q in s.execute(select(ExamQuestion).where(
                ExamQuestion.exam_id == other.id, ExamQuestion.is_leaf.is_(True))).scalars() if q.normalized_text]
            if not theirs:
                continue
            matched = sum(1 for a in mine if max(fuzz.token_set_ratio(a, b) for b in theirs) >= 92)
            share = matched / max(len(mine), len(theirs))
            if share >= 0.8:
                exam.include_in_analysis = False
                exam.duplicate_of_id = other.id
                exam.exclusion_reason = (f"Duplicate of an already uploaded paper ({other.structure.get('label') or other.id}): "
                                         f"{matched} of {len(mine)} questions match. Excluded so it does not count twice.")
                log_event(log, "duplicate_paper", exam_id=exam.id, duplicate_of=other.id, share=round(share, 2))
                return other.id
        return None

    # --------------------------------------------------------------- syllabus
    def _parse_syllabus_file(self, s: Session, f: SourceFile, settings) -> ParsedSyllabus:
        pages = [PageText(p.page_no, p.text, p.method, p.ocr_confidence, list(p.quality_flags or []),
                          dict(p.details or {})) for p in f.pages]
        return SyllabusParser(settings).parse(pages, f.filename, f.id)

    def _ingest_syllabus(self, course_id: int, file_id: int) -> dict[str, Any]:
        """Merge course-content documents into the canonical topic tree.

        While no topic has been edited by hand and no question is mapped manually, all documents
        of the syllabus version are re-merged from scratch (the most detailed document becomes the
        base). After manual work exists, new documents are merged into the existing tree instead,
        so nothing the user did is lost.
        """
        with self.app.db.session() as s:
            f = s.get(SourceFile, file_id)
            course = s.get(Course, course_id)
            settings = self.app.course_settings(course)
            threshold = float(settings.syllabus.merge_title_ratio)
            version_id = f.syllabus_version_id
            existing = topics_of(s, course_id, version_id)
            ids = [t.id for t in existing]
            manual = s.execute(select(QuestionTopicMapping.id).where(QuestionTopicMapping.method == "manual",
                                                                     QuestionTopicMapping.topic_id.in_(ids))).first() if ids else None
            locked = any(t.user_edited for t in existing) or manual is not None
            parsed_new = self._parse_syllabus_file(s, f, settings)
            if existing and locked:
                merged = _merge_into_existing(tree_from_db_as_parsed(existing), parsed_new, threshold)
            else:
                files = s.execute(select(SourceFile).where(SourceFile.course_id == course_id,
                                                           SourceFile.kind == "syllabus",
                                                           SourceFile.syllabus_version_id == version_id)).scalars().all()
                docs = [parsed_new if other.id == file_id else self._parse_syllabus_file(s, other, settings)
                        for other in files if other.id == file_id or other.status == "done"]
                merged = merge_syllabi(docs, threshold)
                s.execute(delete(CourseTopic).where(CourseTopic.course_id == course_id,
                                                    CourseTopic.syllabus_version_id == version_id))
                s.flush()
            added = persist_tree(s, course_id, version_id, merged)
            if parsed_new.course_code and not course.code:
                course.code = parsed_new.course_code
        return {"topics_added": added, "warnings": parsed_new.warnings + merged.warnings,
                "objectives": parsed_new.objectives[:10], "merge_mode": "into_existing" if existing and locked else "rebuilt"}

    def rebuild_syllabus(self, course_id: int) -> dict[str, Any]:
        """Re-parse every course-content file from scratch (topics without user edits are replaced)."""
        with self.app.db.session() as s:
            files = s.execute(select(SourceFile).where(SourceFile.course_id == course_id,
                                                       SourceFile.kind == "syllabus")).scalars().all()
            edited = s.execute(select(CourseTopic).where(CourseTopic.course_id == course_id,
                                                         CourseTopic.user_edited.is_(True))).scalars().first()
            if edited is not None:
                raise ValueError("Some topics were edited by hand. Rebuilding would discard those edits; delete "
                                 "the edited topics first or edit the tree directly.")
            s.execute(delete(CourseTopic).where(CourseTopic.course_id == course_id))
            ids = [f.id for f in files]
        out = {"files": len(ids), "topics_added": 0}
        for fid in ids:
            out["topics_added"] += self._ingest_syllabus(course_id, fid)["topics_added"]
        self.reindex(course_id)
        return out

    # ----------------------------------------------------------------- search
    def reindex(self, course_id: int) -> None:
        if not self.app.db.fts_available:
            return
        with self.app.db.session() as s:
            s.execute(text("DELETE FROM search_index WHERE course_id = :c"), {"c": course_id})
            rows = s.execute(select(ExamQuestion.id, ExamQuestion.context_text).join(Exam).where(
                Exam.course_id == course_id, ExamQuestion.is_leaf.is_(True))).all()
            for qid, ctx in rows:
                s.execute(text("INSERT INTO search_index(kind, ref_id, course_id, text) VALUES ('question', :r, :c, :t)"),
                          {"r": qid, "c": course_id, "t": ctx or ""})
            for t in s.execute(select(CourseTopic).where(CourseTopic.course_id == course_id)).scalars():
                body = " ".join([t.title, *(t.concepts or []), t.description or "", *(t.aliases or [])])
                s.execute(text("INSERT INTO search_index(kind, ref_id, course_id, text) VALUES ('topic', :r, :c, :t)"),
                          {"r": t.id, "c": course_id, "t": body})


def _merge_into_existing(base, new, threshold):
    """Merge ``new`` into the DB-backed ``base`` tree, keeping ``base`` as the canonical structure."""
    from ..syllabus.merge import _best_match, _merge_into  # local import keeps the helpers private

    def place(node, parent):
        scope = parent.children if parent is not None else base.roots
        target = _best_match(node, scope, threshold) or _best_match(node, base.all_nodes(), threshold + 4)
        if target is not None:
            _merge_into(target, node)
            for c in node.children:
                place(c, target)
            return
        kids = node.children
        node.children = []
        if parent is not None:
            parent.add_child(node)
        else:
            base.roots.append(node)
        for c in kids:
            place(c, node)

    for r in new.roots:
        place(r, None)
    return base
