"""Syllabus Explorer: the course contents as a tree, with each node's examination history.

Everything shown comes from two places only:

* the current syllabus tree and the stored papers (names, source passages, original question
  wording, files and pages), and
* the latest completed analysis run: its ``topic_history`` artifact (all counts) and its topic
  predictions (priority, score, probability, uncertainty, format guide). The Predictions tab
  reads the same run through ``prediction_dict``, so both views agree.

If the course data changed after that run, ``freshness`` says what changed; nothing is
recomputed behind the user's back.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from ..analysis.topic_history import node_index, node_stats, timeline
from ..database.models import (AnalysisArtifact, AnalysisRun, Course, CourseTopic, DocumentPage, Exam, ExamQuestion,
                               PredictedQuestion, Prediction, SourceFile)
from ..parsing.format_labels import display as label_display
from ..prediction.priority_reason import priority_word
from ..syllabus.tree import TopicTree
from .context import AppContext
from .fingerprint import freshness
from .results_service import prediction_dict
from .syllabus_store import current_version, to_tree, topics_of

LEVEL_WORD = re.compile(r"^\s*(unit|chapter|module|part|section|block|lesson|topic)\b", re.IGNORECASE)
PASSAGE_STOP = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+\S|(?:unit|chapter|module|part|section)\b|practical\b|"
                          r"(?:text|reference)\s*books?\b|marks?\s+distribution\b)", re.IGNORECASE)
RELATION_TEXT = {"exact": "Exact or near-exact repeat of an earlier question",
                 "paraphrase": "Paraphrase of an earlier question",
                 "concept": "Same underlying concept as an earlier question",
                 "new": "No close earlier question (different question on the topic, or its first appearance)"}


class ExplorerService:
    def __init__(self, app: AppContext):
        self.app = app

    # ------------------------------------------------------------------ courses
    def courses(self) -> list[dict[str, Any]]:
        with self.app.db.session() as s:
            out = []
            for c in s.execute(select(Course).order_by(Course.id)).scalars():
                version = current_version(s, c.id, create=False)
                topics = topics_of(s, c.id, version.id) if version else []
                run = _latest_done(s, c.id)
                exams = s.execute(select(Exam).where(Exam.course_id == c.id)).scalars().all()
                out.append({"id": c.id, "name": c.name, "code": c.code, "is_synthetic": bool(c.is_synthetic),
                            "nodes": len(topics), "papers_included": sum(1 for e in exams if e.include_in_analysis),
                            "papers": len(exams),
                            "run": {"id": run.id, "finished_at": run.finished_at.isoformat() if run.finished_at else None}
                            if run else None,
                            "freshness": freshness(s, run, self.app.course_settings(c)) if run else None})
            return out

    # ------------------------------------------------------------------ tree
    def overview(self, course_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            course = s.get(Course, course_id)
            if course is None:
                raise KeyError(course_id)
            version = current_version(s, course_id, create=False)
            rows = topics_of(s, course_id, version.id) if version else []
            tree = to_tree(rows)
            run = _latest_done(s, course_id)
            history = _artifact(s, run.id, "topic_history") if run else None
            preds = _predictions(s, run.id) if run else {}
            topic_set = set(tree.topic_ids())
            index = node_index(history) if history else {}
            nodes = []
            for t in rows:
                node = _node_info(t, tree, topic_set)
                if history is not None:
                    node["stats"] = _compact(self._stats(tree, history, index, t.id))
                p = preds.get(t.id)
                node["prediction"] = _compact_prediction(p) if p is not None else None
                nodes.append(node)
            return {
                "course": {"id": course.id, "name": course.name, "code": course.code},
                "run": _run_info(run, history),
                "freshness": freshness(s, run, self.app.course_settings(course)) if run else None,
                "papers": history["papers"] if history else [],
                "usable_papers": history["usable_papers"] if history else 0,
                "excluded_papers": history["excluded_papers"] if history else [],
                "missing_years": history["missing_years"] if history else [],
                "nodes": nodes,
                "roots": tree.roots(),
                "families": sorted({f["display"] for n in nodes for f in ((n.get("stats") or {}).get("families") or [])}),
            }

    def _stats(self, tree: TopicTree, history: dict[str, Any], index: dict[int, set[int]], node_id: int
               ) -> dict[str, Any]:
        topic = history.get("topics", {}).get(str(node_id))
        if topic is not None:
            return topic
        sub = set([node_id] + tree.descendants(node_id))
        candidates = set().union(*(index.get(n, set()) for n in sub)) if sub else set()
        return node_stats(history, lambda n: n in sub, candidates)

    # ------------------------------------------------------------------ one node
    def node(self, course_id: int, node_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            course = s.get(Course, course_id)
            topic = s.get(CourseTopic, node_id)
            if course is None or topic is None or topic.course_id != course_id:
                raise KeyError(node_id)
            version = current_version(s, course_id, create=False)
            tree = to_tree(topics_of(s, course_id, version.id if version else None))
            if node_id not in tree.nodes:
                raise KeyError(node_id)
            topic_set = set(tree.topic_ids())
            run = _latest_done(s, course_id)
            history = _artifact(s, run.id, "topic_history") if run else None
            info = _node_info(topic, tree, topic_set)
            info["path"] = [{"id": a, "number": tree.nodes[a].number, "title": tree.nodes[a].title}
                            for a in reversed(tree.ancestors(node_id))]
            info["source_refs"] = [_ref_dict(s, r) for r in (topic.source_refs or [])]
            out: dict[str, Any] = {"node": info, "run": _run_info(run, history),
                                   "freshness": freshness(s, run, self.app.course_settings(course)) if run else None,
                                   "calibrated": bool(((run.summary if run else None) or {}).get("calibrated"))}
            if history is None:
                out["history_available"] = False
                out["message"] = ("No completed analysis has topic history yet. Run Analyze & Predict to count this "
                                  "topic's appearances." if run else "This course has not been analysed yet.")
                return out
            index = node_index(history)
            stats = self._stats(tree, history, index, node_id)
            out["history_available"] = True
            out["stats"] = stats
            out["timeline"] = timeline(history, stats)
            out["papers"] = history["papers"]
            out["excluded_papers"] = history["excluded_papers"]
            out["questions"] = self._questions(s, tree, history, stats, node_id)
            out["uncounted"] = self._uncounted(s, tree, history, stats)
            out["children"] = []
            for c in tree.kids(node_id):
                st = self._stats(tree, history, index, c)
                out["children"].append({"id": c, "number": tree.nodes[c].number, "title": tree.nodes[c].title,
                                        "stats": _compact(st)})
            preds = _predictions(s, run.id)
            p = preds.get(node_id)
            out["prediction"] = prediction_dict(p) if p is not None else None
            if p is None:
                out["child_predictions"] = [
                    {"topic_id": tid, "label": preds[tid].label, "rank": preds[tid].rank,
                     "priority": priority_word(preds[tid].category or ""), "category": preds[tid].category}
                    for tid in [node_id] + tree.descendants(node_id) if tid in preds]
                out["child_predictions"].sort(key=lambda d: d["rank"])
                if node_id not in topic_set and info["level"] == "sub-topic":
                    parent_topic = tree.topic_of(node_id)
                    if parent_topic in preds:
                        out["parent_prediction"] = _compact_prediction(preds[parent_topic])
            out["formulations"] = [
                {"text": f.text, "format": f.question_type, "kind": (f.grounding or {}).get("kind"), "basis": f.basis,
                 "marks_low": f.marks_low, "marks_high": f.marks_high, "evidence_question_ids": f.evidence_question_ids,
                 "label": (f.grounding or {}).get("label"), "note": (f.grounding or {}).get("note")}
                for f in s.execute(select(PredictedQuestion).where(PredictedQuestion.run_id == run.id,
                                                                   PredictedQuestion.topic_id == node_id)
                                   .order_by(PredictedQuestion.rank)).scalars()]
            facts = (out["prediction"] or {}).get("facts") or {}
            out["related"] = [e for e in facts.get("semantic_evidence") or [] if not e.get("mapped_here")]
            return out

    def _questions(self, s, tree: TopicTree, history: dict[str, Any], stats: dict[str, Any], node_id: int
                   ) -> list[dict[str, Any]]:
        roles = stats.get("roles", {})
        papers = history["papers"]
        hq = history["questions"]
        ids = [int(k) for k in roles]
        rows = {q.id: q for q in s.execute(select(ExamQuestion).where(ExamQuestion.id.in_(ids))).scalars()} if ids else {}
        files = _files_for(s, {q.exam_id for q in rows.values()})
        member = set([node_id] + tree.descendants(node_id))
        out = []
        for key, role in roles.items():
            qid = int(key)
            h = hq[key]
            q = rows.get(qid)
            paper = papers[h["e"]]
            d: dict[str, Any] = {"id": qid, "paper_index": h["e"], "paper": paper["label"], "year": paper["year"],
                                 "role": role, "status": h["status"], "confidence": h["conf"], "manual": h["manual"],
                                 "labels": [label_display(lb) for lb in h["labels"] if lb != "unclassified"],
                                 "families": h["families"], "relation": h["rel"],
                                 "relation_text": RELATION_TEXT.get(h["rel"], ""),
                                 "earlier": [_earlier(hq, papers, rows, e) for e in h.get("prev", [])][:6],
                                 "provisional": _q_provisional(h, paper),
                                 "other_topics": [{"id": tree.topic_of(n) if n in tree.nodes else n,
                                                   "label": tree.nodes[n].label() if n in tree.nodes else str(n),
                                                   "role": "primary" if r == 1 else "secondary"}
                                                  for n, r, _ in h["nodes"] if n not in member]}
            if q is None:
                d.update({"missing": True, "text": None})
            else:
                f = files.get(q.exam_id) or {}
                d.update({"text": q.text, "raw_text": q.raw_text, "question": q.path_label, "label": q.label,
                          "marks": q.marks, "marks_source": q.marks_source, "page": q.page_no, "line": q.line_no,
                          "file": f.get("file"), "file_id": f.get("file_id"), "is_pdf": f.get("is_pdf", False),
                          "order_no": q.order_no, "is_optional": bool(q.is_optional or q.or_group)})
            out.append(d)
        out.sort(key=lambda d: (-d["paper_index"], d.get("order_no") or 0))
        return out

    def _uncounted(self, s, tree: TopicTree, history: dict[str, Any], stats: dict[str, Any]) -> list[dict[str, Any]]:
        ids = stats.get("uncounted_questions") or []
        if not ids:
            return []
        rows = {q.id: q for q in s.execute(select(ExamQuestion).where(ExamQuestion.id.in_(ids))).scalars()}
        out = []
        for qid in ids:
            h = history["questions"][str(qid)]
            q = rows.get(qid)
            out.append({"id": qid, "paper": history["papers"][h["e"]]["label"], "status": h["status"],
                        "text": q.text if q else None, "question": q.path_label if q else None,
                        "matches": [{"id": n, "label": tree.nodes[n].label() if n in tree.nodes else str(n),
                                     "status": st, "score": sc} for n, st, sc in h.get("uncounted", [])]})
        return out

    # ------------------------------------------------------------------ sources
    def topic_source(self, topic_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            t = s.get(CourseTopic, topic_id)
            if t is None:
                raise KeyError(topic_id)
            return {"id": t.id, "number": t.number, "title": t.title, "concepts": t.concepts or [],
                    "description": t.description or "", "refs": [_ref_dict(s, r, passage=True) for r in t.source_refs or []]}

    def question_source(self, question_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            q = s.get(ExamQuestion, question_id)
            if q is None:
                raise KeyError(question_id)
            exam = q.exam
            src = s.get(SourceFile, exam.source_file_id) if exam and exam.source_file_id else None
            page_text = None
            if src is not None and q.page_no:
                page = s.execute(select(DocumentPage).where(DocumentPage.file_id == src.id,
                                                            DocumentPage.page_no == q.page_no)).scalars().first()
                page_text = page.text if page else None
            return {"id": q.id, "question": q.path_label, "text": q.text, "raw_text": q.raw_text,
                    "context": q.context_text, "marks": q.marks, "exam": (exam.structure or {}).get("label") if exam else None,
                    "year": exam.year if exam else None, "file": src.filename if src else None,
                    "file_id": src.id if src else None, "is_pdf": _is_pdf(src), "page": q.page_no, "line": q.line_no,
                    "page_text": page_text, "flags": q.quality_flags or [], "parse_confidence": q.parse_confidence}

    def page_image(self, file_id: int, page_no: int, highlight: str = "") -> bytes:
        """The PDF page as PNG, with the passage highlighted when it can be found on the page."""
        try:
            import pymupdf as fitz  # PyMuPDF, a core dependency
        except ImportError:  # pragma: no cover - PyMuPDF before 1.24
            import fitz

        with self.app.db.session() as s:
            src = s.get(SourceFile, file_id)
            if src is None or not _is_pdf(src):
                raise KeyError(file_id)
            path = src.stored_path
        doc = fitz.open(path)
        try:
            if not 1 <= page_no <= doc.page_count:
                raise KeyError(page_no)
            page = doc[page_no - 1]
            needle = re.sub(r"\s+", " ", highlight or "").strip()
            for probe in (needle[:80], needle[:40], needle[:24]):
                if len(probe) < 6:
                    break
                rects = page.search_for(probe)
                if rects:
                    for r in rects[:4]:
                        page.add_highlight_annot(r)
                    break
            pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), annots=True)
            return pix.tobytes("png")
        finally:
            doc.close()


# ---------------------------------------------------------------------- helpers
def _latest_done(s, course_id: int) -> AnalysisRun | None:
    return s.execute(select(AnalysisRun).where(AnalysisRun.course_id == course_id, AnalysisRun.status == "done")
                     .order_by(AnalysisRun.id.desc())).scalars().first()


def _artifact(s, run_id: int, key: str) -> dict[str, Any] | None:
    row = s.execute(select(AnalysisArtifact).where(AnalysisArtifact.run_id == run_id,
                                                   AnalysisArtifact.key == key)).scalars().first()
    return row.data if row else None


def _predictions(s, run_id: int) -> dict[int, Prediction]:
    return {p.topic_id: p for p in s.execute(select(Prediction).where(Prediction.run_id == run_id,
                                                                    Prediction.layer == "topic")).scalars()
            if p.topic_id is not None}


def _run_info(run: AnalysisRun | None, history: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    summary = run.summary or {}
    return {"id": run.id, "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "calibrated": bool(summary.get("calibrated")), "calibration_reason": summary.get("calibration_reason"),
            "selected_display": summary.get("selected_display"), "has_history": history is not None,
            "usable_papers": history["usable_papers"] if history else summary.get("exams")}


def _node_info(t: CourseTopic, tree: TopicTree, topic_set: set[int]) -> dict[str, Any]:
    kids = tree.kids(t.id)
    refs = t.source_refs or []
    original = refs[0].get("text") if refs else None
    # "group": a heading with entries below it; whether the document calls it a unit or chapter is level_word.
    level = "topic" if t.id in topic_set else "group" if kids else "sub-topic"
    m = LEVEL_WORD.match(original or "")
    inferred, reason = _inferred(t, original)
    return {"id": t.id, "parent_id": t.parent_id, "depth": t.depth, "number": t.number, "title": t.title,
            "label": tree.nodes[t.id].label() if t.id in tree.nodes else t.title, "original": original,
            "level": level, "level_word": m.group(1).capitalize() if m else None, "children": kids,
            "concepts": list(t.concepts or []), "kinds": list(t.kinds or []), "excluded": bool(t.excluded),
            "is_lab": "lab" in (t.kinds or []), "inferred": inferred, "inferred_reason": reason,
            "user_edited": bool(t.user_edited), "has_source": bool(refs), "hours": t.hours,
            "description": t.description or ""}


def _inferred(t: CourseTopic, original: str | None) -> tuple[bool, str]:
    """Whether the node's name was supplied by the app rather than read from the document."""
    if not t.source_refs:
        return (True, "Added by you; no source passage.") if t.user_edited else (True, "No source passage recorded.")
    words = set(re.findall(r"[a-z]{3,}", (t.title or "").lower()))
    src = set(re.findall(r"[a-z]{3,}", (original or "").lower()))
    missing = words - src
    if words and missing and len(missing) >= max(1, len(words) // 2):
        return True, (f"Name supplied by the app; the document heading reads \"{(original or '').strip()}\".")
    return False, ""


def _compact(st: dict[str, Any]) -> dict[str, Any]:
    keys = ("usable_papers", "exam_frequency", "question_frequency", "primary_questions", "secondary_questions",
            "secondary_only_papers", "last_index", "last_label", "last_year", "recent_window", "recent_hits",
            "provisional", "marks", "repetition")
    out = {k: st.get(k) for k in keys}
    out["families"] = [{"family": f["family"], "display": f["display"], "papers": f["papers"],
                        "questions": f["questions"]} for f in (st.get("families") or [])]
    out["uncounted"] = len(st.get("uncounted_questions") or [])
    return out


def _compact_prediction(p: Prediction) -> dict[str, Any]:
    unc = p.uncertainty or {}
    feats = p.features or {}
    return {"rank": p.rank, "category": p.category, "priority": priority_word(p.category or ""),
            "probability": p.probability, "prob_low": p.prob_low, "prob_high": p.prob_high,
            "calibrated": bool(p.calibrated), "relative_score": feats.get("relative_score"),
            "confidence": p.confidence, "evidence_strength": p.evidence_strength, "uncertainty": unc.get("level"),
            "rank_low": unc.get("rank_low"), "rank_high": unc.get("rank_high"),
            "format": ((feats.get("format_guide") or {}).get("display"))}


def _files_for(s, exam_ids: set[int]) -> dict[int, dict[str, Any]]:
    out = {}
    for e in s.execute(select(Exam).where(Exam.id.in_(exam_ids))).scalars() if exam_ids else []:
        src = s.get(SourceFile, e.source_file_id) if e.source_file_id else None
        out[e.id] = {"file": src.filename if src else None, "file_id": src.id if src else None, "is_pdf": _is_pdf(src)}
    return out


def _is_pdf(src: SourceFile | None) -> bool:
    return bool(src is not None and ((src.mime or "").endswith("pdf") or (src.filename or "").lower().endswith(".pdf")))


def _earlier(hq: dict[str, Any], papers: list[dict[str, Any]], rows: dict[int, ExamQuestion], qid: int
             ) -> dict[str, Any]:
    h = hq.get(str(qid))
    return {"id": qid, "paper": papers[h["e"]]["label"] if h else None,
            "question": rows[qid].path_label if qid in rows else None}


def _q_provisional(h: dict[str, Any], paper: dict[str, Any]) -> list[str]:
    reasons = []
    if h.get("pc", 1.0) < 0.7:
        reasons.append(f"low parse confidence ({h['pc']:.2f})")
    if h.get("review"):
        reasons.append("flagged for review")
    if h.get("status") == "B" and not h.get("manual"):
        reasons.append("probable (not certain) syllabus match")
    reasons += paper.get("provisional") or []
    return reasons


def _ref_dict(s, r: dict[str, Any], passage: bool = True) -> dict[str, Any]:
    out = {"file": r.get("file"), "file_id": r.get("file_id"), "page": r.get("page"), "line": r.get("line"),
           "text": r.get("text")}
    src = s.get(SourceFile, r.get("file_id")) if r.get("file_id") else None
    out["is_pdf"] = _is_pdf(src)
    if passage and src is not None and r.get("page"):
        page = s.execute(select(DocumentPage).where(DocumentPage.file_id == src.id,
                                                    DocumentPage.page_no == r.get("page"))).scalars().first()
        if page is not None:
            out["passage"] = _passage(page.text or "", r.get("line"), r.get("text") or "")
    return out


def _passage(page_text: str, line: int | None, ref_text: str, max_lines: int = 10) -> str:
    """The syllabus passage for a heading: its line and the lines that follow, up to the next heading."""
    lines = page_text.split("\n")
    start = (line - 1) if line and 0 < line <= len(lines) else None
    probe = ref_text.strip()[:30]
    if start is None or (probe and probe not in lines[start]):
        start = next((i for i, x in enumerate(lines) if probe and probe in x), start)
    if start is None:
        return ref_text
    out = [lines[start]]
    for x in lines[start + 1: start + max_lines]:
        if not x.strip() or PASSAGE_STOP.match(x):
            break
        out.append(x)
    return "\n".join(out).strip()
