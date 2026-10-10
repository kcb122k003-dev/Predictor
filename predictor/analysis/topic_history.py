"""Per-topic examination history, computed once per analysis run.

Both the Syllabus Explorer and the Predictions tab read this artifact, so they show the same
counts as the model saw. It is built from exactly the question records that fill the topic
panel: a question counts for a topic when one of its *counted* mappings (the statuses that
pass the syllabus-strictness rule) rolls up to that topic. Three counts are kept apart:

* exam frequency: usable papers containing at least one counted question on the topic;
* question frequency: counted questions on the topic (several can sit in one paper);
* repetition: for each of those questions, whether it repeats an earlier paper's question
  near-verbatim, as a paraphrase, as the same concept, or not at all.

Missing years, excluded papers (duplicates, undated papers, papers you excluded) and papers or
questions with extraction problems are recorded, never treated as "topic not examined".
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from ..parsing.format_labels import FAMILY_ORDER, LABELS, display, family_display

VERSION = 1
RECENT_WINDOW = 3
LOW_PARSE_CONFIDENCE = 0.7


@dataclass
class PaperInfo:
    """A usable paper (included in the analysis), in panel order."""

    index: int
    exam_id: int
    label: str
    year: int | None = None
    session: str = ""
    file: str | None = None
    file_id: int | None = None
    provisional: list[str] = field(default_factory=list)  # reasons its counts may be unreliable


@dataclass
class ExcludedPaper:
    exam_id: int
    label: str
    year: int | None
    kind: str  # duplicate | undated | excluded
    reason: str
    duplicate_of: int | None = None


@dataclass
class QuestionInfo:
    id: int
    exam_index: int
    nodes: list[tuple[int, int, float]]  # counted mappings: (syllabus node, rank, weight)
    uncounted: list[tuple[int, str, float]] = field(default_factory=list)  # (node, status, score), not counted
    status: str = "A"
    manual: bool = False
    confidence: float = 1.0
    labels: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    format: str = "theory"
    marks: float | None = None
    relation: str = "new"  # exact | paraphrase | concept | new
    earlier: list[int] = field(default_factory=list)
    family_key: str | None = None
    parse_confidence: float = 1.0
    needs_review: bool = False


def relation_of(qid: int, exact_prev: dict[int, list[int]], para_prev: dict[int, list[int]],
                concept_prev: dict[int, list[int]]) -> tuple[str, list[int]]:
    """How a question relates to questions in earlier papers (same-paper pairs never count)."""
    for kind, links in (("exact", exact_prev), ("paraphrase", para_prev), ("concept", concept_prev)):
        if links.get(qid):
            return kind, sorted(links[qid])
    return "new", []


def missing_years(papers: list[PaperInfo]) -> list[int]:
    """Calendar years between the first and last usable paper with no usable paper.

    A missing year means no paper was supplied (or it was excluded); it is never evidence that a
    topic was not examined that year.
    """
    years = sorted({p.year for p in papers if p.year is not None})
    if len(years) < 2:
        return []
    have = set(years)
    return [y for y in range(years[0], years[-1] + 1) if y not in have]


def build_history(papers: list[PaperInfo], excluded: list[ExcludedPaper], questions: list[QuestionInfo],
                  topic_of: Callable[[int], int | None], topic_ids: list[int]) -> dict[str, Any]:
    """The run's history artifact: papers, questions and per-topic statistics."""
    data: dict[str, Any] = {
        "version": VERSION,
        "usable_papers": len(papers),
        "papers": [_paper_dict(p) for p in papers],
        "excluded_papers": [vars(e).copy() for e in excluded],
        "missing_years": missing_years(papers),
        "undated_papers": [p.index for p in papers if p.year is None],
        "questions": {str(q.id): _question_dict(q) for q in questions},
        "course_families": _course_families(questions),
    }
    topics = {}
    for tid in topic_ids:
        topics[str(tid)] = node_stats(data, lambda n, t=tid: topic_of(n) == t)
    data["topics"] = topics
    return data


def node_index(history: dict[str, Any]) -> dict[int, set[int]]:
    """Mapped node -> ids of questions mapped to it (counted or shown as uncertain), for fast subtree lookups."""
    index: dict[int, set[int]] = defaultdict(set)
    for key, q in history["questions"].items():
        for node, *_ in q["nodes"] + q.get("uncounted", []):
            index[int(node)].add(int(key))
    return index


def node_stats(history: dict[str, Any], member: Callable[[int], bool],
               candidates: set[int] | None = None) -> dict[str, Any]:
    """Statistics for one syllabus node: ``member(node_id)`` says whether a mapped node belongs to it.

    For a topic, members are the topic and its sub-topics; for a unit, every node below it. ``candidates`` limits
    the scan to questions that can match (from ``node_index``); it never changes the result.
    """
    papers = history["papers"]
    T = len(papers)
    provisional_papers = {p["index"]: p.get("provisional") or [] for p in papers}
    rows = []  # (question dict, role)
    uncounted = []
    items = (history["questions"].items() if candidates is None else
             ((str(c), history["questions"][str(c)]) for c in sorted(candidates) if str(c) in history["questions"]))
    for key, q in items:
        qid = int(key)
        nodes = q["nodes"]
        roles = [rank for node, rank, _ in nodes if member(node)]
        if roles:
            rows.append((qid, q, "primary" if min(roles) == min(r for _, r, _ in nodes) else "secondary"))
        elif any(member(node) for node, _, _ in q.get("uncounted", [])):
            uncounted.append(qid)

    per_paper: dict[int, list[int]] = defaultdict(list)
    roles_in_paper: dict[int, set[str]] = defaultdict(set)
    label_q, label_p = Counter(), defaultdict(set)
    fam_q, fam_p, fam_last = Counter(), defaultdict(set), {}
    repetition = Counter()
    marks = []
    reasons: list[str] = []
    multi = 0
    for qid, q, role in rows:
        e = q["e"]
        per_paper[e].append(qid)
        roles_in_paper[e].add(role)
        for label in q["labels"]:
            label_q[label] += 1
            label_p[label].add(e)
        for fam in q["families"]:
            fam_q[fam] += 1
            fam_p[fam].add(e)
            fam_last[fam] = max(fam_last.get(fam, -1), e)
        if len([x for x in q["labels"] if x != "unclassified"]) > 1:
            multi += 1
        repetition[q["rel"]] += 1
        if role == "primary" and q.get("marks"):
            marks.append(float(q["marks"]))
        if q.get("pc", 1.0) < LOW_PARSE_CONFIDENCE:
            reasons.append(f"question {qid}: low parse confidence ({q['pc']:.2f})")
        elif q.get("review"):
            reasons.append(f"question {qid}: flagged for review")
    appeared = sorted(per_paper)
    for e in appeared:
        for r in provisional_papers.get(e, []):
            reasons.append(f"{papers[e]['label']}: {r}")
    recent = list(range(max(0, T - RECENT_WINDOW), T))
    last = appeared[-1] if appeared else None
    return {
        "usable_papers": T,
        "papers": appeared,
        "exam_frequency": len(appeared),
        "secondary_only_papers": sum(1 for e in appeared if roles_in_paper[e] == {"secondary"}),
        "question_frequency": len(rows),
        "primary_questions": sum(1 for _, _, r in rows if r == "primary"),
        "secondary_questions": sum(1 for _, _, r in rows if r == "secondary"),
        "per_paper": {str(e): sorted(v) for e, v in sorted(per_paper.items())},
        "roles": {str(qid): role for qid, _, role in rows},
        "first_index": appeared[0] if appeared else None,
        "last_index": last,
        "last_label": papers[last]["label"] if last is not None else None,
        "last_year": papers[last]["year"] if last is not None else None,
        "recent_window": len(recent),
        "recent_hits": sum(1 for e in recent if e in per_paper),
        "marks": _marks_summary(marks),
        "labels": sorted(({"label": lb, "display": display(lb), "questions": label_q[lb], "papers": len(label_p[lb])}
                          for lb in label_q), key=lambda d: (-d["papers"], -d["questions"], list(LABELS).index(d["label"])
                                                             if d["label"] in LABELS else 99)),
        "families": sorted(({"family": f, "display": family_display(f), "questions": fam_q[f], "papers": len(fam_p[f]),
                             "last_index": fam_last[f]} for f in fam_q),
                           key=lambda d: (-d["papers"], -d["questions"], -d["last_index"], FAMILY_ORDER.index(d["family"])
                                          if d["family"] in FAMILY_ORDER else 99)),
        "multi_label_questions": multi,
        "repetition": {k: repetition.get(k, 0) for k in ("exact", "paraphrase", "concept", "new")},
        "uncounted_questions": sorted(uncounted),
        "provisional": bool(reasons),
        "provisional_reasons": sorted(set(reasons))[:12],
    }


def timeline(history: dict[str, Any], stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows in time order: usable papers (appeared or not), years with no usable paper and excluded papers.

    A year without a usable paper is shown as missing data, never as "not examined".
    """
    papers = history["papers"]
    qs = history["questions"]
    per_paper = {int(k): v for k, v in stats.get("per_paper", {}).items()}
    rows = []
    for p in papers:
        ids = per_paper.get(p["index"], [])
        labels = sorted({lb for q in ids for lb in qs[str(q)]["labels"]} - {"unclassified"},
                        key=lambda lb: list(LABELS).index(lb) if lb in LABELS else 99)
        rows.append({"kind": "paper", "index": p["index"], "exam_id": p["exam_id"], "label": p["label"],
                     "year": p["year"], "appeared": bool(ids), "questions": ids, "formats": [display(lb) for lb in labels],
                     "provisional": p.get("provisional") or [], "undated": p["year"] is None})
    excluded_by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in history.get("excluded_papers", []):
        if e.get("year") is not None:
            excluded_by_year[e["year"]].append(e)
    for y in history.get("missing_years", []):
        ex = excluded_by_year.get(y)
        if ex:
            for e in ex:
                rows.append({"kind": "excluded", "year": y, "label": e["label"], "exam_id": e["exam_id"],
                             "reason": e["reason"], "excluded_kind": e["kind"], "appeared": None})
        else:
            rows.append({"kind": "missing", "year": y, "label": str(y), "appeared": None,
                         "reason": "No usable paper for this year in the supplied data."})
    # Excluded papers in years that do have a usable paper (for example a duplicate) are listed too.
    usable_years = {p["year"] for p in papers}
    for e in history.get("excluded_papers", []):
        if e.get("year") is not None and e["year"] in usable_years:
            rows.append({"kind": "excluded", "year": e["year"], "label": e["label"], "exam_id": e["exam_id"],
                         "reason": e["reason"], "excluded_kind": e["kind"], "appeared": None})

    def key(r):
        if r["kind"] == "paper" and r.get("undated"):
            return (-1, r["index"], 0)
        year = r.get("year") or 0
        return (year, r.get("index", 10_000), 0 if r["kind"] == "paper" else 1)

    rows.sort(key=key)
    return rows


# ---------------------------------------------------------------------- helpers
def _paper_dict(p: PaperInfo) -> dict[str, Any]:
    return {"index": p.index, "exam_id": p.exam_id, "label": p.label, "year": p.year, "session": p.session,
            "file": p.file, "file_id": p.file_id, "provisional": list(p.provisional)}


def _question_dict(q: QuestionInfo) -> dict[str, Any]:
    return {"e": q.exam_index, "nodes": [[int(n), int(r), round(float(w), 3)] for n, r, w in q.nodes],
            "uncounted": [[int(n), s, round(float(sc), 3)] for n, s, sc in q.uncounted],
            "status": q.status, "manual": q.manual, "conf": round(float(q.confidence), 3), "labels": list(q.labels),
            "families": list(q.families), "fmt": q.format, "marks": q.marks, "rel": q.relation,
            "prev": list(q.earlier), "fam": q.family_key, "pc": round(float(q.parse_confidence), 3),
            "review": bool(q.needs_review)}


def _marks_summary(marks: list[float]) -> dict[str, Any]:
    if not marks:
        return {"known": 0, "min": None, "max": None, "median": None}
    return {"known": len(marks), "min": min(marks), "max": max(marks), "median": float(statistics.median(marks))}


def _course_families(questions: list[QuestionInfo]) -> list[dict[str, Any]]:
    """Format families across every counted question of the course (used only to infer a format for topics with
    no past question, and labelled as inferred)."""
    counted = [q for q in questions if q.nodes]
    fam_q, fam_p = Counter(), defaultdict(set)
    for q in counted:
        for f in q.families:
            fam_q[f] += 1
            fam_p[f].add(q.exam_index)
    return sorted(({"family": f, "display": family_display(f), "questions": fam_q[f], "papers": len(fam_p[f]),
                    "of_questions": len(counted)} for f in fam_q),
                  key=lambda d: (-d["questions"], FAMILY_ORDER.index(d["family"]) if d["family"] in FAMILY_ORDER else 99))
