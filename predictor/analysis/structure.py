"""Examination template discovery (spec section 19).

Each structural property is summarised across exams. A property is a *high-confidence
historical pattern* only when the same value held in at least 80% of the recent exams
(and at least three exams exist). Anything said about the future paper is labelled a
*speculative hypothesis*.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MainQuestion:
    label: str
    marks: float | None
    n_parts: int
    or_group: str | None
    is_optional: bool
    section: int | None
    is_short_notes: bool = False


@dataclass
class LeafInfo:
    format: str
    marks: float | None
    topic_col: int | None
    unit_col: int | None


@dataclass
class ExamSummary:
    label: str
    order: float
    full_marks: float | None
    mains: list[MainQuestion]
    leaves: list[LeafInfo]
    n_sections: int = 0
    attempt_count: int | None = None
    duration: str = ""

    def slots(self) -> list[list[MainQuestion]]:
        """Main questions grouped so OR alternatives count once."""
        groups: dict[str, list[MainQuestion]] = {}
        out: list[list[MainQuestion]] = []
        for m in self.mains:
            if m.or_group:
                if m.or_group not in groups:
                    groups[m.or_group] = []
                    out.append(groups[m.or_group])
                groups[m.or_group].append(m)
            else:
                out.append([m])
        return out

    def stats(self) -> dict[str, Any]:
        slots = self.slots()
        fmts = Counter(l.format for l in self.leaves)
        marks_main = [max((m.marks or 0) for m in s) for s in slots]
        total = self.full_marks or (sum(marks_main) if any(marks_main) else None)
        return {
            "main_questions": len(slots),
            "sections": self.n_sections,
            "or_alternatives": sum(1 for s in slots if len(s) > 1),
            "short_notes_question": any(m.is_short_notes for m in self.mains),
            "attempt_choice": self.attempt_count is not None or any(m.is_optional and not m.or_group for m in self.mains),
            "total_marks": total,
            "leaf_questions": len(self.leaves),
            "parts_per_question": int(round(np.median([max(m.n_parts for m in s) for s in slots]))) if slots else 0,
            "marks_per_question": float(np.median([x for x in marks_main if x])) if any(marks_main) else None,
            "numerical_leaves": fmts.get("numerical", 0),
            "derivation_leaves": fmts.get("derivation", 0),
            "theory_leaves": fmts.get("theory", 0) + fmts.get("definition", 0) + fmts.get("diagram", 0),
            "objective_leaves": fmts.get("objective", 0),
            "duration": self.duration,
        }


@dataclass
class StructureReport:
    per_exam: list[dict[str, Any]]
    patterns: list[dict[str, Any]]
    hypotheses: list[str]
    typical: dict[str, Any] = field(default_factory=dict)


def _classify(values: list[Any], recent: int) -> tuple[Any, float, str]:
    vals = [v for v in values if v is not None and v != ""]
    if not vals:
        return None, 0.0, "unknown"
    tail = vals[-recent:]
    mode, count = Counter(tail).most_common(1)[0]
    share = count / len(tail)
    if len(vals) >= 3 and share >= 0.8:
        level = "high-confidence historical pattern"
    elif share >= 0.6:
        level = "usual pattern"
    else:
        level = "variable"
    return mode, share, level


PROPERTY_LABELS = {
    "main_questions": "Number of main questions", "sections": "Number of sections/groups",
    "or_alternatives": "Questions with an OR alternative", "short_notes_question": "Has a short-notes question",
    "attempt_choice": "Students choose among questions", "total_marks": "Total / full marks",
    "parts_per_question": "Sub-parts per main question", "marks_per_question": "Marks per main question",
    "leaf_questions": "Answerable items (leaf questions)", "numerical_leaves": "Numerical items",
    "derivation_leaves": "Derivation items", "theory_leaves": "Theory items", "objective_leaves": "Objective items",
    "duration": "Duration",
}


def discover_structure(exams: list[ExamSummary]) -> StructureReport:
    exams = sorted(exams, key=lambda e: e.order)
    per_exam = [{"exam": e.label, **e.stats()} for e in exams]
    recent = min(5, len(exams)) or 1
    patterns, typical = [], {}
    for key, label in PROPERTY_LABELS.items():
        values = [row[key] for row in per_exam]
        mode, share, level = _classify(values, recent)
        if mode is None:
            continue
        typical[key] = mode
        patterns.append({"property": key, "label": label, "typical": mode, "share_recent": round(share, 2),
                         "level": level, "values": values})
    hypotheses = []
    for p in patterns:
        if p["level"] == "high-confidence historical pattern":
            hypotheses.append(f"Speculative: the next paper will probably keep '{p['label'].lower()}' at "
                              f"{_fmt(p['typical'])}, as in {int(p['share_recent'] * 100)}% of recent papers.")
    if not exams:
        hypotheses.append("No exams to analyse yet.")
    elif len(exams) < 3:
        hypotheses.append("Fewer than three exams: structural patterns cannot be established.")
    return StructureReport(per_exam, patterns, hypotheses, typical)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
