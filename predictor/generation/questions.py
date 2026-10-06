"""Grounded predicted question formulations (spec sections 27 and 28).

Formulations are assembled from three grounded parts and then checked:

1. an opener taken from this course's own historical questions of the same format
   ("Derive an expression for", "With a neat sketch, explain", ...);
2. a slot filled only with a phrase from the syllabus node or its sub-topics/concepts;
3. for numerical questions, a past numerical question on the topic reused verbatim
   (no new numbers are invented).

The grounding check rejects any formulation that contains a content word found neither in
the syllabus subtree nor in the topic's historical questions. Every formulation is labelled
"PREDICTED QUESTION FORMULATION" and lists the historical questions that shaped it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.settings import Settings
from ..preprocessing.textnorm import content_terms
from ..syllabus.tree import TopicTree

LABEL = "PREDICTED QUESTION FORMULATION"
OPENER_PATTERNS = {
    "definition": [r"^(define)\b", r"^(what (?:is|are) meant by)\b", r"^(what do you (?:mean|understand) by)\b"],
    "derivation": [r"^(derive an expression for)\b", r"^(derive the expression for)\b", r"^(derive)\b",
                   r"^(obtain (?:an |the )?expression for)\b", r"^(prove that)\b", r"^(show that)\b"],
    "theory": [r"^(with a neat sketch, explain)\b", r"^(explain with a neat sketch)\b", r"^(explain the working of)\b",
               r"^(explain the concept of)\b", r"^(explain)\b", r"^(describe)\b", r"^(discuss)\b",
               r"^(write short notes? on)\b"],
    "diagram": [r"^(with a neat sketch, explain)\b", r"^(draw)\b", r"^(sketch)\b"],
    "compare": [r"^(differentiate between)\b", r"^(distinguish between)\b", r"^(compare)\b"],
}
DEFAULT_OPENERS = {"definition": "Define", "derivation": "Derive", "theory": "Explain", "diagram": "With a neat sketch, explain",
                   "compare": "Differentiate between"}
DERIVABLE = re.compile(r"\b(equation|law|theorem|expression|formula|relation|condition|"
                       r"distribution|thickness|loss|rise|discharge|force|efficiency|height)\b", re.IGNORECASE)
# Words that name an aspect of a topic rather than a thing to ask about ("derivation, assumptions and
# applications"). A slot made only of these words is skipped.
ASPECT_WORDS = {"derivation", "derivations", "assumption", "assumptions", "application", "applications",
                "introduction", "overview", "definition", "definitions", "example", "examples", "numerical",
                "numericals", "problem", "problems", "type", "types", "property", "properties", "significance",
                "use", "uses", "limitation", "limitations", "advantage", "advantages", "disadvantage",
                "disadvantages", "importance", "basics", "basic", "concept", "concepts", "general", "characteristics",
                "classification", "measurement", "analysis", "theory", "method", "methods"}
TEMPLATE_WORDS = {"assumption", "assumptions", "made", "state", "neat", "sketch", "working", "concept", "brief",
                  "briefly", "expression", "explain", "derive", "define", "differentiate", "between", "meant",
                  "write", "short", "note", "notes", "with", "suitable", "example", "examples", "significance",
                  "applications", "application", "following", "its", "list"}


@dataclass
class HistoricalQuestion:
    id: int
    text: str
    format: str
    types: list[str]
    marks: float | None
    exam_label: str
    exam_index: int
    family: str | None = None


@dataclass
class Formulation:
    text: str
    format: str
    marks_low: float | None
    marks_high: float | None
    basis: str
    evidence_question_ids: list[int]
    grounding: dict[str, Any]
    note: str = ""
    label: str = LABEL
    rank: int = 0
    types: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "text": self.text, "format": self.format, "marks_low": self.marks_low,
                "marks_high": self.marks_high, "basis": self.basis, "evidence_question_ids": self.evidence_question_ids,
                "grounding": self.grounding, "note": self.note, "rank": self.rank}


def harvest_openers(history: list[HistoricalQuestion]) -> dict[str, Counter]:
    """Count the instruction openers this course actually uses, per format."""
    out: dict[str, Counter] = {f: Counter() for f in OPENER_PATTERNS}
    for q in history:
        text = q.text.strip().lower()
        for fmt, patterns in OPENER_PATTERNS.items():
            for p in patterns:
                m = re.match(p, text)
                if m:
                    out[fmt][m.group(1)] += 1
                    break
    return out


def _opener(openers: dict[str, Counter], fmt: str) -> str:
    counts = openers.get(fmt)
    if counts:
        phrase = counts.most_common(1)[0][0]
        return phrase[0].upper() + phrase[1:]
    return DEFAULT_OPENERS.get(fmt, "Explain")


def _clean_phrase(p: str) -> str:
    p = re.sub(r"\s+", " ", p).strip(" .,:;")
    p = re.sub(r"^(?:and|the|of)\s+", "", p, flags=re.IGNORECASE)
    # "Momentum equation and its applications" -> "Momentum equation"
    p = re.sub(r"\s+(?:and|&)\s+(?:its|their)\s+(?:applications?|uses?|significance|limitations?|control)$", "",
               p, flags=re.IGNORECASE)
    return p


def proper_nouns(texts: list[str]) -> set[str]:
    """Words written with a capital letter in the middle of a sentence (names such as Reynolds)."""
    out: set[str] = set()
    for text in texts:
        for sentence in re.split(r"(?<=[.?!:;])\s+", text):
            words = sentence.split()
            for w in words[1:]:
                core = re.sub(r"[^A-Za-z'\-]", "", w)
                if len(core) > 1 and core[0].isupper() and not core.isupper():
                    out.add(core.split("'")[0].split("-")[0].lower())
    return out


def display_phrase(phrase: str, names: set[str]) -> str:
    """Lower-case the first word of a syllabus phrase unless it is a name ("Bernoulli's equation")."""
    words = phrase.split()
    if not words:
        return phrase
    first = words[0]
    core = re.sub(r"[^A-Za-z'\-]", "", first)
    is_name = ("'" in core or "-" in core or core.isupper() or core.split("'")[0].lower() in names)
    if not is_name and len(core) > 1 and core[0].isupper() and core[1:].islower():
        words[0] = first[0].lower() + first[1:]
    return " ".join(words)


def _slots(tree: TopicTree, topic_id: int, topic_history: list[HistoricalQuestion]) -> list[str]:
    ids = [topic_id] + tree.descendants(topic_id)
    phrases: list[str] = []
    for nid in ids:
        node = tree.nodes[nid]
        for p in [node.title, *node.concepts]:
            p = _clean_phrase(p)
            # A title that is itself a list ("Darcy-Weisbach equation, friction factor, Moody diagram") is not a
            # single thing to ask about; its concepts are used instead.
            if "," in p or ";" in p:
                continue
            words = [w.lower() for w in re.findall(r"[A-Za-z]+", p)]
            if not words or all(w in ASPECT_WORDS or w in ("and", "or", "of", "the") for w in words):
                continue
            if 1 <= len(p.split()) <= 8 and p.lower() not in {x.lower() for x in phrases}:
                phrases.append(p)
    # Prefer phrases that past questions on this topic actually used, and specific multi-word phrases.
    hist_terms = Counter(t for q in topic_history for t in set(content_terms(q.text)))
    scored = []
    for i, p in enumerate(phrases):
        terms = content_terms(p)
        used = sum(hist_terms.get(t, 0) for t in terms) / max(len(terms), 1)
        specificity = 1.0 + 0.3 * min(len(p.split()), 4) if len(p.split()) > 1 else 0.6
        scored.append((-(used + 0.5) * specificity, i, p))
    return [p for _, _, p in sorted(scored)]


def _allowed_terms(tree: TopicTree, topic_id: int, topic_history: list[HistoricalQuestion]) -> set[str]:
    allowed: set[str] = set()
    for nid in [topic_id] + tree.descendants(topic_id) + tree.ancestors(topic_id):
        allowed |= set(content_terms(tree.term_text(nid)))
    for q in topic_history:
        allowed |= set(content_terms(q.text))
    return allowed


def check_grounding(text: str, allowed: set[str], instruction_words: list[str]) -> dict[str, Any]:
    terms = content_terms(text, list(instruction_words) + sorted(TEMPLATE_WORDS))
    unsupported = sorted({t for t in terms if t not in allowed})
    return {"grounded": not unsupported, "unsupported_terms": unsupported,
            "rule": "Every content word must appear in the syllabus subtree or the topic's past questions."}


def _marks_range(history: list[HistoricalQuestion], fmt: str, course: list[HistoricalQuestion]
                 ) -> tuple[float | None, float | None]:
    vals = [q.marks for q in history if q.format == fmt and q.marks]
    if len(vals) < 2:
        vals = [q.marks for q in course if q.format == fmt and q.marks]
    if not vals:
        return None, None
    lo, hi = np.percentile(vals, [25, 75])
    return float(round(lo)), float(round(hi))


def generate_formulations(tree: TopicTree, topic_id: int, topic_history: list[HistoricalQuestion],
                          course_history: list[HistoricalQuestion], format_forecast: dict[str, float] | None,
                          settings: Settings, limit: int | None = None) -> list[Formulation]:
    limit = limit or int(settings.generation.formulations_per_topic)
    instruction = list(settings.alignment.instruction_words)
    openers = harvest_openers(course_history)
    allowed = _allowed_terms(tree, topic_id, topic_history)
    slots = _slots(tree, topic_id, topic_history)
    if not slots:
        return []
    node = tree.nodes[topic_id]
    fmt_order = sorted((format_forecast or {}).items(), key=lambda kv: -kv[1])
    formats = [f for f, _ in fmt_order] or ["theory", "definition"]
    hist_formats = {q.format for q in topic_history}
    diagram_style = any(re.search(r"sketch|diagram", q.text, re.IGNORECASE) for q in topic_history)
    assumption_style = any("assumption" in q.text.lower() for q in topic_history if q.format == "derivation")
    names = proper_nouns([q.text for q in course_history])
    out: list[Formulation] = []
    used_slots: set[str] = set()

    def next_slot(predicate=None) -> str | None:
        for s in slots:
            if s in used_slots:
                continue
            if predicate is None or predicate(s):
                used_slots.add(s)
                return display_phrase(s, names)
        return None

    def same_format_evidence(fmt: str) -> list[int]:
        return [q.id for q in sorted(topic_history, key=lambda q: -q.exam_index) if q.format == fmt][:5]

    for fmt in formats + ["theory", "definition", "compare"]:
        if len(out) >= limit:
            break
        if any(f.format == fmt for f in out):
            continue
        text, basis, note, evidence = None, "template", "", same_format_evidence(fmt)
        if fmt == "numerical":
            past = [q for q in sorted(topic_history, key=lambda q: -q.exam_index) if q.format == "numerical"]
            if not past:
                continue
            q = past[0]
            text, basis = q.text, "historical_variant"
            note = (f"Pattern from {q.exam_label}. Numbers are copied from that paper; no new values were invented. "
                    f"Expect different data in a real paper.")
            evidence = [p.id for p in past[:5]]
        elif fmt == "derivation":
            derivable = "derivation" in node.kinds or "derivation" in hist_formats
            slot = next_slot(lambda s: bool(DERIVABLE.search(s))) if derivable or any(DERIVABLE.search(s) for s in slots) else None
            if not slot:
                continue
            opener = _opener(openers, "derivation")
            if opener.lower() == "derive" and not re.search(r"equation|law|theorem|formula", slot, re.IGNORECASE):
                opener = "Derive an expression for"
            text = f"{opener} {slot}."
            if assumption_style:
                text = text[:-1] + ". State the assumptions made."
        elif fmt == "definition":
            slot = next_slot()
            if not slot:
                continue
            opener = _opener(openers, "definition")
            text = f"{opener} {slot}." if opener.lower() == "define" else f"{opener} {slot}?"
        elif fmt in ("theory", "diagram", "mixed"):
            slot = next_slot()
            if not slot:
                continue
            if fmt == "diagram" or diagram_style:
                text = f"With a neat sketch, explain {slot}."
            else:
                text = f"{_opener(openers, 'theory')} {slot}."
            fmt = "diagram" if fmt == "diagram" else "theory"
        elif fmt == "compare":
            pair = [s for s in slots if s not in used_slots and s.lower() != node.title.lower()][:2]
            if len(pair) < 2:
                continue
            used_slots.update(pair)
            a, b = (display_phrase(x, names) for x in pair)
            text = f"{_opener(openers, 'compare')} {a} and {b}."
            fmt = "theory"
        else:
            continue
        grounding = check_grounding(text, allowed, instruction)
        if not grounding["grounded"]:
            continue
        lo, hi = _marks_range(topic_history, fmt, course_history)
        out.append(Formulation(text=text, format=fmt, marks_low=lo, marks_high=hi, basis=basis,
                               evidence_question_ids=evidence, grounding=grounding, note=note, rank=len(out) + 1))
    return out
