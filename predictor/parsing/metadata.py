"""Exam metadata: year (Gregorian or Bikram Sambat), session, exam type, marks, subject.

Every field carries a confidence and the text that supports it, so the review screen can
show why a value was chosen. Users can override any field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
# Bikram Sambat months in calendar order (Baishakh starts mid-April).
BS_MONTHS = {
    "baishakh": 1, "baisakh": 1, "baishak": 1, "baisak": 1, "vaisakh": 1,
    "jestha": 2, "jeth": 2, "jyeshtha": 2,
    "ashadh": 3, "asadh": 3, "asar": 3, "ashar": 3, "aashadh": 3,
    "shrawan": 4, "shravan": 4, "saun": 4, "srawan": 4,
    "bhadra": 5, "bhadau": 5,
    "ashwin": 6, "asoj": 6, "aswin": 6, "ashoj": 6,
    "kartik": 7, "kartika": 7,
    "mangsir": 8, "mangshir": 8, "marga": 8,
    "poush": 9, "paush": 9, "poos": 9,
    "magh": 10, "magha": 10,
    "falgun": 11, "phalgun": 11, "fagun": 11, "phagun": 11,
    "chaitra": 12,
}
SEASONS = {"winter": 0.05, "spring": 0.3, "summer": 0.55, "fall": 0.8, "autumn": 0.8}
EXAM_TYPES = [
    (r"\bback\b|\bre-?sit\b|\brepeat\b", "back"),
    (r"\bsupplementary\b|\bsupple\b", "supplementary"),
    (r"\bre-?exam(?:ination)?\b", "re-exam"),
    (r"\bmake-?up\b", "make-up"),
    (r"\bmid-?\s?term\b|\bmid[- ]?sem(?:ester)?\b", "mid-term"),
    (r"\bpre-?board\b", "pre-board"),
    (r"\bmodel (?:question|exam)\b", "model"),
    (r"\binternal\b|\bassessment\b", "internal"),
    (r"\bend[- ]?sem(?:ester)?\b|\bsemester[- ]end\b", "end-semester"),
    (r"\bfinal\b", "final"),
    (r"\bannual\b", "annual"),
    (r"\bregular\b", "regular"),
]
EXAM_WORDS = re.compile(
    r"exam|examination|year|session|semester|batch|held|regular|back|board|final|annual|"
    r"spring|fall|summer|winter|autumn|test|term|paper|question|sem\b|" +
    "|".join(sorted(set(MONTHS) | set(BS_MONTHS), key=len, reverse=True)), re.IGNORECASE)
YEAR_RE = re.compile(r"(?<![\d.])(19[5-9]\d|20\d\d)(?![\d.])")
ACADEMIC_RANGE_RE = re.compile(r"(?<!\d)(19[5-9]\d|20\d\d)\s*[/\-–]\s*(\d{2}|\d{4})(?!\d)")


@dataclass
class FieldValue:
    value: Any
    confidence: float
    evidence: str = ""


@dataclass
class ExamMetadata:
    fields: dict[str, FieldValue] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        fv = self.fields.get(name)
        return fv.value if fv is not None else default

    def set(self, name: str, value: Any, confidence: float, evidence: str = "") -> None:
        current = self.fields.get(name)
        if current is None or confidence > current.confidence:
            self.fields[name] = FieldValue(value, round(confidence, 2), evidence.strip()[:200])

    def confidence_map(self) -> dict[str, dict[str, Any]]:
        return {k: {"confidence": v.confidence, "evidence": v.evidence} for k, v in self.fields.items()}

    def as_dict(self) -> dict[str, Any]:
        return {k: v.value for k, v in self.fields.items()}


def _year_candidates(text: str, source: str, base: float, min_year: int, max_year: int) -> list[tuple[int, float, str]]:
    out: list[tuple[int, float, str]] = []
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        for m in YEAR_RE.finditer(line):
            year = int(m.group(1))
            if not (min_year <= year <= max_year):
                continue
            score = base
            if EXAM_WORDS.search(line):
                score += 0.25
            before = line[max(0, m.start() - 12):m.start()].lower()
            after = line[m.end():m.end() + 10].lower()
            if re.search(r"[a-z]{2,4}\s*-?\s*$", before) and not EXAM_WORDS.search(before):
                score -= 0.35  # course codes such as "CH 2051" or "ENG2019"
            if re.search(r"marks|hrs|hours|min", after):
                score -= 0.4
            if re.search(r"syllabus|revised|est\.?|established|since|copyright|©", line, re.IGNORECASE):
                score -= 0.4
            score -= min(idx, 30) * 0.004  # earlier lines are more likely to be the exam header
            out.append((year, score, line.strip()))
    return out


def detect_year(header: str, filename: str, *, min_year: int, max_year: int,
                today: date | None = None) -> tuple[int | None, float, str, str]:
    """Return (year, confidence, calendar, evidence). Calendar is AD, BS or unknown."""
    today = today or date.today()
    cands = _year_candidates(header, "header", 0.55, min_year, max_year)
    stem = re.sub(r"[_\-.]+", " ", filename)
    for year, score, _ in _year_candidates(stem, "filename", 0.6, min_year, max_year):
        cands.append((year, score + 0.15, f"file name: {filename}"))
    if not cands:
        return None, 0.0, "unknown", ""
    # Agreement between several mentions raises confidence.
    totals: dict[int, float] = {}
    best_evidence: dict[int, tuple[float, str]] = {}
    for year, score, ev in cands:
        totals[year] = totals.get(year, 0.0) + max(score, 0.0)
        if year not in best_evidence or score > best_evidence[year][0]:
            best_evidence[year] = (score, ev)
    year = max(totals, key=lambda y: (totals[y], best_evidence[y][0]))
    top = best_evidence[year][0]
    agreement = min(totals[year] / max(sum(totals.values()), 1e-9), 1.0)
    confidence = max(0.05, min(0.98, 0.5 * top + 0.5 * agreement))
    calendar = "AD"
    lowered = (header + " " + filename).lower()
    has_bs_month = any(re.search(rf"\b{m}\b", lowered) for m in BS_MONTHS)
    if year > today.year + 1 or (year >= 2040 and has_bs_month):
        calendar = "BS"
    return year, confidence, calendar, best_evidence[year][1]


def bs_to_ad_approx(year: int) -> float:
    """Bikram Sambat year to an approximate Gregorian year (BS starts mid-April)."""
    return year - 56.7


def detect_session(text: str) -> tuple[str, float, float, str]:
    """Return (label, order_fraction within year, confidence, evidence)."""
    lowered = text.lower()
    for name, idx in BS_MONTHS.items():
        m = re.search(rf"\b{name}\b", lowered)
        if m:
            return name.capitalize(), (idx - 0.5) / 12.0, 0.8, _line_at(text, m.start())
    for name, frac in SEASONS.items():
        m = re.search(rf"\b{name}\b", lowered)
        if m:
            return name.capitalize(), frac, 0.8, _line_at(text, m.start())
    for name, idx in MONTHS.items():
        m = re.search(rf"\b{name}\b\.?(?:\s*,?\s*(?:19|20)\d\d)", lowered)
        if m:
            label = [k for k, v in MONTHS.items() if v == idx and len(k) > 3]
            return (label[0] if label else name).capitalize(), (idx - 0.5) / 12.0, 0.75, _line_at(text, m.start())
    m = re.search(r"\b(odd|even)\s+semester\b|\bsemester\s*[-:]?\s*(i{1,3}|iv|v|vi{0,3}|[1-8])\b", lowered)
    if m:
        return m.group(0).strip().title(), 0.5, 0.5, _line_at(text, m.start())
    return "", 0.5, 0.0, ""


def _line_at(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start:end if end != -1 else None]


def detect_exam_type(text: str) -> tuple[str, float, str]:
    lowered = text.lower()
    for pattern, label in EXAM_TYPES:
        m = re.search(pattern, lowered)
        if m:
            return label, 0.75, _line_at(text, m.start())
    return "", 0.0, ""


def _number_after(pattern: str, text: str) -> tuple[float | None, str]:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None, ""
    try:
        return float(m.group(1)), _line_at(text, m.start())
    except (ValueError, IndexError):
        return None, ""


def extract_metadata(header: str, full_text: str, filename: str, *, min_year: int = 1950,
                     max_year: int = 2100, today: date | None = None) -> ExamMetadata:
    md = ExamMetadata()
    scope = header if header.strip() else full_text[:2500]

    year, conf, calendar, ev = detect_year(scope, filename, min_year=min_year, max_year=max_year, today=today)
    if year is None and scope is not full_text:
        year, conf, calendar, ev = detect_year(full_text[:6000], filename, min_year=min_year,
                                               max_year=max_year, today=today)
        conf *= 0.7
    if year is not None:
        md.set("year", year, conf, ev)
        md.set("calendar", calendar, conf if calendar != "unknown" else 0.0, ev)
        range_match = ACADEMIC_RANGE_RE.search(scope)
        if range_match and int(range_match.group(1)) == year:
            md.set("academic_year", range_match.group(0), 0.7, _line_at(scope, range_match.start()))

    session, frac, sconf, sev = detect_session(scope + "\n" + re.sub(r"[_\-.]+", " ", filename))
    if session:
        md.set("session", session, sconf, sev)
    md.set("session_fraction", frac, sconf, sev)

    etype, econf, eev = detect_exam_type(scope + "\n" + filename)
    if etype:
        md.set("exam_type", etype, econf, eev)

    fm, fev = _number_after(r"(?:full|max(?:imum)?|total)\.?\s*marks?\s*[:=\-]?\s*(\d{1,3})", scope)
    if fm is None:
        fm, fev = _number_after(r"\bF\.?\s?M\.?\s*[:=\-]?\s*(\d{1,3})\b", scope)
    if fm is not None:
        md.set("full_marks", fm, 0.85, fev)
    pm, pev = _number_after(r"pass\.?\s*marks?\s*[:=\-]?\s*(\d{1,3})", scope)
    if pm is None:
        pm, pev = _number_after(r"\bP\.?\s?M\.?\s*[:=\-]?\s*(\d{1,3})\b", scope)
    if pm is not None:
        md.set("pass_marks", pm, 0.85, pev)

    m = re.search(r"(?:time|duration)\s*(?:allowed)?\s*[:=\-]?\s*(\d+(?:\.\d+)?\s*(?:hrs?\.?|hours?|minutes?|mins?\.?))",
                  scope, re.IGNORECASE)
    if m:
        md.set("duration", m.group(1).strip(), 0.85, _line_at(scope, m.start()))

    m = re.search(r"(?:subject|course(?:\s*title)?|paper)\s*[:\-]+\s*(.+)", scope, re.IGNORECASE)
    if m:
        subject = re.split(r"\s{3,}|\s+(?:full|pass|time)\s*marks?", m.group(1), flags=re.IGNORECASE)[0]
        md.set("subject", subject.strip(" -:"), 0.8, _line_at(scope, m.start()))
    m = re.search(r"(?:course|subject|paper)\s*code\s*[:\-]?\s*([A-Z]{2,6}\s*-?\s*\d{2,4}[A-Z]?)", scope, re.IGNORECASE)
    if not m:
        m = re.search(r"\(([A-Z]{2,6}\s*-?\s*\d{3,4}[A-Z]?)\)", scope)
    if m:
        md.set("course_code", m.group(1).replace(" ", ""), 0.75, _line_at(scope, m.start()))

    m = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.]((?:19|20)\d\d)\b", scope)
    if m:
        md.set("exam_date", m.group(0), 0.6, _line_at(scope, m.start()))

    first_lines = [ln.strip() for ln in header.split("\n") if ln.strip()][:3]
    if first_lines:
        md.set("title", " | ".join(first_lines)[:300], 0.5, first_lines[0])

    m = re.search(r"(?:examiner|set by|paper setter)\s*[:\-]\s*(.+)", scope, re.IGNORECASE)
    if m:
        md.set("examiner", m.group(1).strip()[:120], 0.6, _line_at(scope, m.start()))
    return md


def order_index_for(year: int | None, calendar: str, session_fraction: float, exam_type: str = "") -> float:
    """Sortable time position for an exam. Regular exams sort before back exams of the same session."""
    if year is None:
        return 0.0
    base = bs_to_ad_approx(year) if calendar == "BS" else float(year)
    tweak = 0.01 if exam_type in {"back", "supplementary", "re-exam", "make-up"} else 0.0
    return round(base + min(max(session_fraction, 0.0), 0.99) * 0.98 + tweak, 4)
